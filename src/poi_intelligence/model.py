from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import empty_profile, pair_features


class _RollingProfile:
    def __init__(self):
        self.cat = {}
        self.tag = {}
        self.local_sum = 0.0
        self.pop_sum = 0.0
        self.weight_sum = 0.0
        self.positive_count = 0

    def snapshot(self):
        if self.weight_sum <= 0:
            return empty_profile()
        cat_total = max(sum(self.cat.values()), 1e-9)
        tag_total = max(sum(self.tag.values()), 1e-9)
        return {
            "category_affinity": {k: v / cat_total for k, v in self.cat.items()},
            "tag_affinity": {k: v / tag_total for k, v in self.tag.items()},
            "local_affinity": self.local_sum / self.weight_sum,
            "popularity_affinity": self.pop_sum / self.weight_sum,
            "history_strength": min(1.0, self.positive_count / 8.0),
        }

    def update(self, poi: pd.Series, signal: float):
        if signal < 0.6:
            return
        w = float(signal)
        self.cat[poi.category] = self.cat.get(poi.category, 0.0) + w
        for tag in str(poi.tags).split("|"):
            self.tag[tag] = self.tag.get(tag, 0.0) + w
        self.local_sum += w * float(poi.localness)
        self.pop_sum += w * float(poi.popularity)
        self.weight_sum += w
        self.positive_count += 1


class PreferenceRanker:
    """Hybrid ranker: learned high-intent probability + transparent content prior.

    Training features use only behavior that occurred before each interaction, avoiding
    target leakage from full-history aggregates.
    """

    def __init__(self, random_state: int = 42):
        self.feature_names = None
        self.model = HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=0.5,
            random_state=random_state,
        )

    def fit(self, travelers: pd.DataFrame, pois: pd.DataFrame, interactions: pd.DataFrame):
        t_index = travelers.set_index("traveler_id")
        p_index = pois.set_index("poi_id")
        rows, targets, weights = [], [], []
        ordered = interactions.copy()
        ordered["timestamp"] = pd.to_datetime(ordered["timestamp"])
        ordered = ordered.sort_values(["traveler_id", "timestamp"])

        for uid, grp in ordered.groupby("traveler_id", sort=False):
            profile = _RollingProfile()
            t = t_index.loc[uid]
            for _, inter in grp.iterrows():
                p = p_index.loc[inter.poi_id]
                rows.append(pair_features(t, p, profile.snapshot()))
                y = int(float(inter.signal) >= 0.6)
                targets.append(y)
                if float(inter.signal) < 0:
                    weights.append(1.5)
                elif y:
                    weights.append(1.0 + 0.5 * float(inter.signal))
                else:
                    weights.append(0.75)
                profile.update(p, float(inter.signal))

        x = pd.DataFrame(rows)
        self.feature_names = list(x.columns)
        self.model.fit(x[self.feature_names], np.asarray(targets), sample_weight=np.asarray(weights))
        return self

    def predict(self, traveler: pd.Series, candidates: pd.DataFrame, history_profile: dict | None = None):
        feats = pd.DataFrame([pair_features(traveler, p, history_profile) for _, p in candidates.iterrows()])
        learned = self.model.predict_proba(feats[self.feature_names])[:, 1]
        prior = (
            0.42 * feats["interest_match"].to_numpy(float)
            + 0.18 * feats["preference_match"].to_numpy(float)
            + 0.10 * feats["explicit_local_x_localness"].to_numpy(float)
            + 0.08 * feats["explicit_famous_x_popularity"].to_numpy(float)
            + 0.10 * feats["historical_category_affinity"].to_numpy(float)
            + 0.04 * feats["historical_tag_affinity"].to_numpy(float)
            + 0.05 * feats["rating_norm"].to_numpy(float)
            + 0.03 * feats["history_local_similarity"].to_numpy(float)
        )
        # Explicit/content evidence is intentionally dominant in this sparse prototype;
        # the learned component adds nonlinear behavioral refinement.
        preference = np.clip(0.30 * learned + 0.70 * prior, 0.0, 1.0)
        return preference, feats
