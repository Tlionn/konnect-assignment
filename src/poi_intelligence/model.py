from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_user_history_profiles, pair_features


class PreferenceRanker:
    def __init__(self, random_state: int = 42):
        self.feature_names = None
        self.model = Pipeline([
            ("scale", StandardScaler()),
            ("ranker", HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=0.2, random_state=random_state)),
        ])

    def fit(self, travelers: pd.DataFrame, pois: pd.DataFrame, interactions: pd.DataFrame):
        profiles = build_user_history_profiles(travelers, pois, interactions)
        t_index = travelers.set_index("traveler_id")
        p_index = pois.set_index("poi_id")
        rows, targets = [], []
        for _, inter in interactions.iterrows():
            t = t_index.loc[inter.traveler_id]
            p = p_index.loc[inter.poi_id]
            rows.append(pair_features(t, p, profiles.get(inter.traveler_id)))
            targets.append(float(inter.signal))
        x = pd.DataFrame(rows)
        self.feature_names = list(x.columns)
        self.model.fit(x[self.feature_names], np.asarray(targets))
        return self

    def predict(self, traveler: pd.Series, candidates: pd.DataFrame, history_profile: dict | None = None):
        feats = pd.DataFrame([pair_features(traveler, p, history_profile) for _, p in candidates.iterrows()])
        raw = self.model.predict(feats[self.feature_names])
        preference = 1/(1+np.exp(-3*raw))
        return preference, feats
