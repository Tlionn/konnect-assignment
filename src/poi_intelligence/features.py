from __future__ import annotations

import numpy as np
import pandas as pd


def _set(v):
    if pd.isna(v) or not str(v):
        return set()
    return set(str(v).split("|"))


def empty_profile():
    return {"category_affinity": {}, "tag_affinity": {}, "local_affinity": 0.5, "popularity_affinity": 0.5, "history_strength": 0.0}


def build_user_history_profiles(travelers: pd.DataFrame, pois: pd.DataFrame, interactions: pd.DataFrame):
    merged = interactions.merge(pois[["poi_id","category","tags","localness","popularity"]], on="poi_id", how="left")
    profiles = {}
    for uid, grp in merged.groupby("traveler_id"):
        # Strong positive events shape long-term preference; weak views/clicks do not.
        pos = grp[grp.signal >= 0.6].copy()
        if pos.empty:
            profiles[uid] = empty_profile()
            continue
        weights = pos.signal.to_numpy(float)
        cat, tag = {}, {}
        for (_, row), w in zip(pos.iterrows(), weights):
            cat[row.category] = cat.get(row.category, 0.0) + float(w)
            for tg in _set(row.tags):
                tag[tg] = tag.get(tg, 0.0) + float(w)
        norm = max(sum(cat.values()), 1e-9)
        cat = {k: v/norm for k,v in cat.items()}
        tnorm = max(sum(tag.values()), 1e-9)
        tag = {k: v/tnorm for k,v in tag.items()}
        profiles[uid] = {
            "category_affinity": cat,
            "tag_affinity": tag,
            "local_affinity": float(np.average(pos.localness, weights=weights)),
            "popularity_affinity": float(np.average(pos.popularity, weights=weights)),
            "history_strength": min(1.0, len(pos)/8.0),
        }
    return profiles


def pair_features(traveler: pd.Series, poi: pd.Series, history_profile: dict | None = None) -> dict:
    interests = _set(traveler.interests)
    prefs = _set(traveler.explicit_preferences)
    tags = _set(poi.tags)
    interest_match = 1.0 if poi.category in interests else 0.0
    preference_match = len(tags & prefs) / max(1, len(prefs))
    budget_gap = max(0, int(poi.price_level) - int(traveler.budget_level))
    budget_fit = max(0.0, 1.0 - 0.3*budget_gap)
    if traveler.mobility == "public_transport":
        mobility_fit = float(poi.transit_score)
    elif traveler.mobility == "walking":
        mobility_fit = 0.55*float(poi.accessibility_score) + 0.45*float(poi.transit_score)
    else:
        mobility_fit = 0.9
    family_fit = float(poi.family_friendly) if traveler.party_type == "family" else 0.7
    long_tail = 1.0 - float(poi.popularity)
    explicit_local = 1.0 if ({"local", "less_touristy"} & prefs) else 0.0
    explicit_famous = 1.0 if ({"famous", "landmark"} & prefs) else 0.0
    history_profile = history_profile or empty_profile()
    hist_cat = float(history_profile.get("category_affinity", {}).get(poi.category, 0.0))
    hist_tag = sum(float(history_profile.get("tag_affinity", {}).get(t, 0.0)) for t in tags)
    hist_strength = float(history_profile.get("history_strength", 0.0))
    local_affinity = 1.0 - abs(float(poi.localness)-float(history_profile.get("local_affinity", 0.5)))
    popularity_affinity = 1.0 - abs(float(poi.popularity)-float(history_profile.get("popularity_affinity", 0.5)))
    return {
        "interest_match": interest_match,
        "preference_match": preference_match,
        "budget_fit": budget_fit,
        "mobility_fit": mobility_fit,
        "family_fit": family_fit,
        "rating_norm": float(poi.rating)/5.0,
        "review_log_norm": min(1.0, np.log1p(float(poi.review_count))/np.log1p(6000)),
        "popularity": float(poi.popularity),
        "localness": float(poi.localness),
        "long_tail": long_tail,
        "explicit_local_x_localness": explicit_local*float(poi.localness),
        "explicit_famous_x_popularity": explicit_famous*float(poi.popularity),
        "historical_category_affinity": hist_cat,
        "historical_tag_affinity": min(1.0, hist_tag),
        "history_strength": hist_strength,
        "history_local_similarity": local_affinity*hist_strength,
        "history_popularity_similarity": popularity_affinity*hist_strength,
        "interest_x_history": interest_match * (0.25 + hist_cat),
        "duration_fit": 1.0 if float(poi.expected_duration_min) <= max(120, float(traveler.trip_duration_days)*90) else 0.6,
        "reservation_friction": 0.0 if not bool(poi.reservation_required) else 1.0,
        "weekend_open": 1.0 if bool(poi.open_weekend) else 0.0,
        "accessibility": float(poi.accessibility_score),
    }
