from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CATEGORIES = {
    "food": ["market", "street_food", "restaurant", "cafe"],
    "history": ["palace", "museum", "heritage", "monument"],
    "culture": ["gallery", "craft", "performance", "temple"],
    "nature": ["park", "trail", "river", "garden"],
    "family": ["science", "zoo", "aquarium", "playground"],
    "shopping": ["local_market", "mall", "design_store", "souvenir"],
    "neighborhood": ["walking_area", "local_alley", "village", "waterfront"],
}

DESTINATIONS = {
    "Seoul": (37.5665, 126.9780),
    "Busan": (35.1796, 129.0756),
}


def _choice(rng: np.random.Generator, values: Iterable):
    values = list(values)
    return values[int(rng.integers(0, len(values)))]


def generate_synthetic_data(output_dir: str | Path, seed: int = 42) -> dict[str, Path]:
    """Create deterministic POI, traveler and implicit-feedback datasets."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    pois = []
    poi_id = 1
    for destination, (base_lat, base_lon) in DESTINATIONS.items():
        for category, subs in CATEGORIES.items():
            for idx in range(12):
                sub = _choice(rng, subs)
                localness = float(np.clip(rng.beta(2.2, 2.0), 0.05, 0.98))
                popularity = float(np.clip(rng.beta(1.5, 2.3), 0.02, 0.99))
                family_friendly = float(np.clip(rng.beta(2, 2), 0.05, 0.98))
                accessibility = float(np.clip(rng.beta(3, 1.5), 0.1, 1.0))
                price_level = int(rng.integers(1, 5))
                rating = round(float(rng.uniform(3.4, 4.9)), 2)
                review_count = int(np.exp(rng.uniform(np.log(20), np.log(6000))))
                duration = int(rng.choice([45, 60, 75, 90, 120, 150, 180]))
                transit_score = float(np.clip(rng.beta(3, 1.6), 0.05, 1.0))
                child_score = max(family_friendly, 0.75 if category == "family" else 0.0)
                reservation_required = bool(rng.random() < (0.25 if category == "food" else 0.08))
                open_weekend = bool(rng.random() > 0.08)
                tags = [category, sub]
                if localness > 0.68:
                    tags += ["local", "less_touristy"]
                if popularity > 0.75:
                    tags += ["famous", "landmark"]
                if child_score > 0.72:
                    tags += ["child_friendly", "family"]
                if accessibility > 0.8:
                    tags += ["accessible"]
                if transit_score > 0.75:
                    tags += ["public_transport"]

                pois.append({
                    "poi_id": f"P{poi_id:04d}",
                    "destination": destination,
                    "name": f"{destination} {sub.replace('_', ' ').title()} {idx + 1}",
                    "category": category,
                    "subcategory": sub,
                    "latitude": round(base_lat + float(rng.normal(0, 0.055)), 6),
                    "longitude": round(base_lon + float(rng.normal(0, 0.055)), 6),
                    "description": f"A {sub.replace('_',' ')} experience in {destination} with {category} focus.",
                    "tags": "|".join(sorted(set(tags))),
                    "price_level": price_level,
                    "rating": rating,
                    "review_count": review_count,
                    "popularity": round(popularity, 4),
                    "localness": round(localness, 4),
                    "family_friendly": round(child_score, 4),
                    "transit_score": round(transit_score, 4),
                    "accessibility_score": round(accessibility, 4),
                    "expected_duration_min": duration,
                    "reservation_required": reservation_required,
                    "open_weekend": open_weekend,
                })
                poi_id += 1
    pois_df = pd.DataFrame(pois)

    traveler_templates = [
        ("food_local", ["food", "neighborhood"], "less_touristy|local", 2, "public_transport", "solo"),
        ("history", ["history", "culture"], "famous|architecture", 3, "public_transport", "couple"),
        ("family", ["family", "nature"], "child_friendly|accessible", 2, "car", "family"),
        ("culture", ["culture", "history"], "local|craft", 2, "walking", "solo"),
        ("nature", ["nature", "neighborhood"], "less_touristy|outdoor", 1, "walking", "couple"),
        ("shopping_food", ["shopping", "food"], "local|market", 3, "public_transport", "friends"),
    ]
    travelers = []
    for i in range(90):
        archetype, interests, prefs, budget, mobility, party = traveler_templates[i % len(traveler_templates)]
        destination = "Seoul" if i % 3 else "Busan"
        travelers.append({
            "traveler_id": f"U{i+1:03d}",
            "destination": destination,
            "trip_duration_days": int(rng.integers(2, 8)),
            "interests": "|".join(interests),
            "budget_level": int(np.clip(budget + rng.integers(-1, 2), 1, 4)),
            "party_type": party,
            "mobility": mobility,
            "explicit_preferences": prefs,
            "archetype": archetype,
        })
    travelers_df = pd.DataFrame(travelers)

    event_strength = {"view": 0.15, "click": 0.3, "save": 0.6, "navigate": 0.75, "visit": 0.9, "booking": 1.0, "dismiss": -0.7}
    positive_events = ["view", "click", "save", "navigate", "visit", "booking"]
    interactions = []
    ts0 = pd.Timestamp("2026-01-01")

    for _, t in travelers_df.iterrows():
        dest = pois_df[pois_df.destination == t.destination]
        interests = set(t.interests.split("|"))
        prefs = set(t.explicit_preferences.split("|"))
        scored = []
        for _, p in dest.iterrows():
            tags = set(p.tags.split("|"))
            interest_match = 1.0 if p.category in interests else 0.15
            preference_match = len(tags & prefs) / max(1, len(prefs))
            budget_fit = max(0.0, 1.0 - 0.28 * max(0, p.price_level - t.budget_level))
            mobility_fit = p.transit_score if t.mobility == "public_transport" else (p.accessibility_score if t.mobility == "walking" else 0.9)
            family_fit = p.family_friendly if t.party_type == "family" else 0.6
            local_bonus = p.localness if "less_touristy" in prefs or "local" in prefs else 0.25 * p.popularity
            utility = 0.33*interest_match + 0.2*preference_match + 0.12*budget_fit + 0.1*mobility_fit + 0.1*family_fit + 0.1*local_bonus + 0.05*p.rating/5
            utility += float(rng.normal(0, 0.06))
            scored.append((p.poi_id, utility))
        scored.sort(key=lambda x: x[1], reverse=True)
        tail_pool = scored[14:]
        sample_idx = rng.choice(len(tail_pool), size=6, replace=False)
        selected = scored[:14] + [tail_pool[int(i)] for i in sample_idx]
        for j, (pid, utility) in enumerate(selected):
            prob = 1/(1+np.exp(-7*(utility-0.58)))
            if rng.random() < prob:
                event = _choice(rng, positive_events if utility > 0.55 else ["view", "click", "dismiss"])
            else:
                event = _choice(rng, ["view", "dismiss", "click"])
            interactions.append({
                "traveler_id": t.traveler_id,
                "poi_id": pid,
                "interaction": event,
                "signal": event_strength[event],
                "timestamp": (ts0 + pd.Timedelta(days=int(rng.integers(0, 240)), hours=int(rng.integers(0, 24)))).isoformat(),
            })
    interactions_df = pd.DataFrame(interactions)

    paths = {
        "pois": out / "pois.csv",
        "travelers": out / "travelers.csv",
        "interactions": out / "interactions.csv",
    }
    pois_df.to_csv(paths["pois"], index=False)
    travelers_df.to_csv(paths["travelers"], index=False)
    interactions_df.to_csv(paths["interactions"], index=False)
    with open(out / "data_dictionary.json", "w", encoding="utf-8") as f:
        json.dump({
            "poi_rows": len(pois_df),
            "traveler_rows": len(travelers_df),
            "interaction_rows": len(interactions_df),
            "interaction_signal": event_strength,
            "seed": seed,
        }, f, indent=2)
    return paths
