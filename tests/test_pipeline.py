import pandas as pd

from src.poi_intelligence.data import generate_synthetic_data
from src.poi_intelligence.pipeline import POIRecommendationPipeline


def _data(tmp_path, seed=7):
    generate_synthetic_data(tmp_path, seed=seed)
    pois = pd.read_csv(tmp_path / "pois.csv")
    travelers = pd.read_csv(tmp_path / "travelers.csv")
    interactions = pd.read_csv(tmp_path / "interactions.csv")
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    return travelers, pois, interactions


def _pipeline(tmp_path):
    travelers, pois, interactions = _data(tmp_path)
    return POIRecommendationPipeline(random_state=7).fit(travelers, pois, interactions), travelers, interactions


def test_recommendation_schema_and_order(tmp_path):
    pipe, _, _ = _pipeline(tmp_path)
    r = pipe.recommend("U001", k=10)
    assert len(r) == 10
    assert r.final_score.is_monotonic_decreasing
    for col in ["preference_score", "context_compatibility", "confidence", "explanation"]:
        assert col in r.columns


def test_profiles_are_personalized(tmp_path):
    pipe, _, _ = _pipeline(tmp_path)
    a = set(pipe.recommend("U001", k=10).poi_id)
    b = set(pipe.recommend("U002", k=10).poi_id)
    assert a != b
    assert len(a & b) < 10


def test_long_tail_lane_survives_candidate_generation(tmp_path):
    pipe, travelers, _ = _pipeline(tmp_path)
    t = travelers.iloc[0]
    c = pipe._candidate_generation(t, max_candidates=40)
    assert (c.popularity < 0.35).any()


def test_excluded_seen_items_do_not_reappear(tmp_path):
    pipe, _, interactions = _pipeline(tmp_path)
    seen = set(interactions[interactions.traveler_id == "U001"].poi_id.head(8))
    r = pipe.recommend("U001", k=10, exclude_poi_ids=seen)
    assert not (set(r.poi_id) & seen)


def test_holdout_evaluation_beats_popularity_baseline(tmp_path):
    travelers, pois, interactions = _data(tmp_path, seed=42)
    train, test = [], []
    for _, grp in interactions.sort_values("timestamp").groupby("traveler_id"):
        cut = int(len(grp) * 0.75)
        train.append(grp.iloc[:cut])
        test.append(grp.iloc[cut:])
    train = pd.concat(train, ignore_index=True)
    test = pd.concat(test, ignore_index=True)
    pipe = POIRecommendationPipeline(random_state=42).fit(travelers, pois, train)
    metrics, _ = pipe.evaluate(k=10, eval_interactions=test)
    assert metrics["candidate_recall"] >= 0.95
    assert metrics["ndcg_at_10"] > metrics["popularity_baseline_ndcg_at_10"]
    assert metrics["ndcg_lift_vs_popularity"] > 0.25


def test_evaluation_metrics_in_range(tmp_path):
    pipe, _, _ = _pipeline(tmp_path)
    m, _ = pipe.evaluate(k=10)
    for key, value in m.items():
        if key in {"evaluated_users", "avg_relevant_items_per_test_user"}:
            continue
        assert 0.0 <= value <= 1.0
