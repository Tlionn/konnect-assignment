from pathlib import Path
import pandas as pd

from src.poi_intelligence.data import generate_synthetic_data
from src.poi_intelligence.pipeline import POIRecommendationPipeline


def _pipeline(tmp_path):
    generate_synthetic_data(tmp_path, seed=7)
    pois=pd.read_csv(tmp_path/"pois.csv")
    travelers=pd.read_csv(tmp_path/"travelers.csv")
    interactions=pd.read_csv(tmp_path/"interactions.csv")
    return POIRecommendationPipeline(random_state=7).fit(travelers,pois,interactions), travelers


def test_recommendation_schema_and_order(tmp_path):
    pipe,_=_pipeline(tmp_path)
    r=pipe.recommend("U001",k=10)
    assert len(r)==10
    assert r.final_score.is_monotonic_decreasing
    for col in ["preference_score","context_compatibility","confidence","explanation"]:
        assert col in r.columns


def test_profiles_are_personalized(tmp_path):
    pipe,_=_pipeline(tmp_path)
    a=set(pipe.recommend("U001",k=10).poi_id)
    b=set(pipe.recommend("U002",k=10).poi_id)
    assert a != b
    assert len(a & b) < 10


def test_long_tail_lane_survives_candidate_generation(tmp_path):
    pipe,travelers=_pipeline(tmp_path)
    t=travelers.iloc[0]
    c=pipe._candidate_generation(t,max_candidates=40)
    assert (c.popularity < 0.35).any()


def test_evaluation_metrics_in_range(tmp_path):
    pipe,_=_pipeline(tmp_path)
    m,_=pipe.evaluate(k=10)
    for key,value in m.items():
        if key != "evaluated_users":
            assert 0.0 <= value <= 1.0
