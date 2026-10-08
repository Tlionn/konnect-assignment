from pathlib import Path
import pandas as pd
import pytest
from src.poi_intelligence.pipeline import POIRecommendationPipeline

ROOT=Path(__file__).resolve().parents[1]

def _load():
    pois=pd.read_csv(ROOT/'data/pois.csv'); travelers=pd.read_csv(ROOT/'data/travelers.csv'); interactions=pd.read_csv(ROOT/'data/interactions.csv'); interactions['timestamp']=pd.to_datetime(interactions.timestamp)
    train=[];test=[]
    for _,g in interactions.sort_values('timestamp').groupby('traveler_id'):
        c=int(len(g)*.75);train.append(g.iloc[:c]);test.append(g.iloc[c:])
    return travelers,pois,pd.concat(train,ignore_index=True),pd.concat(test,ignore_index=True)

@pytest.fixture(scope='module')
def fitted():
    travelers,pois,train,test=_load(); return POIRecommendationPipeline(42).fit(travelers,pois,train),travelers,train,test

def test_recommendation_schema_and_order(fitted):
    pipe,_,_,_=fitted; r=pipe.recommend(traveler_id='U001',k=10)
    assert len(r)==10 and r.final_score.is_monotonic_decreasing
    for c in ['preference_score','context_compatibility','confidence','explanation']: assert c in r.columns

def test_preference_ranking_is_ordered(fitted):
    pipe,_,_,_=fitted; r=pipe.recommend(traveler_id='U001',k=10,ranking='preference')
    assert r.preference_score.is_monotonic_decreasing

def test_seen_items_are_excluded(fitted):
    pipe,_,train,_=fitted; seen=set(train[train.traveler_id=='U001'].poi_id)
    r=pipe.recommend(traveler_id='U001',k=10,exclude_poi_ids=seen)
    assert not (set(r.poi_id)&seen)

def test_candidate_generation_preserves_long_tail(fitted):
    pipe,travelers,_,_=fitted; c=pipe._candidate_generation(travelers.iloc[0],max_candidates=40)
    assert (c.popularity<.35).any()

def test_holdout_beats_popularity_and_has_high_candidate_recall(fitted):
    pipe,_,_,test=fitted; m,_=pipe.evaluate(10,test)
    assert m['candidate_recall']>=.95
    assert m['ndcg_at_10']>m['popularity_baseline_ndcg_at_10']+.50
    assert m['recall_at_10']>=.85

def test_cold_start_traveler_can_be_ranked(fitted):
    pipe,_,_,_=fitted
    new_user={'traveler_id':'NEW','destination':'Seoul','trip_duration_days':4,'interests':'history|culture','budget_level':2,'party_type':'couple','mobility':'public_transport','explicit_preferences':'local|craft','archetype':'cold_start'}
    r=pipe.recommend(traveler_override=new_user,k=5)
    assert len(r)==5 and r.final_score.notna().all()
