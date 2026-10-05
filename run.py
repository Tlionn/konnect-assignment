from pathlib import Path
import json
import pandas as pd

from src.poi_intelligence.data import generate_synthetic_data
from src.poi_intelligence.pipeline import POIRecommendationPipeline

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
REPORTS=ROOT/"reports"

def main():
    generate_synthetic_data(DATA, seed=42)
    pois=pd.read_csv(DATA/"pois.csv")
    travelers=pd.read_csv(DATA/"travelers.csv")
    interactions=pd.read_csv(DATA/"interactions.csv")
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    train_parts, test_parts = [], []
    for _, grp in interactions.sort_values("timestamp").groupby("traveler_id"):
        cut = max(1, int(len(grp) * 0.75))
        train_parts.append(grp.iloc[:cut])
        test_parts.append(grp.iloc[cut:])
    train_interactions = pd.concat(train_parts, ignore_index=True)
    test_interactions = pd.concat(test_parts, ignore_index=True)
    pipe=POIRecommendationPipeline(random_state=42).fit(travelers,pois,train_interactions)
    metrics=pipe.save_report(REPORTS, eval_interactions=test_interactions)
    print("Evaluation summary")
    print(json.dumps(metrics, indent=2))
    for uid in ["U001","U002","U003"]:
        t=travelers[travelers.traveler_id==uid].iloc[0]
        print(f"\n{uid}: interests={t.interests}, preferences={t.explicit_preferences}, party={t.party_type}")
        print(pipe.recommend(traveler_id=uid,k=5)[["name","category","final_score","context_compatibility","explanation"]].to_string(index=False))

if __name__ == "__main__":
    main()
