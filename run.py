from pathlib import Path
import json
import pandas as pd
from src.poi_intelligence.pipeline import POIRecommendationPipeline
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data';REPORTS=ROOT/'reports'
def chronological_split(interactions,train_fraction=.75):
    train=[];test=[]
    for _,g in interactions.sort_values('timestamp').groupby('traveler_id'):
        c=max(1,int(len(g)*train_fraction));train.append(g.iloc[:c]);test.append(g.iloc[c:])
    return pd.concat(train,ignore_index=True),pd.concat(test,ignore_index=True)
def main():
    pois=pd.read_csv(DATA/'pois.csv');travelers=pd.read_csv(DATA/'travelers.csv');interactions=pd.read_csv(DATA/'interactions.csv');interactions['timestamp']=pd.to_datetime(interactions.timestamp)
    train,test=chronological_split(interactions);pipe=POIRecommendationPipeline(42).fit(travelers,pois,train);metrics=pipe.save_report(REPORTS,test)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
