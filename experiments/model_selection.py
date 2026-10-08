"""Temporal model selection experiment.

Protocol:
- first 60% of each user's timeline: train
- next 15%: validation/model selection
- final 25%: untouched test

The selected Logistic Regression is then refit on the first 75%.
XGBoost is optional and only required for the LambdaMART comparison.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.poi_intelligence.evaluation import ndcg_at_k, precision_at_k, recall_at_k
from src.poi_intelligence.features import build_user_history_profiles, pair_features
from src.poi_intelligence.model import _RollingProfile, PreferenceRanker
from src.poi_intelligence.pipeline import POIRecommendationPipeline

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"
REPORTS=ROOT/"reports"

def split_three(interactions):
    train=[]; val=[]; test=[]
    for _,g in interactions.sort_values("timestamp").groupby("traveler_id"):
        a=int(len(g)*0.60); b=int(len(g)*0.75)
        train.append(g.iloc[:a]); val.append(g.iloc[a:b]); test.append(g.iloc[b:])
    return pd.concat(train,ignore_index=True),pd.concat(val,ignore_index=True),pd.concat(test,ignore_index=True)

def causal_matrix(travelers,pois,interactions):
    ti=travelers.set_index("traveler_id"); pi=pois.set_index("poi_id")
    rows=[]; y=[]; weights=[]; qids=[]; graded=[]
    for qid,(uid,g) in enumerate(interactions.sort_values(["traveler_id","timestamp"]).groupby("traveler_id",sort=False)):
        profile=_RollingProfile(); t=ti.loc[uid]
        for _,r in g.iterrows():
            p=pi.loc[r.poi_id]
            rows.append(pair_features(t,p,profile.snapshot()))
            target=int(float(r.signal)>=0.6)
            y.append(target)
            weights.append(1.5 if float(r.signal)<0 else (1.0+0.5*float(r.signal) if target else 0.75))
            qids.append(qid)
            s=float(r.signal)
            graded.append(4 if s>=0.95 else 3 if s>=0.85 else 2 if s>=0.70 else 1 if s>=0.60 else 0)
            profile.update(p,s)
    return pd.DataFrame(rows),np.asarray(y),np.asarray(weights),np.asarray(qids),np.asarray(graded)

def evaluate_scores(travelers,pois,history,eval_df,score_fn,min_positives=1):
    helper=POIRecommendationPipeline(42)
    helper.travelers=travelers; helper.pois=pois; helper.interactions=history
    helper.profiles=build_user_history_profiles(travelers,pois,history)
    ti=travelers.set_index("traveler_id")
    out=[]
    for uid,g in eval_df.groupby("traveler_id"):
        pos=g[g.signal>=0.6]
        if len(pos)<min_positives: continue
        t=ti.loc[uid]
        seen=set(history[history.traveler_id==uid].poi_id)
        cand=helper._candidate_generation(t,exclude_poi_ids=seen)
        feats=pd.DataFrame([pair_features(t,p,helper.profiles.get(uid)) for _,p in cand.iterrows()])
        scores=score_fn(t,cand,feats)
        ids=cand.assign(_score=scores).sort_values("_score",ascending=False).head(10).poi_id.tolist()
        rel=pos.poi_id.tolist(); rel_map=dict(zip(g.poi_id,g.signal))
        out.append((precision_at_k(ids,rel,10),recall_at_k(ids,rel,10),ndcg_at_k(ids,rel_map,10)))
    a=np.asarray(out,float)
    return float(a[:,0].mean()),float(a[:,1].mean()),float(a[:,2].mean()),len(a)

def main():
    pois=pd.read_csv(DATA/"pois.csv")
    travelers=pd.read_csv(DATA/"travelers.csv")
    interactions=pd.read_csv(DATA/"interactions.csv")
    interactions["timestamp"]=pd.to_datetime(interactions["timestamp"])
    train,val,test=split_three(interactions)
    X,y,w,qids,graded=causal_matrix(travelers,pois,train)
    features=list(X.columns)

    rows=[]
    models={}
    for C in [0.05,0.1,0.25,0.5,1.0,2.0]:
        m=Pipeline([("scale",StandardScaler()),("ranker",LogisticRegression(C=C,max_iter=2000,random_state=42))])
        m.fit(X,y,ranker__sample_weight=w)
        models[f"Logistic C={C:g}"]=m
    hgb=HistGradientBoostingClassifier(max_iter=220,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=12,l2_regularization=0.5,random_state=42)
    hgb.fit(X,y,sample_weight=w)
    models["HistGradientBoosting"]=hgb

    for name,m in models.items():
        p,r,n,u=evaluate_scores(travelers,pois,train,val,lambda t,c,f,m=m:m.predict_proba(f[features])[:,1],1)
        rows.append((name,p,r,n,u))
    p,r,n,u=evaluate_scores(travelers,pois,train,val,lambda t,c,f:PreferenceRanker.content_prior(f),1)
    rows.append(("Content/history scorer",p,r,n,u))

    try:
        from xgboost import XGBRanker
        ltr=XGBRanker(objective="rank:ndcg",eval_metric="ndcg@10",n_estimators=220,max_depth=4,learning_rate=0.05,
                      subsample=0.9,colsample_bytree=0.9,min_child_weight=3,reg_lambda=1.0,random_state=42,n_jobs=4)
        ltr.fit(X,graded,qid=qids,verbose=False)
        def ltr_score(t,c,f):
            raw=ltr.predict(f[features]); sd=float(np.std(raw))
            return np.full_like(raw,0.5) if sd<1e-9 else 1/(1+np.exp(-(raw-np.mean(raw))/sd))
        p,r,n,u=evaluate_scores(travelers,pois,train,val,ltr_score,1)
        rows.append(("LambdaMART LTR",p,r,n,u))
    except ImportError:
        pass

    pd.DataFrame(rows,columns=["model","precision_at_10","recall_at_10","ndcg_at_10","users"]).sort_values("ndcg_at_10",ascending=False).to_csv(REPORTS/"model_selection_validation.csv",index=False)
    print(pd.read_csv(REPORTS/"model_selection_validation.csv").to_string(index=False))

if __name__=="__main__":
    main()
