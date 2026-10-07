from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .features import empty_profile, pair_features

class _RollingProfile:
    def __init__(self):
        self.cat={}; self.tag={}; self.local_sum=0.; self.pop_sum=0.; self.weight_sum=0.; self.positive_count=0
    def snapshot(self):
        if self.weight_sum<=0: return empty_profile()
        ct=max(sum(self.cat.values()),1e-9); tt=max(sum(self.tag.values()),1e-9)
        return {"category_affinity":{k:v/ct for k,v in self.cat.items()},"tag_affinity":{k:v/tt for k,v in self.tag.items()},
                "local_affinity":self.local_sum/self.weight_sum,"popularity_affinity":self.pop_sum/self.weight_sum,
                "history_strength":min(1.0,self.positive_count/8.0)}
    def update(self,poi,signal):
        if signal<0.6: return
        w=float(signal); self.cat[poi.category]=self.cat.get(poi.category,0.0)+w
        for tag in str(poi.tags).split('|'): self.tag[tag]=self.tag.get(tag,0.0)+w
        self.local_sum+=w*float(poi.localness); self.pop_sum+=w*float(poi.popularity); self.weight_sum+=w; self.positive_count+=1

class PreferenceRanker:
    """Leakage-safe, regularized linear preference ranker selected by temporal validation."""
    def __init__(self,random_state=42,C=2.0):
        self.feature_names=None
        self.model=Pipeline([
            ('scale',StandardScaler()),
            ('ranker',LogisticRegression(C=C,max_iter=2000,random_state=random_state)),
        ])
    def fit(self,travelers,pois,interactions):
        ti=travelers.set_index('traveler_id'); pi=pois.set_index('poi_id')
        rows=[]; targets=[]; weights=[]
        ordered=interactions.copy(); ordered['timestamp']=pd.to_datetime(ordered['timestamp']); ordered=ordered.sort_values(['traveler_id','timestamp'])
        for uid,grp in ordered.groupby('traveler_id',sort=False):
            profile=_RollingProfile(); t=ti.loc[uid]
            for _,inter in grp.iterrows():
                p=pi.loc[inter.poi_id]; rows.append(pair_features(t,p,profile.snapshot())); y=int(float(inter.signal)>=0.6); targets.append(y)
                weights.append(1.5 if float(inter.signal)<0 else (1.0+0.5*float(inter.signal) if y else 0.75)); profile.update(p,float(inter.signal))
        x=pd.DataFrame(rows); self.feature_names=list(x.columns)
        self.model.fit(x[self.feature_names],np.asarray(targets),ranker__sample_weight=np.asarray(weights)); return self
    @staticmethod
    def content_prior(feats):
        return np.clip(0.42*feats['interest_match'].to_numpy(float)+0.18*feats['preference_match'].to_numpy(float)+
                       0.10*feats['explicit_local_x_localness'].to_numpy(float)+0.08*feats['explicit_famous_x_popularity'].to_numpy(float)+
                       0.10*feats['historical_category_affinity'].to_numpy(float)+0.04*feats['historical_tag_affinity'].to_numpy(float)+
                       0.05*feats['rating_norm'].to_numpy(float)+0.03*feats['history_local_similarity'].to_numpy(float),0.,1.)
    def predict_components(self,traveler,candidates,history_profile=None):
        feats=pd.DataFrame([pair_features(traveler,p,history_profile) for _,p in candidates.iterrows()])
        learned=self.model.predict_proba(feats[self.feature_names])[:,1]
        prior=self.content_prior(feats)
        return {'learned':learned,'prior':prior},feats
    def predict(self,traveler,candidates,history_profile=None):
        comps,feats=self.predict_components(traveler,candidates,history_profile); return comps['learned'],feats
