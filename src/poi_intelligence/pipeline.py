from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from .features import build_user_history_profiles
from .model import PreferenceRanker
from .evaluation import precision_at_k, recall_at_k, ndcg_at_k, intra_list_category_diversity, personalization_jaccard


class POIRecommendationPipeline:
    """Two-stage recommender: multi-lane candidates -> learned preference -> context utility."""

    def __init__(self, random_state=42, context_weight=0.25):
        self.ranker = PreferenceRanker(random_state=random_state)
        self.profiles = {}
        self.context_weight = context_weight

    def fit(self, travelers, pois, interactions):
        self.travelers = travelers.copy()
        self.pois = pois.copy()
        self.interactions = interactions.copy()
        self.profiles = build_user_history_profiles(travelers, pois, interactions)
        self.ranker.fit(travelers, pois, interactions)
        return self

    def _candidate_generation(self, traveler, max_candidates=72, exclude_poi_ids=None):
        pool = self.pois[self.pois.destination == traveler.destination].copy()
        if exclude_poi_ids:
            pool = pool[~pool.poi_id.isin(set(exclude_poi_ids))].copy()
        interests = set(str(traveler.interests).split("|"))
        prefs = set(str(traveler.explicit_preferences).split("|"))
        pool["interest_gate"] = pool.category.isin(interests).astype(float)
        pool["budget_gate"] = (pool.price_level <= traveler.budget_level + 1).astype(float)
        pool["local_gate"] = pool.localness if ({"local","less_touristy"} & prefs) else 0.0
        pool["famous_gate"] = pool.popularity if ({"famous","landmark"} & prefs) else 0.0
        pool["quality_gate"] = 0.65*(pool.rating/5) + 0.35*np.log1p(pool.review_count)/np.log1p(max(2,pool.review_count.max()))
        pool["candidate_score"] = 0.48*pool.interest_gate + 0.16*pool.budget_gate + 0.14*pool.quality_gate + 0.14*pool.local_gate + 0.08*pool.famous_gate
        n = min(max_candidates, len(pool))
        if n == 0:
            return pool
        main_n = int(n*0.70)
        main = pool.nlargest(main_n, "candidate_score")
        remaining = pool[~pool.poi_id.isin(main.poi_id)]
        tail_n = int(n*0.18)
        tail = remaining.assign(tail_score=0.62*remaining.interest_gate + 0.38*(1-remaining.popularity)).nlargest(tail_n, "tail_score")
        remaining2 = remaining[~remaining.poi_id.isin(tail.poi_id)]
        explore_n = n - len(main) - len(tail)
        explore = remaining2.nlargest(explore_n, "quality_gate")
        return pd.concat([main, tail, explore]).drop_duplicates("poi_id").head(n)

    @staticmethod
    def _context_scores(traveler, candidates):
        budget = np.maximum(0.0, 1.0 - 0.3*np.maximum(0, candidates.price_level.to_numpy()-int(traveler.budget_level)))
        if traveler.mobility == "public_transport":
            mobility = candidates.transit_score.to_numpy(float)
        elif traveler.mobility == "walking":
            mobility = 0.55*candidates.accessibility_score.to_numpy(float)+0.45*candidates.transit_score.to_numpy(float)
        else:
            mobility = np.full(len(candidates),0.9)
        family = candidates.family_friendly.to_numpy(float) if traveler.party_type == "family" else np.full(len(candidates),0.75)
        availability = candidates.open_weekend.astype(float).to_numpy()
        practical = 0.32*budget+0.28*mobility+0.20*family+0.20*availability
        return budget, mobility, family, availability, practical

    def recommend(self, traveler_id=None, traveler_override=None, k=10, exclude_poi_ids=None, ranking="utility"):
        if traveler_override is not None:
            traveler = pd.Series(traveler_override)
            uid = traveler.get("traveler_id", "NEW_USER")
            profile = self.profiles.get(uid)
        else:
            traveler = self.travelers[self.travelers.traveler_id == traveler_id].iloc[0]
            uid = traveler_id
            profile = self.profiles.get(uid)
        cand = self._candidate_generation(traveler, exclude_poi_ids=exclude_poi_ids)
        pref, feats = self.ranker.predict(traveler, cand, profile)
        budget, mobility, family, availability, practical = self._context_scores(traveler, cand)
        final = ((1-self.context_weight)*pref + self.context_weight*practical) * (0.25+0.75*availability)
        rank_score = pref if ranking == "preference" else final
        confidence = np.clip(0.40 + 0.40*feats.history_strength.to_numpy() + 0.20*np.minimum(1, cand.review_count.to_numpy()/500), 0, 1)
        result = cand[["poi_id","name","category","subcategory","rating","popularity","localness"]].copy()
        result["preference_score"] = pref
        result["context_compatibility"] = practical
        result["final_score"] = final
        result["confidence"] = confidence
        result["budget_fit"] = budget
        result["mobility_fit"] = mobility
        result["availability"] = availability
        result["interest_match"] = feats.interest_match.to_numpy()
        result["historical_category_affinity"] = feats.historical_category_affinity.to_numpy()
        result["_rank_score"] = rank_score
        result = result.sort_values("_rank_score", ascending=False).head(k).drop(columns=["_rank_score"]).reset_index(drop=True)
        result["explanation"] = result.apply(lambda r: self._explain(r, traveler), axis=1)
        return result

    @staticmethod
    def _explain(row, traveler):
        reasons=[]
        if row.interest_match > 0.5: reasons.append("matches stated interests")
        if row.historical_category_affinity > 0.03: reasons.append("consistent with past high-intent behavior")
        if row.localness > 0.65 and ("local" in str(traveler.explicit_preferences) or "less_touristy" in str(traveler.explicit_preferences)): reasons.append("strong local / less-touristy fit")
        if row.budget_fit > 0.8: reasons.append("within budget")
        if row.mobility_fit > 0.75: reasons.append("mobility-compatible")
        if row.availability == 1: reasons.append("available for the trip context")
        return "; ".join(reasons[:4]) or "balanced preference and practical fit"

    def _popularity_baseline(self, traveler, k=10, exclude_poi_ids=None):
        pool = self.pois[self.pois.destination == traveler.destination].copy()
        if exclude_poi_ids:
            pool = pool[~pool.poi_id.isin(set(exclude_poi_ids))]
        score = 0.7*pool.popularity.to_numpy(float) + 0.3*(pool.rating.to_numpy(float)/5.0)
        pool = pool.assign(_score=score).sort_values("_score", ascending=False)
        return pool.head(k).poi_id.tolist()

    def evaluate(self, k=10, eval_interactions=None):
        per_user=[]
        utility_rows=[]
        rec_lists=[]
        covered=set()
        top_long_tail=[]
        compat=[]
        diversities=[]
        candidate_recalls=[]
        relevant_counts=[]
        holdout_mode = eval_interactions is not None
        evaluation_data = self.interactions if eval_interactions is None else eval_interactions
        for uid, grp in evaluation_data.groupby("traveler_id"):
            positives = grp[grp.signal >= 0.6]
            if len(positives) < 2:
                continue
            traveler = self.travelers[self.travelers.traveler_id == uid].iloc[0]
            seen = set(self.interactions[self.interactions.traveler_id == uid].poi_id) if holdout_mode else set()
            candidates = self._candidate_generation(traveler, exclude_poi_ids=seen)
            relevant = positives.poi_id.tolist()
            unique_relevant = len(set(relevant))
            relevant_counts.append(unique_relevant)
            candidate_recalls.append(len(set(candidates.poi_id) & set(relevant))/max(1,unique_relevant))
            pref_rec = self.recommend(traveler_id=uid, k=k, exclude_poi_ids=seen, ranking="preference")
            util_rec = self.recommend(traveler_id=uid, k=k, exclude_poi_ids=seen, ranking="utility")
            pref_ids = pref_rec.poi_id.tolist()
            util_ids = util_rec.poi_id.tolist()
            rel_map = dict(zip(grp.poi_id, grp.signal))
            baseline = self._popularity_baseline(traveler, k=k, exclude_poi_ids=seen)
            per_user.append({
                "traveler_id": uid,
                "precision_at_k": precision_at_k(pref_ids,relevant,k),
                "recall_at_k": recall_at_k(pref_ids,relevant,k),
                "ndcg_at_k": ndcg_at_k(pref_ids,rel_map,k),
                "precision_at_5": precision_at_k(pref_ids,relevant,5),
                "recall_at_5": recall_at_k(pref_ids,relevant,5),
                "ndcg_at_5": ndcg_at_k(pref_ids,rel_map,5),
                "baseline_ndcg_at_k": ndcg_at_k(baseline, rel_map, k),
            })
            utility_rows.append((
                precision_at_k(util_ids,relevant,k),
                recall_at_k(util_ids,relevant,k),
                ndcg_at_k(util_ids,rel_map,k),
            ))
            rec_lists.append(util_ids)
            covered.update(util_ids)
            top_long_tail.extend((util_rec.popularity < 0.35).astype(float).tolist())
            compat.extend((util_rec.context_compatibility >= 0.7).astype(float).tolist())
            diversities.append(intra_list_category_diversity(util_rec.category.tolist()))
        pf=pd.DataFrame(per_user)
        ua=np.asarray(utility_rows,float)
        p10=float(pf.precision_at_k.mean())
        p10_ceiling=float(np.mean([min(k,n)/k for n in relevant_counts])) if relevant_counts else 0.0
        summary={
            "precision_at_5": float(pf.precision_at_5.mean()),
            "recall_at_5": float(pf.recall_at_5.mean()),
            "ndcg_at_5": float(pf.ndcg_at_5.mean()),
            "precision_at_10": p10,
            "precision_at_10_ceiling": p10_ceiling,
            "precision_at_10_fraction_of_ceiling": p10/p10_ceiling if p10_ceiling else 0.0,
            "avg_relevant_items_per_test_user": float(np.mean(relevant_counts)) if relevant_counts else 0.0,
            "recall_at_10": float(pf.recall_at_k.mean()),
            "ndcg_at_10": float(pf.ndcg_at_k.mean()),
            "utility_precision_at_10": float(ua[:,0].mean()) if len(ua) else 0.0,
            "utility_recall_at_10": float(ua[:,1].mean()) if len(ua) else 0.0,
            "utility_ndcg_at_10": float(ua[:,2].mean()) if len(ua) else 0.0,
            "popularity_baseline_ndcg_at_10": float(pf.baseline_ndcg_at_k.mean()),
            "ndcg_lift_vs_popularity": float(pf.ndcg_at_k.mean()-pf.baseline_ndcg_at_k.mean()),
            "candidate_recall": float(np.mean(candidate_recalls)) if candidate_recalls else 0.0,
            "personalization_distance": personalization_jaccard(rec_lists),
            "catalog_coverage_at_10": len(covered)/max(1,len(self.pois)),
            "long_tail_share_at_10": float(np.mean(top_long_tail)) if top_long_tail else 0.0,
            "constraint_compatibility_at_10": float(np.mean(compat)) if compat else 0.0,
            "intra_list_category_diversity": float(np.mean(diversities)) if diversities else 0.0,
            "evaluated_users": int(len(pf)),
        }
        return summary,pf

    def save_report(self, output_dir, eval_interactions=None):
        out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
        summary,per_user=self.evaluate(k=10, eval_interactions=eval_interactions)
        with open(out/"metrics.json","w") as f: json.dump(summary,f,indent=2)
        per_user.to_csv(out/"per_user_metrics.csv",index=False)
        scenarios = ["U001","U002","U003"]
        all_rows=[]
        for uid in scenarios:
            r=self.recommend(traveler_id=uid,k=10); r.insert(0,"traveler_id",uid); all_rows.append(r)
        pd.concat(all_rows).to_csv(out/"example_recommendations.csv",index=False)
        return summary
