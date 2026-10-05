from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from .features import build_user_history_profiles
from .model import PreferenceRanker
from .evaluation import precision_at_k, recall_at_k, ndcg_at_k, intra_list_category_diversity, personalization_jaccard


class POIRecommendationPipeline:
    """Two-stage recommender: quota-aware candidate generation + learned preference ranking + context utility."""

    def __init__(self, random_state=42):
        self.ranker = PreferenceRanker(random_state=random_state)
        self.profiles = {}

    def fit(self, travelers, pois, interactions):
        self.travelers = travelers.copy()
        self.pois = pois.copy()
        self.interactions = interactions.copy()
        self.profiles = build_user_history_profiles(travelers, pois, interactions)
        self.ranker.fit(travelers, pois, interactions)
        return self

    def _candidate_generation(self, traveler, max_candidates=60):
        pool = self.pois[self.pois.destination == traveler.destination].copy()
        interests = set(str(traveler.interests).split("|"))
        prefs = set(str(traveler.explicit_preferences).split("|"))
        pool["interest_gate"] = pool.category.isin(interests).astype(float)
        pool["budget_gate"] = (pool.price_level <= traveler.budget_level + 1).astype(float)
        pool["local_gate"] = pool.localness if ({"local","less_touristy"} & prefs) else 0.0
        pool["quality_gate"] = 0.65*(pool.rating/5) + 0.35*np.log1p(pool.review_count)/np.log1p(max(2,pool.review_count.max()))
        pool["candidate_score"] = 0.46*pool.interest_gate + 0.18*pool.budget_gate + 0.18*pool.quality_gate + 0.18*pool.local_gate
        n = min(max_candidates, len(pool))
        main_n = int(n*0.75)
        main = pool.nlargest(main_n, "candidate_score")
        remaining = pool[~pool.poi_id.isin(main.poi_id)]
        # Dedicated long-tail lane prevents popularity-heavy candidate generation from erasing niche POIs.
        tail_n = n-main_n
        tail = remaining.assign(tail_score=0.55*remaining.interest_gate + 0.45*(1-remaining.popularity)).nlargest(tail_n, "tail_score")
        out = pd.concat([main, tail]).drop_duplicates("poi_id").head(n)
        return out

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

    def recommend(self, traveler_id=None, traveler_override=None, k=10):
        if traveler_override is not None:
            traveler = pd.Series(traveler_override)
            uid = traveler.get("traveler_id", "NEW_USER")
            profile = self.profiles.get(uid)
        else:
            traveler = self.travelers[self.travelers.traveler_id == traveler_id].iloc[0]
            uid = traveler_id
            profile = self.profiles.get(uid)
        cand = self._candidate_generation(traveler)
        pref, feats = self.ranker.predict(traveler, cand, profile)
        budget, mobility, family, availability, practical = self._context_scores(traveler, cand)
        # Preference is dominant; context acts as utility correction. Availability is also a hard guardrail.
        final = (0.68*pref + 0.32*practical) * (0.35+0.65*availability)
        confidence = np.clip(0.45 + 0.35*feats.history_strength.to_numpy() + 0.20*np.minimum(1, cand.review_count.to_numpy()/500), 0, 1)
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
        result = result.sort_values("final_score", ascending=False).head(k).reset_index(drop=True)
        result["explanation"] = result.apply(lambda r: self._explain(r, traveler), axis=1)
        return result

    @staticmethod
    def _explain(row, traveler):
        reasons=[]
        if row.interest_match > 0.5: reasons.append("matches stated interests")
        if row.historical_category_affinity > 0.03: reasons.append("consistent with past positive behavior")
        if row.localness > 0.65 and ("local" in str(traveler.explicit_preferences) or "less_touristy" in str(traveler.explicit_preferences)): reasons.append("strong local / less-touristy fit")
        if row.budget_fit > 0.8: reasons.append("within budget")
        if row.mobility_fit > 0.75: reasons.append("mobility-compatible")
        if row.availability == 1: reasons.append("available for the trip context")
        return "; ".join(reasons[:4]) or "balanced preference and practical fit"

    def evaluate(self, k=10, eval_interactions=None):
        per_user=[]
        rec_lists=[]
        covered=set()
        top_long_tail=[]
        compat=[]
        diversities=[]
        evaluation_data = self.interactions if eval_interactions is None else eval_interactions
        for uid, grp in evaluation_data.groupby("traveler_id"):
            positives = grp[grp.signal >= 0.6]
            if len(positives) < 2:
                continue
            rec = self.recommend(traveler_id=uid, k=k)
            ids = rec.poi_id.tolist()
            rec_lists.append(ids)
            covered.update(ids)
            rel_map = dict(zip(grp.poi_id, grp.signal))
            relevant = positives.poi_id.tolist()
            per_user.append({
                "traveler_id": uid,
                "precision_at_k": precision_at_k(ids,relevant,k),
                "recall_at_k": recall_at_k(ids,relevant,k),
                "ndcg_at_k": ndcg_at_k(ids,rel_map,k),
            })
            top_long_tail.extend((rec.popularity < 0.35).astype(float).tolist())
            compat.extend((rec.context_compatibility >= 0.7).astype(float).tolist())
            diversities.append(intra_list_category_diversity(rec.category.tolist()))
        pf=pd.DataFrame(per_user)
        summary={
            "precision_at_10": float(pf.precision_at_k.mean()),
            "recall_at_10": float(pf.recall_at_k.mean()),
            "ndcg_at_10": float(pf.ndcg_at_k.mean()),
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
