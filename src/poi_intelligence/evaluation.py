from __future__ import annotations

import math
import numpy as np


def precision_at_k(recommended, relevant, k=10):
    rec = recommended[:k]
    return len(set(rec) & set(relevant))/max(1, k)


def recall_at_k(recommended, relevant, k=10):
    rel = set(relevant)
    return len(set(recommended[:k]) & rel)/max(1, len(rel))


def ndcg_at_k(recommended, relevance_map, k=10):
    dcg = sum((2**max(0.0, relevance_map.get(pid, 0.0))-1)/math.log2(i+2) for i,pid in enumerate(recommended[:k]))
    ideal = sorted([max(0.0,v) for v in relevance_map.values()], reverse=True)[:k]
    idcg = sum((2**v-1)/math.log2(i+2) for i,v in enumerate(ideal))
    return dcg/idcg if idcg else 0.0


def intra_list_category_diversity(categories):
    if len(categories) < 2:
        return 0.0
    pairs = 0
    different = 0
    for i in range(len(categories)):
        for j in range(i+1, len(categories)):
            pairs += 1
            different += categories[i] != categories[j]
    return different/pairs


def personalization_jaccard(lists):
    vals=[]
    for i in range(len(lists)):
        for j in range(i+1,len(lists)):
            a,b=set(lists[i]),set(lists[j])
            vals.append(1-len(a&b)/max(1,len(a|b)))
    return float(np.mean(vals)) if vals else 0.0
