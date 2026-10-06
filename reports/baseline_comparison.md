# Baseline vs improved ranking

The improved implementation keeps the same deterministic seed-42 dataset and chronological 75/25 holdout.

| Metric | Baseline | Improved |
|---|---:|---:|
| Precision@10 | 0.088 | **0.206** |
| Recall@10 | 0.355 | **0.833** |
| NDCG@10 | 0.198 | **0.610** |
| Personalization distance | 0.927 | **0.939** |
| Catalog coverage@10 | 0.690 | **0.738** |

Additional diagnostics:

- Precision@5: 0.284
- Recall@5: 0.570
- NDCG@5: 0.506
- Candidate recall: 1.000
- Popularity baseline NDCG@10: 0.070
- NDCG lift vs popularity: +0.540
- Mean theoretical Precision@10 ceiling: 0.248
- Achieved fraction of Precision@10 ceiling: 83.1%

Main causes of improvement:

1. The train target now matches the evaluation relevance definition.
2. Historical features are chronological/causal rather than target-leaking.
3. Holdout evaluation masks POIs already seen in training.
4. Candidate generation has explicit long-tail and exploration lanes.
5. Hybrid content + behavioral ranking is better suited to sparse/cold-start users.
6. A popularity baseline makes personalization gain measurable rather than asserted.
