# Baseline vs final selected model

The original prototype used a pointwise tree regressor. After temporal model selection, the final preference ranker is regularized Logistic Regression.

| Metric | Original | Final |
|---|---:|---:|
| Precision@10 | 0.088 | **0.226** |
| Recall@10 | 0.355 | **0.910** |
| NDCG@10 | 0.198 | **0.709** |
| Candidate recall | — | **1.000** |
| Popularity baseline NDCG@10 | — | 0.070 |

The mean theoretical Precision@10 ceiling on this holdout is 0.248 because there are only 2.48 relevant items per evaluated user. Final Precision@10 therefore reaches **91.1% of the ceiling**.

The context-adjusted utility surface is reported separately: NDCG@10 = **0.650**, Recall@10 = **0.835**, constraint compatibility@10 = **0.866**.