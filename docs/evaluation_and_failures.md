# Evaluation, Ablations and Failure Analysis

## Temporal protocol

Model selection uses a per-user chronological split:

- first 60%: train candidate models;
- next 15%: select model/hyperparameters;
- final 25%: untouched holdout.

After selection, the winner is refit on the first 75% and evaluated once on the final 25%. POIs already seen in the training history are masked during holdout ranking.

## Why Logistic Regression won

Regularized Logistic Regression (`C=2`) produced the highest validation NDCG@10.

That is a useful result, not a limitation. The dataset is small, sparse, tabular and heavily driven by explicit/content signals. The simpler model has lower variance and generalizes better than both HistGradientBoosting and LambdaMART-style learning-to-rank on this prototype.

Historical aggregate features are event-time-correct: every training event sees only behavior that happened earlier.

## Final-test results

Preference relevance:

- Precision@5: **0.352**
- Recall@5: **0.700**
- NDCG@5: **0.628**
- Precision@10: **0.226**
- Recall@10: **0.910**
- NDCG@10: **0.709**
- Candidate recall: **1.000**
- Popularity baseline NDCG@10: **0.070**

Final utility after contextual guardrails:

- Precision@10: **0.206**
- Recall@10: **0.835**
- NDCG@10: **0.650**
- Constraint compatibility@10: **0.866**

## Precision ceiling

There are 2.48 relevant items per evaluated test user on average. At K=10, the mean theoretical Precision@10 ceiling is therefore 0.248. The achieved 0.226 is **91.1% of that ceiling**.

## Why context-adjusted NDCG is lower

Preference relevance and practical trip utility are different objectives.

A POI can be behaviorally relevant but still be demoted because of budget, mobility, family fit or availability. The synthetic interaction generator also does not encode full live trip-date semantics, so behavioral relevance is not a complete ground truth for feasibility.

For that reason the submission reports preference ranking quality and final utility quality separately rather than forcing both into one metric.

## Failure patterns

The five weakest users are listed in `reports/failure_analysis.csv`.

Observed patterns:

1. **Behavior/profile conflict.** Some held-out synthetic positives fall outside the user's explicit interests.
2. **Long-tail misses.** Rare POIs remain retrievable but can lose in rank order when evidence is sparse.
3. **Graded-ordering errors.** In some cases every relevant POI is present in top 10, but NDCG drops because the strongest positive is not ranked first.

Since candidate recall is 1.0, these errors are primarily ranking or label ambiguity rather than retrieval failure.

## With real production data

I would add:

- impression-aware negative sampling;
- propensity-aware/counterfactual evaluation where appropriate;
- time decay and richer session context;
- destination-specific calibration;
- a separately validated contextual utility objective;
- online A/B testing against engagement and downstream itinerary quality.