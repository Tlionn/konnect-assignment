# Technical Design

## 1. Problem formulation

Given traveler `u`, trip context `c`, and destination POIs `P`, produce a personalized ranked POI list for a downstream itinerary planner.

Two scores are intentionally separated:

- **preference relevance**: how likely the traveler is to value the POI;
- **context compatibility**: whether the POI is practical for this trip.

Final utility combines preference and compatibility and applies an availability guardrail.

## 2. Data and representations

The deterministic synthetic dataset contains 168 POIs, 90 travelers and 1,800 implicit-feedback events across Seoul and Busan.

POI features include category/subcategory, tags, rating, review volume, price, popularity, localness, accessibility, transit, family suitability, duration, reservation friction and availability.

Traveler features include destination, trip duration, explicit interests/preferences, budget, party type and mobility.

Interaction signals are graded:

- view 0.15
- click 0.30
- save 0.60
- navigate 0.75
- visit 0.90
- booking 1.00
- dismiss -0.70

## 3. Leakage-safe historical features

Training rows are processed chronologically per traveler.

Before scoring each interaction, the user profile contains only earlier high-intent events. The current interaction is added afterward.

This avoids target leakage from full-history aggregates.

## 4. Candidate generation

Candidate retrieval uses three lanes:

1. primary interest/budget/quality relevance;
2. long-tail/local discovery;
3. exploration/quality.

This prevents popularity-heavy retrieval from eliminating niche neighborhood experiences before ranking.

On the final holdout, candidate recall is 1.0.

## 5. Model selection

A temporal 60/15/25 split is used for model selection.

Compared approaches:

- content/history heuristic;
- Logistic Regression;
- HistGradientBoosting;
- LambdaMART-style XGBoost learning-to-rank.

Regularized Logistic Regression (`C=2`) has the best validation NDCG@10 and is selected.

This result is consistent with the small sparse tabular regime: lower-variance linear decision boundaries generalize better than more expressive models.

## 6. Preference ranking

The selected model predicts the probability of a high-intent event (`signal >= 0.6`) using explicit, contextual and historical pair features.

Important features include:

- interest/category match;
- interest × history interaction;
- explicit tag overlap;
- explicit localness interaction;
- budget fit;
- historical localness/popularity affinity;
- accessibility and family fit.

The standardized coefficients are written to `reports/feature_coefficients.csv`.

## 7. Context compatibility

Practical compatibility is scored independently from preference:

- budget fit: 32%
- mobility fit: 28%
- family/party fit: 20%
- availability: 20%

The serving utility is:

`utility = (0.75 * preference + 0.25 * compatibility) * availability_guardrail`

This allows a POI to be highly preferred while still being demoted for the current trip.

## 8. Evaluation

Holdout evaluation uses the final 25% of each user timeline after model selection is complete. POIs already seen in the fit window are excluded.

Preference-ranking results:

- Precision@5: 0.352
- Recall@5: 0.700
- NDCG@5: 0.628
- Precision@10: 0.226
- Recall@10: 0.910
- NDCG@10: 0.709

Additional diagnostics:

- candidate recall: 1.000
- popularity baseline NDCG@10: 0.070
- personalization distance: 0.936
- catalog coverage@10: 0.720
- long-tail share@10: 0.438
- constraint compatibility@10: 0.866

Final context-adjusted utility is reported separately: NDCG@10 = 0.650.

## 9. Precision ceiling

The test set has only 2.48 relevant items per evaluated user on average, so mean Precision@10 cannot exceed 0.248. The model reaches 0.226, or 91.1% of that ceiling.

## 10. Failure analysis

The weakest cases show three main patterns:

- held-out behavior conflicts with explicit traveler interests;
- long-tail items have insufficient evidence to rank high;
- all relevant items are retrieved but graded order is imperfect.

These are documented in `reports/failure_analysis.csv`.

## 11. Cold start

### New traveler

Use explicit interests, destination, budget, party and mobility immediately. Historical features default to neutral values. Controlled exploration collects evidence.

### New POI

Content/context features make new items rankable without interactions. Use smoothed quality priors and exploration exposure.

### New destination

Start with content/context relevance and global feature relationships, then adapt as destination-specific feedback accumulates.

## 12. Production design

See `docs/production_architecture.md` for the feature-store split, online serving path, feedback loop, freshness, retraining and monitoring strategy.