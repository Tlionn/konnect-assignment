# Technical Design

## 1. Problem formulation

Given traveler `u`, trip context `c`, and destination POIs `P`, return a weighted ranking for downstream itinerary planning.

The system separates:

1. **Preference relevance** — likelihood the traveler meaningfully wants the POI.
2. **Context compatibility** — whether it is practical for this trip.

Final utility:

`final = (0.75 * preference + 0.25 * compatibility) * availability_guardrail`

## 2. Synthetic data

The deterministic dataset contains:

- 168 POIs across Seoul and Busan
- 90 travelers
- 1,800 implicit feedback events

Signals:

- view = 0.15
- click = 0.30
- save = 0.60
- navigate = 0.75
- visit = 0.90
- booking = 1.00
- dismiss = -0.70

Synthetic metrics demonstrate system behavior and evaluation mechanics; they are not claims of real-world business lift.

## 3. Feature engineering

POI features include quality, popularity, localness, price, transit/accessibility, duration, reservation friction and availability.

Traveler/trip features include interests, explicit preference tags, budget, mobility, party type and trip duration.

Pair features include interest match, preference overlap, budget/mobility/family fit, explicit local/famous interactions, historical category/tag affinities, and similarity to historical localness/popularity preferences.

## 4. Leakage-safe behavioral history

Training interactions are processed chronologically per traveler.

Before predicting each event, the ranker receives a profile built only from **earlier high-intent events**. The current event is added afterward.

This avoids the common leakage bug where a full-history aggregate contains the very interaction being predicted.

At serving time, the profile naturally uses all behavior known up to request time.

## 5. Ranking objective

The original pointwise regressor predicted raw signal strength, while evaluation later treated `signal >= 0.60` as relevant. That objective mismatch hurt ranking.

The improved `HistGradientBoostingClassifier` directly predicts the probability of a **high-intent event**. Dismissals and strong conversions receive larger sample weights.

Because behavior is sparse, the learned probability is blended with a transparent prior:

`preference = 0.30 * learned_probability + 0.70 * content_history_prior`

This intentionally favors robust cold-start behavior in the small prototype.

## 6. Candidate generation

Candidate retrieval uses three lanes:

- primary relevance lane: interest, budget, quality, local/famous preference fit
- long-tail lane: interest relevance + inverse popularity
- exploration/quality lane

The holdout achieves **1.00 candidate recall**, meaning ranking—not retrieval—is the remaining bottleneck on the synthetic benchmark.

## 7. Context scoring

Context compatibility remains separate:

- budget: 32%
- mobility: 28%
- family/party fit: 20%
- availability: 20%

Availability also applies a multiplicative guardrail to final utility.

## 8. Offline evaluation

Interactions are split chronologically per traveler: 75% train, 25% test.

During holdout evaluation, POIs already seen in training are excluded from candidate generation. This prevents the metric from rewarding memorized historical items when the task is future discovery.

Reported metrics:

- Precision@5 / @10
- Recall@5 / @10
- NDCG@5 / @10
- candidate recall
- popularity baseline NDCG@10
- NDCG lift over popularity
- personalization
- catalog coverage
- long-tail exposure
- context compatibility
- intra-list category diversity

Current seed-42 results:

- Precision@5: **0.284**
- Recall@5: **0.570**
- NDCG@5: **0.506**
- Precision@10: **0.206**
- Recall@10: **0.833**
- NDCG@10: **0.610**
- Popularity baseline NDCG@10: **0.070**
- Candidate recall: **1.000**

The test set has 2.48 relevant POIs per evaluated user on average, which caps mean Precision@10 at 0.248. The system reaches 83.1% of that ceiling.

## 9. Cold start

### New traveler
Use explicit destination, interests, preferences, budget, party and mobility. Add controlled exploration to learn quickly.

### New POI
Content features make new POIs rankable without interactions. Use smoothed quality priors and exploration exposure.

### New destination
Start with content/context relevance and global feature relationships, then adapt as destination-specific interactions accumulate.

## 10. Production evolution

A production system would add:

- impression logging and exposure-aware negatives
- explicit train/validation/test time windows
- LambdaMART or pairwise/listwise ranking comparisons
- calibrated recommendation probabilities
- ANN/two-tower retrieval for large catalogs
- online A/B testing
- monitoring for drift, retrieval recall, latency, popularity concentration, coverage, long-tail exposure and constraint violations
