# Technical Design

## 1. Problem formulation

For traveler `u`, trip context `c`, and destination POI catalog `P`, produce a ranked list of POIs with weights that represent downstream itinerary utility. The ranking should be personalized rather than popularity-driven and should distinguish user preference from practical feasibility.

The prototype uses two scores:

- **Preference score**: learned affinity between traveler/trip and POI.
- **Context compatibility**: deterministic trip feasibility score using budget, mobility, party fit and availability.

Final utility is:

`utility = (0.68 * preference + 0.32 * compatibility) * availability_guardrail`

The weights are explicit prototype choices, not claimed business-optimal constants. In production they would be tuned from online/offline objectives and product constraints.

## 2. Architecture

1. Data preparation and normalization
2. POI/traveler feature construction
3. Behavioral profile aggregation from historical interactions
4. Candidate generation with a long-tail quota
5. Learned preference ranking using gradient-boosted trees
6. Contextual/practical scoring
7. Final weighted ranking
8. Explanation generation
9. Offline evaluation

This is intentionally a two-stage recommender. Candidate generation provides scalability; the ranker spends more compute only on a reduced candidate set.

## 3. Data assumptions

The submitted data is synthetic and deterministic. It contains two destinations, 168 POIs, 90 travelers and implicit-feedback events. POIs include category, tags, geography, price, ratings, review counts, popularity, localness, family suitability, transit/accessibility, duration, reservation and availability fields.

Travelers include destination, trip duration, interests, budget, party type, mobility, and explicit preferences. Interactions include graded positive signals and dismissals.

Synthetic labels are not evidence of real-world business lift; they exist to demonstrate a complete, reproducible ML pipeline and evaluation mechanics.

## 4. Feature engineering

### POI features

- rating and log-normalized review volume
- popularity and inverse-popularity (long-tail)
- localness
- accessibility and transit score
- expected visit duration
- reservation friction and weekend availability
- category/tags

### Traveler/trip features

- explicit interests
- explicit textual preference tags
- budget
- mobility
- party type
- trip duration

### Pairwise / interaction features

- interest/category match
- explicit tag overlap
- budget fit
- mobility fit
- family fit
- localness interaction for travelers requesting local/less-touristy POIs
- historical category affinity
- historical tag affinity
- historical localness/popularity similarity
- history strength

Explicit preferences and implicit behavior are kept distinct in feature construction, then learned jointly by the ranker.

## 5. Model selection

The prototype uses `HistGradientBoostingRegressor` as a pointwise ranking model. This was selected because the problem is heterogeneous tabular data, the dataset is intentionally small, CPU inference is cheap, nonlinear feature interactions matter, and complexity is lower than neural/two-tower approaches.

For a production system with much larger interaction volume, I would compare this baseline against pairwise/listwise learning-to-rank and a two-tower retrieval model.

## 6. Training methodology

Historical traveler–POI events are converted to graded implicit targets:

- view 0.15
- click 0.30
- save 0.60
- navigate 0.75
- visit 0.90
- booking 1.00
- dismiss -0.70

Each interaction becomes one traveler–POI training pair. Historical user-profile features are aggregated from positive interactions. The model predicts a raw utility signal which is transformed to a bounded preference score for downstream combination.

The demo already uses a chronological holdout at evaluation time. A production implementation should additionally make every aggregated feature event-time-correct, use impression-aware negatives, and reserve a separate validation window for tuning.

## 7. Candidate generation

Candidate generation combines interest relevance, budget fit, quality and localness. A dedicated long-tail lane fills approximately 25% of the candidate budget from lower-popularity POIs that still match interests.

This is important because a ranker cannot recover a relevant POI that retrieval eliminated. Long-tail preservation must therefore happen before ranking, not merely as a post-ranking diversity adjustment.

At large scale, the retrieval stage would evolve to multiple candidate sources: ANN/two-tower semantic retrieval, collaborative candidates, destination/category indexes, exploration/long-tail source, and business/context rules followed by union + deduplication.

## 8. Ranking and scoring

The learned model estimates **preference relevance**. Separately, context compatibility combines:

- budget fit: 32%
- mobility fit: 28%
- party/family fit: 20%
- availability: 20%

Final utility combines 68% preference and 32% compatibility. Closed/unavailable POIs are strongly demoted by a multiplicative guardrail.

The separation is intentional: a user can strongly prefer a POI while the system can still communicate that it is unsuitable for the current trip.

## 9. Evaluation

The runnable demo performs a per-user chronological 75/25 split. The ranker and historical profiles are fit only on the earlier interactions; Precision@10, Recall@10 and NDCG@10 are computed against high-strength held-out interactions (`signal >= 0.6`). It additionally measures:

- personalization: mean pairwise Jaccard distance between top-10 lists
- catalog coverage@10
- long-tail share@10 (popularity < 0.35)
- constraint compatibility@10 (context score >= 0.70)
- intra-list category diversity

### Important evaluation caveat

Because this is synthetic implicit-feedback data, the numerical metrics validate pipeline behavior rather than estimate real-world CTR/booking uplift. Production evaluation would retain chronological holdouts while adding impression logs, propensity-aware counterfactual evaluation where appropriate, and online A/B tests.

## 10. Cold-start strategy

### New traveler

Use explicit interests/preferences, destination, budget, party and mobility immediately. Back off to destination/category priors only where explicit evidence is absent. Introduce exploration slots to collect feedback quickly.

### New POI

Content and metadata features make new items rankable without interactions. Apply Bayesian-smoothed quality priors instead of treating missing reviews as low quality. Ensure a controlled exploration quota gives new POIs exposure.

### New destination

Rely initially on content/category/context relevance and global behavioral priors. Transfer category/tag embeddings or learned feature relationships across destinations, then adapt as destination-specific interactions arrive.

## 11. Production considerations

### Scale

Store POI features offline and retrieve candidates from vector/feature indexes. Use multiple retrieval sources and a low-latency ranker service. Cache destination-level candidate pools where safe.

### Offline vs online features

Offline: POI embeddings, popularity windows, smoothed ratings, long-term traveler profile, collaborative statistics.

Online: current trip context, availability/opening hours, live distance/location, session behavior, recent saves/dismissals.

### Serving

Request -> fetch traveler/trip features -> multi-source retrieval -> online feature join -> ranker -> context guardrails -> explanations -> response. Feature contracts should be versioned so training and serving use the same definitions.

### Freshness

Use event-driven or short-TTL updates for opening hours/availability and session signals; scheduled recomputation for popularity and long-term profiles.

### Retraining

Start daily/weekly depending on volume and drift. Trigger retraining or rollback based on data/model drift and business KPIs rather than cadence alone.

### Feedback loop

Log impressions, rank position, clicks, saves, navigation, visits, bookings and dismissals. Impression logging is essential so non-interaction can be interpreted correctly and selection bias can be measured.

### Monitoring

Monitor retrieval recall, NDCG/engagement proxies, feature missingness, latency, candidate-source mix, popularity concentration, catalog coverage, long-tail exposure, constraint violation rate, score calibration, drift, and slices by destination/party/budget/mobility.

## 12. What I would improve with real data

1. Replace synthetic supervision with impression-aware logs.
2. Use chronological train/validation/test splits.
3. Compare pointwise baseline with LambdaMART/pairwise ranking.
4. Add calibrated probability/confidence rather than evidence-strength proxy.
5. Train semantic/two-tower retrieval for large catalogs.
6. Tune objective weights against itinerary-planner and business outcomes.
7. Add fairness/exposure monitoring for suppliers and neighborhoods.
