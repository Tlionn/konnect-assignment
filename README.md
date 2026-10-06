# Personalized POI Intelligence & Ranking System

Take-home ML prototype for ranking travel Points of Interest (POIs) from traveler context, explicit preferences, historical behavior, and practical trip constraints.

The design separates **preference relevance** ("would this traveler want it?") from **context compatibility** ("does it make sense for this trip?").

## Architecture

```text
POI catalog + traveler/trip context + historical interactions
        -> feature preparation
        -> multi-lane candidate generation
        -> hybrid personalized preference ranker
        -> context / constraint scoring
        -> weighted POI ranking + explanations
        -> chronological offline evaluation
```

### Key design choices

- **Multi-lane candidates:** interest/budget/quality retrieval plus dedicated long-tail and exploration lanes.
- **Hybrid ranker:** `HistGradientBoostingClassifier` predicts high-intent relevance and is blended with a transparent content/history prior, which makes the system robust for sparse/cold-start users.
- **Leakage-safe behavioral features:** each training event uses only interactions that occurred before that event.
- **Seen-item masking:** held-out evaluation removes POIs already seen in the training window.
- **Separate context utility:** budget, mobility, party fit and availability remain distinct from preference relevance.
- **Popularity baseline:** the personalized system is compared against a non-personalized baseline rather than reporting isolated numbers.

## Current held-out results

Per-user chronological 75/25 split, deterministic seed 42:

| Metric | Improved | Original baseline |
|---|---:|---:|
| Precision@5 | **0.284** | — |
| Recall@5 | **0.570** | — |
| NDCG@5 | **0.506** | — |
| Precision@10 | **0.206** | 0.088 |
| Recall@10 | **0.833** | 0.355 |
| NDCG@10 | **0.610** | 0.198 |
| Popularity baseline NDCG@10 | 0.070 | — |
| Candidate recall | **1.000** | — |
| Personalization distance | **0.939** | 0.927 |
| Catalog coverage@10 | **0.738** | 0.690 |

The test set contains only **2.48 relevant POIs per evaluated user on average**, so mean theoretical Precision@10 is capped at **0.248**. The achieved **0.206 is 83.1% of that ceiling**. This is reported explicitly instead of manipulating the synthetic data to make Precision@10 look larger.

## Repository structure

```text
.
├── data/
├── docs/technical_design.md
├── reports/
│   ├── metrics.json
│   ├── per_user_metrics.csv
│   ├── example_recommendations.csv
│   └── baseline_comparison.md
├── src/poi_intelligence/
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── pipeline.py
│   └── evaluation.py
├── tests/test_pipeline.py
├── pytest.ini
├── run.py
└── requirements.txt
```

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
pytest -q
```

`python run.py` regenerates the deterministic synthetic data, applies the chronological split, trains the model, evaluates the holdout, and rewrites the report artifacts.

## Output contract

Each recommendation includes:

- `poi_id`
- `final_score`
- `preference_score`
- `context_compatibility`
- `confidence`
- component fit signals
- human-readable explanation

## Scope intentionally excluded

No RAG, chatbot, LLM travel assistant, itinerary/TSP optimization, booking/payment flow, frontend, or external orchestration. This repository implements only the POI intelligence/ranking layer requested in the assignment.
