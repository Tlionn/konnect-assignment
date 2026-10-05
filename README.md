# Personalized POI Intelligence & Ranking System

Take-home ML prototype for ranking travel Points of Interest (POIs) for a traveler using trip context, explicit preferences, implicit behavior, and practical constraints.

## What this implements

The system deliberately separates **preference relevance** ("would this traveler want it?") from **contextual compatibility** ("does it make sense for this trip?"). The final score combines both, with availability acting as a guardrail.

Pipeline:

```text
POI catalog + traveler/trip context + historical interactions
        -> data preparation / feature engineering
        -> quota-aware candidate generation
        -> learned personalized preference ranker
        -> context & constraint scoring
        -> weighted POI recommendations + explanations
        -> offline ranking evaluation
```

### Design choices

- **Candidate generation:** interest, budget, quality and localness signals, plus a dedicated long-tail lane. This prevents relevant niche POIs from disappearing before ranking.
- **Preference model:** `HistGradientBoostingRegressor` over explicit, behavioral and POI features. A compact tree model is appropriate for the small heterogeneous tabular dataset and makes the prototype reproducible on CPU.
- **Behavioral supervision:** implicit events map to graded signals (`view=0.15`, `click=0.30`, `save=0.60`, `navigate=0.75`, `visit=0.90`, `booking=1.0`, `dismiss=-0.70`).
- **Context utility:** budget, mobility, party/family suitability and availability are scored separately from preference.
- **Explainability:** top recommendations include rule-based reason strings using the strongest interpretable signals.

## Repository structure

```text
.
├── data/                         # generated synthetic data
├── docs/technical_design.md      # architecture, assumptions, cold-start, production plan
├── reports/                      # metrics + example recommendations after run
├── src/poi_intelligence/
│   ├── data.py                   # deterministic synthetic data generator
│   ├── features.py               # explicit/implicit/context feature engineering
│   ├── model.py                  # learned preference ranker
│   ├── pipeline.py               # candidates -> rank -> context utility -> explain
│   └── evaluation.py             # ranking & recommender metrics
├── tests/test_pipeline.py
├── run.py
└── requirements.txt
```

## Reproduce the solution

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python run.py
pytest -q
```

`python run.py` regenerates the synthetic data deterministically, trains the model, evaluates it, and writes:

- `reports/metrics.json`
- `reports/per_user_metrics.csv`
- `reports/example_recommendations.csv`

## Output contract

Each ranked POI contains:

- `poi_id`
- `final_score`: downstream utility weight
- `preference_score`: learned traveler relevance
- `context_compatibility`: practical trip fit
- `confidence`: evidence-strength proxy
- component compatibility/features
- human-readable explanation

The downstream itinerary planner can therefore distinguish relative preference from feasibility without this component attempting route or itinerary optimization.

## Evaluation

The implementation uses a **per-user chronological 75/25 interaction holdout** and reports the assignment-required ranking metrics:

- Precision@10
- Recall@10
- NDCG@10

It also reports:

- personalization distance (pairwise Jaccard distance between recommendation lists)
- catalog coverage@10
- long-tail share@10
- constraint compatibility@10
- intra-list category diversity

See `reports/metrics.json` after running the pipeline and `docs/technical_design.md` for methodology and caveats.

## Scope intentionally excluded

No RAG, chatbot, LLM travel assistant, itinerary/TSP optimization, booking flow, external orchestration, or frontend is included. This repository focuses only on the POI intelligence/ranking layer requested in the assignment.
