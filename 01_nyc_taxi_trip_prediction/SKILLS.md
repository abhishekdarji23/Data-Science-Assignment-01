# SKILLS.md — Skills Demonstrated / Applied in This Build

Scoped-down equivalent of the source repo's agent-skills catalog. Each item
below is a real, checkable practice applied somewhere in this codebase
(file + line pointer), not just a claim.

| # | Skill | Where applied |
|---|---|---|
| 1 | CRISP-DM structuring | Whole repo laid out as Business Understanding (this section / README) → Data Understanding (`data/generate_data.py`) → Data Prep (`src/data_prep.py`) → Modeling (`src/train.py`) → Evaluation (metrics block in `train.py`) → Deployment (`app.py`) |
| 2 | Leakage-safe feature engineering | `src/data_prep.py` docstring + `FEATURE_COLUMNS`: every input feature is derivable *before* the trip happens; `dropoff_datetime`/`trip_duration` never enter `X` |
| 3 | Time-aware train/test split | `src/train.py: time_based_split()` — chronological split, not `train_test_split(shuffle=True)` |
| 4 | Target transform for skewed regression targets | `src/train.py` — `log1p`/`expm1` around `GradientBoostingRegressor` |
| 5 | Model card / observability | `/api/metrics` endpoint (`app.py`) + `models/metrics.json`, surfaced in the UI's "admin" panel (`static/app.js`) |
| 6 | API input validation | `app.py: TripRequest` — Pydantic field bounds on lat/lon/passenger_count, returns HTTP 422 on violation |
| 7 | Train/serve consistency | `src/data_prep.py` imported by both `train.py` and `predict.py` — one feature-engineering function, not two copies |
| 8 | Automated testing | `tests/test_api.py` — health, metrics, happy-path prediction, and boundary-rejection tests, run via `pytest` |
| 9 | Reproducibility | Fixed `RANDOM_SEED = 42` in both `data/generate_data.py` and `src/train.py` |
| 10 | Documentation-as-artifact | This file, plus `PROMPTS.md`, `IMPLEMENTATION_PLANS.md`, `WALKTHROUGH.md`, `AUDIT_REPORT.md`, `README.md` |

## Explicitly not attempted here (see `IMPLEMENTATION_PLANS.md` §9)

AutoML/hyperparameter search, ensemble stacking, a database-backed admin
dashboard, and CI/CD were all present in the source repo's larger vision
but were intentionally left out to keep this build small, "not fancy," and
fully verifiable in one pass.
