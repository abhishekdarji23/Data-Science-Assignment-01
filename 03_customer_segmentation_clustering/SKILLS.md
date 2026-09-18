# SKILLS.md — Skills Demonstrated / Applied in This Build

Scoped-down equivalent of the source repo's agent-skills catalog. Each item
below is a real, checkable practice applied somewhere in this codebase
(file + line pointer), not just a claim.

| # | Skill | Where applied |
|---|---|---|
| 1 | CRISP-DM structuring | Business Understanding (this section / README) → Data Understanding (`data/generate_data.py`) → Data Prep (`src/data_prep.py`) → Modeling (`src/train.py`) → Evaluation (silhouette sweep, `train.py`) → Deployment (`app.py`) |
| 2 | Feature scaling before distance-based clustering | `src/train.py` — `StandardScaler().fit_transform(X)` before `KMeans` (KMeans uses Euclidean distance, so unscaled Income (0–140) would dominate Age (18–75) and Spending Score (1–100)) |
| 3 | Principled model selection (not "eyeball the elbow") | `src/train.py` — silhouette score computed for every k in 2..8, `argmax` picks the final k |
| 4 | Avoiding identifier/demographic leakage into segment logic | `src/data_prep.py` docstring — `CustomerID` and `Gender` deliberately excluded from `FEATURE_COLUMNS` |
| 5 | Train/serve consistency for the scaler | `src/predict.py` loads `scaler.joblib` (fit once in training) rather than fitting a new scaler at serving time |
| 6 | Human-readable model outputs | `src/train.py: label_segment()` turns raw centroid numbers into a business label consumed by the dashboard |
| 7 | Model card / observability | `/api/metrics` endpoint (`app.py`) + `models/metrics.json`, surfaced in the UI's admin panel (`static/app.js`) |
| 8 | API input validation | `app.py: CustomerRequest` — Pydantic field bounds on age/income/spending, returns HTTP 422 on violation |
| 9 | Automated testing | `tests/test_api.py` — health, metrics, cluster shape, customer list, plausible-prediction, and boundary-rejection tests, run via `pytest` |
| 10 | Reproducibility | Fixed `RANDOM_SEED = 42` in both `data/generate_data.py` and `src/train.py` (`KMeans(..., random_state=RANDOM_SEED, n_init=10)`) |
| 11 | Documentation-as-artifact | This file, plus `PROMPTS.md`, `IMPLEMENTATION_PLANS.md`, `WALKTHROUGH.md`, `AUDIT_REPORT.md`, `README.md` |

## Explicitly not attempted here (see `IMPLEMENTATION_PLANS.md` §9)

Hierarchical/DBSCAN comparison, PCA/t-SNE dimensionality-reduction views,
AutoML hyperparameter search, and a database-backed multi-user admin
dashboard were all present in the source repo's larger vision but were
intentionally left out to keep this build small, "not fancy," and fully
verifiable in one pass.
