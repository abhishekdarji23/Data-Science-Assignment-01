# SKILLS.md — Skills Demonstrated / Applied in This Build

Scoped-down equivalent of the source repo's agent-skills catalog. Each item
below is a real, checkable practice applied somewhere in this codebase
(file + line pointer), not just a claim.

| # | Skill | Where applied |
|---|---|---|
| 1 | CRISP-DM structuring | Business Understanding (this section / README) → Data Understanding (`data/generate_data.py`) → Data Prep (`src/data_prep.py`) → Modeling (`src/train.py`) → Evaluation (method comparison, `train.py`) → Deployment (`app.py`) |
| 2 | Using the right metric for imbalanced data | `src/train.py` docstring + `evaluate()` — PR-AUC (average precision) as the headline metric, not accuracy, for a 3% positive class |
| 3 | Honest, multi-method comparison (not cherry-picking the best) | `src/train.py` — 3 named methods trained and reported side by side, including the one (LOF) that performed poorly, with the reason documented rather than hidden (`AUDIT_REPORT.md` §3) |
| 4 | Correct sign-convention handling across different sklearn APIs | `src/train.py` — `-decision_function()` (Isolation Forest) and `-score_samples()` (LOF) both flipped to a consistent "higher = more anomalous" convention, verified against sklearn docs (`AUDIT_REPORT.md` §2) |
| 5 | Fair, threshold-consistent method comparison | `src/train.py: evaluate()` — flags the top-K most anomalous points (K = known contamination × n) for every method, rather than comparing raw score thresholds that aren't on the same scale |
| 6 | Ground-truth label used only for evaluation, never as a feature | `src/data_prep.py: FEATURE_COLUMNS` excludes `is_anomaly`; all 3 detectors fit unsupervised |
| 7 | Train/serve consistency for the scaler | `src/predict.py` loads `scaler.joblib` (fit once in training) rather than fitting a new scaler at serving time |
| 8 | Human-readable score rescaling without changing the underlying ranking | `src/predict.py: _risk_score()` — maps the raw Isolation Forest score to a 0-100 scale for the UI, documented as presentation-only |
| 9 | Model card / observability | `/api/metrics` endpoint (`app.py`) + `models/metrics.json`, surfaced in the UI's method-comparison table |
| 10 | API input validation | `app.py: TransactionRequest` — Pydantic field bounds on all 5 inputs, returns HTTP 422 on violation |
| 11 | Automated testing, including plausibility tests | `tests/test_api.py` — dedicated tests that an obviously-normal transaction scores low and an obviously-suspicious one scores high and gets flagged, not just that the endpoint returns 200 |
| 12 | Reproducibility | Fixed `RNG_SEED = 42` in `data/generate_data.py`, `RANDOM_SEED = 42` in `src/train.py` |
| 13 | Documentation-as-artifact | This file, plus `PROMPTS.md`, `IMPLEMENTATION_PLANS.md`, `WALKTHROUGH.md`, `AUDIT_REPORT.md`, `README.md` |

## Explicitly not attempted here (see `IMPLEMENTATION_PLANS.md` §8)

Autoencoder-based anomaly detection, One-Class SVM (a 4th popular
method), AutoResearch-style hyperparameter search, and a database-backed
multi-user admin dashboard were all present in the source repo's larger
vision but were intentionally left out to keep this build small, "not
fancy," and fully verifiable in one pass.
