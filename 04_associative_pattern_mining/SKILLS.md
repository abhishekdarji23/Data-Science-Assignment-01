# SKILLS.md — Skills Demonstrated / Applied in This Build

Scoped-down equivalent of the source repo's agent-skills catalog. Each item
below is a real, checkable practice applied somewhere in this codebase
(file + line pointer), not just a claim.

| # | Skill | Where applied |
|---|---|---|
| 1 | CRISP-DM structuring | Business Understanding (this section / README) → Data Understanding (`data/generate_data.py`) → Data Prep (`src/data_prep.py`) → Modeling (`src/train.py`) → Evaluation (rule sanity-checking, `train.py`) → Deployment (`app.py`) |
| 2 | Correct market-basket data shaping | `src/data_prep.py: build_basket_matrix()` — pivots long-format transactions into the one-hot matrix `apriori()` requires |
| 3 | Using lift, not just confidence, to rank associations | `src/train.py` docstring explains why (popular items inflate confidence regardless of real association); `MIN_LIFT` filter + `sort_values("lift")` |
| 4 | Documented, non-arbitrary mining thresholds | `src/train.py` — `MIN_SUPPORT`, `MIN_CONFIDENCE`, `MIN_LIFT` are named constants with comments explaining each, not magic numbers |
| 5 | Verifiable synthetic data (ground-truth patterns) | `data/generate_data.py: BAKED_IN_RULES` — known patterns baked in so the mined rules can be checked against a ground truth (see `AUDIT_REPORT.md`) |
| 6 | Graceful handling of unseen inputs | `src/predict.py: recommend()` — items not in the catalog are reported in `unknown_items` rather than silently ignored or causing an error |
| 7 | Model card / observability | `/api/metrics` endpoint (`app.py`) + `models/metrics.json`, surfaced in the UI's admin panel (`static/app.js`) |
| 8 | API input validation | `app.py: CartRequest` — Pydantic `min_length=1` on the cart, returns HTTP 422 on an empty cart |
| 9 | Automated testing, including a domain-specific regression test | `tests/test_api.py: test_recommend_classic_diapers_beer_pattern` — a test that would fail if the data generator, mining thresholds, or recommend logic broke, not just a generic smoke test |
| 10 | Reproducibility | Fixed `RNG_SEED = 42` in `data/generate_data.py` |
| 11 | Documentation-as-artifact | This file, plus `PROMPTS.md`, `IMPLEMENTATION_PLANS.md`, `WALKTHROUGH.md`, `AUDIT_REPORT.md`, `README.md` |

## Explicitly not attempted here (see `IMPLEMENTATION_PLANS.md` §9)

FP-Growth comparison, algorithm performance benchmarking, a co-occurrence
graph visualization, AutoML-style threshold tuning, and a database-backed
multi-user admin dashboard were all present in the source repo's larger
vision but were intentionally left out — per your explicit instruction to
keep this "not fancy," with the main aim being a simple, working frontend
and backend.
