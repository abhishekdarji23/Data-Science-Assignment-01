# SKILLS.md — Skills Demonstrated / Applied in This Build

Scoped-down equivalent of the source repo's agent-skills catalog. Each item
below is a real, checkable practice applied somewhere in this codebase
(file + line pointer), not just a claim.

| # | Skill | Where applied |
|---|---|---|
| 1 | CRISP-DM structuring | Business Understanding (this section / README) → Data Understanding (`data/generate_data.py`) → Data Prep (`src/data_prep.py`) → Modeling (`src/train.py`) → Evaluation (leaderboard, `train.py`) → Deployment (`app.py`) |
| 2 | Leakage-safe stacking via out-of-fold predictions | `src/train.py` — `cross_val_predict(..., method="predict_proba")` generates OOF predictions on the train set before the meta-learner ever sees them, per the module docstring's explanation |
| 3 | Fixed, pre-declared categorical encoding (production-safety) | `src/data_prep.py: CATEGORY_LEVELS` — prevents a real bug class where single-row inference silently produces fewer dummy columns than training did |
| 4 | Honest multi-model comparison via a proper held-out test set | `src/train.py` — every base model AND the stacked ensemble evaluated on the SAME test set, which was never used for OOF generation or meta-learner training |
| 5 | Using the right metric for the task | `src/train.py: evaluate()` — ROC-AUC as the headline metric for a binary classification leaderboard, with accuracy/precision/recall/F1 alongside |
| 6 | Train/serve consistency across a multi-model pipeline | `src/predict.py` reconstructs the exact L1→L2 architecture from training (every base model → meta-learner), not a shortcut approximation |
| 7 | Model card / observability, including ensemble interpretability | `/api/leaderboard` endpoint (`app.py`) + `models/metrics.json`'s `meta_learner_weights` — shows how much the meta-learner trusts each base model, not just the final number |
| 8 | API input validation with domain-typed fields | `app.py: CustomerRequest` — `Literal[...]` types on every categorical field, returns HTTP 422 on an invalid category before it reaches feature encoding |
| 9 | Automated testing, including plausibility tests | `tests/test_api.py` — dedicated tests that an obviously high-risk customer scores >50% and an obviously low-risk one scores <30%, not just that the endpoint returns 200 |
| 10 | Reproducibility | Fixed `RANDOM_SEED = 42` used consistently across the train/test split, every base model, and the stratified k-fold splitter |
| 11 | Documentation-as-artifact | This file, plus `PROMPTS.md`, `IMPLEMENTATION_PLANS.md`, `WALKTHROUGH.md`, `AUDIT_REPORT.md`, `README.md` |

## Explicitly not attempted here (see `IMPLEMENTATION_PLANS.md` §9)

The actual AutoGluon package, multiple task types in one build, a
deeper (3+ layer) stacking DAG, and hyperparameter search were all
present in the source repo's larger vision but were intentionally left
out — per your explicit instruction to keep this "not fancy," with the
main aim being a simple, working frontend and backend.
