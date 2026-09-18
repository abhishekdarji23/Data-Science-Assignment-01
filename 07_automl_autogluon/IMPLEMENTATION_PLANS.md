# IMPLEMENTATION_PLANS.md — Project 07: AutoML Stacking (Customer Churn)

## 1. Objective

Illustrate the core AutoML technique the original project named —
multi-model stacking — by training a zoo of different algorithms,
combining their out-of-fold predictions through a meta-learner, and
comparing everything on a leaderboard. Predicts customer churn (binary
classification) as the demonstration task.

**Note on scope**: the actual AutoGluon package is not used here — see
`PROMPTS.md` for the reasoning (dependency weight vs. "keep it simple").
This reimplements AutoGluon's core stacking technique directly.

## 2. Architecture

```
┌────────────────────┐   GET /api/leaderboard                 ┌──────────────────────┐
│  Browser (leaderboard│ ─────────────────────────────────▶ │   FastAPI app.py     │
│  + scoring form)     │ ◀───────────────────────────────── │   (uvicorn)          │
└────────────────────┘   POST /api/predict (new customer)      └──────────┬───────────┘
                                                                           │ loads
                                                                ┌──────────▼───────────┐
                                                                │ models/                │
                                                                │  base_*.joblib (×5)    │
                                                                │  meta_learner.joblib   │
                                                                │  scaler.joblib          │
                                                                │  metrics.json           │
                                                                └──────────▲───────────┘
                                                                           │ produced by
                                                                ┌──────────┴───────────┐
                                                                │ src/train.py            │
                                                                │  (model zoo + stacking) │
                                                                └──────────▲───────────┘
                                                                           │ reads
                                                                ┌──────────┴───────────┐
                                                                │ data/*.csv              │
                                                                │ (synthetic or real)     │
                                                                └───────────────────────┘
```

## 3. Data schema

| column | type | notes |
|---|---|---|
| customer_id | string | row id |
| tenure_months | int | how long the customer has had service |
| monthly_charges | float | current monthly bill |
| total_charges | float | lifetime billing |
| contract_type | Month-to-month / One year / Two year | strongest churn driver |
| internet_service | DSL / Fiber optic / No | |
| tech_support | Yes / No | |
| payment_method | Electronic check / Mailed check / Bank transfer / Credit card | |
| num_support_calls | int | recent support call volume |
| churn | int (0/1) | **target** |

## 4. Feature engineering (`src/data_prep.py`)

`build_features()` one-hot encodes the 4 categorical columns using
**fixed, pre-declared category levels** (`CATEGORY_LEVELS`), not levels
inferred from whatever rows happen to be present. This matters
specifically for live single-row scoring: encoding one customer row
against inferred categories would silently produce fewer dummy columns
than the model was trained on (since a single row can't contain every
category level), corrupting predictions without raising an error.

## 5. Modeling (`src/train.py`) — the stacking pipeline

**Model zoo** (5 algorithms, different families):
`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`,
`KNeighborsClassifier`, `DecisionTreeClassifier`.

**Stacking procedure** (standard Caruana-style / Wolpert stacked
generalization):
1. 80/20 stratified train/test split.
2. For each base model: 5-fold `cross_val_predict(..., method="predict_proba")`
   on the TRAIN set only → out-of-fold (OOF) probabilities. This is the
   step that prevents leakage — a fold's OOF prediction always comes from
   a model instance that never saw that fold during its own fit.
3. Refit each base model on the FULL train set (for serving + test evaluation).
4. Stack the 5 models' OOF probabilities into a meta-feature matrix,
   train a `LogisticRegression` meta-learner on top of it.
5. Evaluate every base model AND the stacked ensemble on the held-out
   test set (ROC-AUC headline metric, plus accuracy/precision/recall/F1).

**Results on this build's synthetic data** (see `models/metrics.json` for
the live numbers): stacked ensemble ROC-AUC ≈ 0.7385, narrowly ahead of
the best single base model (logistic regression, ≈ 0.7382) — a small but
real lift, consistent with what stacking typically delivers in practice.

## 6. API contract (`app.py`)

| endpoint | method | purpose |
|---|---|---|
| `/` | GET | serves the frontend |
| `/api/health` | GET | liveness check |
| `/api/leaderboard` | GET | every base model + the stacked ensemble, with full metrics, meta-learner weights, dataset stats |
| `/api/predict` | POST | `{tenure_months, monthly_charges, total_charges, contract_type, internet_service, tech_support, payment_method, num_support_calls}` → `{churn_probability, risk_level, base_model_predictions (all 5), leaderboard_best_model}` |

`contract_type`, `internet_service`, `tech_support`, and `payment_method`
are typed as `Literal[...]` in the Pydantic request model, so an invalid
category value is rejected with HTTP 422 before it ever reaches the
feature-encoding step.

## 7. Frontend

Plain HTML/CSS/JS (no build step, no charting library — bar charts are
plain CSS width bars): a leaderboard with a bar chart and full metrics
table (best model starred), and a scoring form that shows the final
stacked prediction AND every individual base model's opinion side by
side — so a reviewer can see the "AutoML" ensembling actually happening,
not just a single opaque number.

## 8. Verification performed

- `python src/train.py` — trains the full model zoo + meta-learner,
  printed leaderboard inspected manually
- `python -c "..."` sanity check: an obviously high-risk customer
  (new account, high bill, month-to-month, no tech support, many support
  calls) scored 76% churn probability; an obviously low-risk one
  (long tenure, low bill, two-year contract, tech support, no support
  calls) scored 15%
- `uvicorn app:app` + `curl` against `/api/health`, `/api/leaderboard`,
  `/api/predict` (both risk profiles, plus an invalid category value) —
  all returned expected shapes and correctly differentiated results
- `python -m pytest tests/` — 5/5 automated tests pass, including
  dedicated high-risk/low-risk scoring tests

## 9. Explicit non-goals (kept out of scope per "not fancy, simple frontend and backend")

- No actual AutoGluon dependency (see `PROMPTS.md`)
- Only one task type (binary classification), not "various data science tasks"
- No multi-level (3+ layer) stacking DAG — one L1→L2 stack, not a deeper architecture
- No hyperparameter search / "AutoResearch hill-climbing" — base models use fixed, reasonable defaults
- No authentication, database, or multi-user admin dashboard
- No containerization/CI — this is a local, single-process demo
