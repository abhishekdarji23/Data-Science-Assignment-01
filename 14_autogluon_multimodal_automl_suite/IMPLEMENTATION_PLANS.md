# IMPLEMENTATION_PLANS.md — Project 14: Multimodal AutoML Suite

## 1. Objective

Demonstrate multimodal fusion — combining tabular and text features into
one model — and prove it genuinely outperforms either modality alone,
using the exact technique real AutoGluon Multimodal is built on
(feature-level fusion), without the heavy deep-learning dependency
stack. Predicts a listing's price from tabular fields + a free-text
description.

## 2. Architecture

```
┌────────────────────┐   GET /api/leaderboard                 ┌──────────────────────┐
│  Browser (leaderboard│ ─────────────────────────────────▶ │   FastAPI app.py     │
│  + prediction form)  │ ◀───────────────────────────────── │   (uvicorn)          │
└────────────────────┘   POST /api/predict (new listing)       └──────────┬───────────┘
                                                                           │ loads
                                                                ┌──────────▼───────────┐
                                                                │ models/                │
                                                                │  model_tabular_only,   │
                                                                │  model_text_only,      │
                                                                │  model_fusion (.joblib)│
                                                                │  scaler, vectorizer    │
                                                                │  metrics.json           │
                                                                └──────────▲───────────┘
                                                                           │ produced by
                                                                ┌──────────┴───────────┐
                                                                │ src/train.py            │
                                                                │  (3 models, offline)    │
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
| listing_id | string | row id |
| bedrooms, bathrooms, accommodates | int | tabular capacity features |
| room_type | Entire home/apt / Private room / Shared room | tabular categorical |
| neighborhood | Downtown / Uptown / Waterfront / Suburb | tabular categorical |
| description | string | free text, carries a "quality tier" signal NOT present in the tabular fields |
| price | float | **target** |

`data/generate_data.py`'s docstring explains the deliberate construction:
price = f(tabular fields) + g(hidden quality tier), and the quality tier
is expressed ONLY in the text (not restated as tabular numbers) — this is
what makes "fusion needs both" a genuine, checkable property of the data
rather than a coincidence.

## 4. Modeling (`src/train.py`) — 3 models, identical algorithm

All 3 use `sklearn.linear_model.Ridge` with the same `alpha` — this is
deliberate: keeping the algorithm fixed across all 3 isolates the
leaderboard's differences to "which features were available," not "which
algorithm was used."

1. **`tabular_only`** — `StandardScaler`'d tabular features only.
2. **`text_only`** — `TfidfVectorizer` (max 300 features, English stop
   words removed) on `description` only.
3. **`fusion`** — the scaled tabular features and the TF-IDF text
   features concatenated (`scipy.sparse.hstack`) into one matrix.

TF-IDF is fit ONLY on the training split (leakage-safe, same discipline
as every other project in this series).

**Results on this build's synthetic data** (see `models/metrics.json`
for the live numbers): `tabular_only` R2≈0.616, `text_only` R2≈0.252,
`fusion` R2≈0.955 — fusion beats both by a wide margin, verified
directly rather than assumed (`AUDIT_REPORT.md`).

## 5. API contract (`app.py`)

| endpoint | method | purpose |
|---|---|---|
| `/` | GET | serves the frontend |
| `/api/health` | GET | liveness check |
| `/api/leaderboard` | GET | all 3 models' R2/MAE, best model, whether fusion beat both singles |
| `/api/predict` | POST | `{bedrooms, bathrooms, accommodates, room_type, neighborhood, description}` → `{tabular_only_prediction, text_only_prediction, fusion_prediction}` |

`room_type` and `neighborhood` are `Literal[...]` types in the Pydantic
request model, rejecting an invalid category with HTTP 422 before it
reaches feature encoding.

## 6. Frontend

Plain HTML/CSS/JS (no build step, no charting library): a leaderboard
bar chart (fusion highlighted), and a prediction form that shows all 3
models' predictions side by side for the SAME listing — so the value of
fusion is visible per-prediction, not just in an aggregate metric.

## 7. Verification performed

- `python src/train.py` — trains and compares all 3 models, printed
  leaderboard inspected manually
- Direct script check: two listings with identical tabular fields but
  different description text — confirmed `tabular_only_prediction` is
  bit-identical between them (structurally blind to text) while
  `fusion_prediction` differs by $132, entirely attributable to the text
- `uvicorn app:app` + `curl` against every endpoint, including an
  invalid `room_type` value — correct results and a clean 422
- `python -m pytest tests/` — 4/4 passing, including a test that fusion
  beats both single modalities and that tabular-only is provably blind
  to text differences

## 8. Explicit non-goals (kept out of scope per "not fancy, simple frontend and backend")

- No actual AutoGluon Multimodal dependency (see `PROMPTS.md`)
- No image modality (would need a vision backbone — a 3rd heavy dependency)
- No deep-learning text encoder (TF-IDF instead of embeddings/transformers)
- No hyperparameter search across the 3 models
- No authentication, database, or multi-user admin dashboard
- No containerization/CI — this is a local, single-process demo
