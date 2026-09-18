# IMPLEMENTATION_PLANS.md — Project 04: Market Basket Pattern Mining

## 1. Objective

Mine "customers who bought X also bought Y" rules from grocery
transaction data (Apriori algorithm), and serve live recommendations for
a shopping cart in progress, in the style of classic market-basket-analysis
tutorials built on the Kaggle "Groceries" / "Online Retail" datasets.

## 2. Architecture

```
┌────────────────────┐   GET /api/items, /api/rules          ┌──────────────────────┐
│  Browser (item      │ ──────────────────────────────────▶ │   FastAPI app.py     │
│  picker + tables)   │ ◀────────────────────────────────── │   (uvicorn)          │
└────────────────────┘   POST /api/recommend (cart items)     └──────────┬───────────┘
                                                                          │ loads
                                                               ┌──────────▼───────────┐
                                                               │ models/               │
                                                               │  rules.json           │
                                                               │  metrics.json         │
                                                               │  item_catalog.json    │
                                                               └──────────▲───────────┘
                                                                          │ produced by
                                                               ┌──────────┴───────────┐
                                                               │ src/train.py          │
                                                               │  (apriori, offline)   │
                                                               └──────────▲───────────┘
                                                                          │ reads
                                                               ┌──────────┴───────────┐
                                                               │ data/*.csv            │
                                                               │ (synthetic or real)   │
                                                               └───────────────────────┘
```

No database, no ML model artifact in the usual sense — the mined rule
list *is* the deployment unit. This matches how market basket analysis
is normally run in practice: rules are mined periodically in a batch job,
then served cheaply (a lookup, not a live inference) at request time.

## 3. Data schema

Long format, one row per (transaction, item) pair — the standard input
shape for `mlxtend.frequent_patterns.apriori` after pivoting:

| column | type | notes |
|---|---|---|
| transaction_id | int | repeats once per item in that basket |
| item | string | grocery item name |

## 4. Data preparation (`src/data_prep.py`)

`build_basket_matrix()` pivots the long-format transactions into a
one-hot **basket matrix**: rows = transactions, columns = items, values =
item present/absent (boolean). This is the input shape `apriori()` requires.

## 5. Modeling (`src/train.py`)

- **Algorithm**: Apriori (`mlxtend.frequent_patterns.apriori`) to find
  frequent itemsets, then `association_rules()` to derive
  antecedent→consequent rules with support/confidence/lift.
- **Thresholds** (documented, not hidden): `min_support=0.03`,
  `min_confidence=0.30`, `min_lift=1.1`. Lift is the key ranking metric —
  it corrects for popular items showing up in most rules regardless of
  actual association (see the `train.py` docstring for the exact
  definitions of support/confidence/lift).
- **"Evaluation"**: unlike supervised learning, there's no held-out test
  set — the sanity check is that mined rules are non-trivial (lift > 1)
  and match expected real-world patterns (see the Audit Report).

## 6. API contract (`app.py`)

| endpoint | method | purpose |
|---|---|---|
| `/` | GET | serves the frontend |
| `/api/health` | GET | liveness check |
| `/api/metrics` | GET | model card: dataset stats, mining thresholds, top items by support |
| `/api/rules` | GET | top rules by lift (`?limit=N`) |
| `/api/items` | GET | full item catalog, for the cart-picker UI |
| `/api/recommend` | POST | `{items: [...]}` → matching rules' consequents, ranked by lift, with an explanation of which cart item triggered each recommendation |

## 7. Frontend

Plain HTML/CSS/JS (no build step, no charting library needed here): a
clickable item-chip grid to build a cart, live recommendation cards that
update on every click, and a static top-rules table.

## 8. Verification performed

- `python src/train.py` — mines rules, printed thresholds and top-5-by-lift rules inspected manually; confirms the deliberately baked-in `diapers → beer/wipes` pattern surfaces with the highest lift in the dataset
- `uvicorn app:app` + `curl` against `/api/health`, `/api/items`, `/api/rules`, `/api/recommend` (including an unknown-item case) — all returned expected shapes and sensible recommendations
- `python -m pytest tests/` — 7/7 automated tests pass, including a dedicated test that the diapers→beer/wipes rule actually surfaces through the API

## 9. Explicit non-goals (kept out of scope per "not fancy" / "simple frontend and backend")

- No FP-Growth comparison or performance benchmarking between algorithms
- No AutoML / hyperparameter search / "AutoResearch hill-climbing"
- No graph/network visualization of item co-occurrence
- No authentication, database, or multi-user admin dashboard
- No containerization/CI — this is a local, single-process demo
