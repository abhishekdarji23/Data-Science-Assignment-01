# IMPLEMENTATION_PLANS.md — Project 06: Anomaly Detection (Transaction Risk Scoring)

## 1. Objective

Flag anomalous (potentially fraudulent) transactions from normal ones
using standard unsupervised anomaly detection methods, compare those
methods honestly against each other and against a simple statistical
baseline, and serve live scoring for a new transaction through a small
dashboard.

## 2. Architecture

```
┌────────────────────┐   GET /api/metrics, /api/anomalies    ┌──────────────────────┐
│  Browser (method     │ ─────────────────────────────────▶ │   FastAPI app.py     │
│  comparison +        │ ◀───────────────────────────────── │   (uvicorn)          │
│  review queue)       │   POST /api/score (new transaction)  └──────────┬───────────┘
└────────────────────┘                                                   │ loads
                                                                ┌──────────▼───────────┐
                                                                │ models/                │
                                                                │  isolation_forest      │
                                                                │   .joblib              │
                                                                │  scaler.joblib          │
                                                                │  metrics.json           │
                                                                │  scored_transactions.csv│
                                                                └──────────▲───────────┘
                                                                           │ produced by
                                                                ┌──────────┴───────────┐
                                                                │ src/train.py            │
                                                                │  (3 methods, offline)   │
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
| transaction_id | string | row id |
| amount_usd | float | transaction amount |
| hour_of_day | int (0-23) | when the transaction occurred |
| account_age_days | float | how old the account is |
| transactions_last_hour | int | recent transaction velocity |
| distance_from_home_km | float | geographic distance from the account's home location |
| is_anomaly | int (0/1) | ground truth, used ONLY for evaluation — never as a model input |

## 4. Modeling (`src/train.py`) — 3 methods, compared honestly

All 3 methods are fit **unsupervised** (never see `is_anomaly`), matching
real-world anomaly detection where large sets of confirmed-fraud labels
are rarely available for training:

1. **Isolation Forest** — isolates points via random recursive splits;
   anomalies need fewer splits to isolate. `sklearn.ensemble.IsolationForest`.
2. **Local Outlier Factor (LOF)** — density-based; flags points in
   sparser neighborhoods than their neighbors.
   `sklearn.neighbors.LocalOutlierFactor(novelty=True)`.
3. **Z-score baseline** — max absolute z-score across standardized
   features. A simple, transparent statistical method the two ML methods
   are expected to beat.

**Evaluation metric**: PR-AUC (average precision) is the headline metric,
appropriate for a rare positive class (3% anomaly rate here) — plain
accuracy would be misleadingly high for a model that flags nothing.
Precision/recall/F1 are also computed at a fixed "flag the top-K most
anomalous" threshold (K = known contamination rate × n), for a fair,
consistent comparison across methods with different score scales.

**Results on this build's synthetic data** (see `models/metrics.json`
for the live numbers):

| Method | PR-AUC | Precision@k | Recall@k |
|---|---|---|---|
| Isolation Forest | ≈0.99 | ≈0.96 | ≈0.96 |
| Z-score baseline | ≈0.95 | ≈0.91 | ≈0.91 |
| Local Outlier Factor | ≈0.10 | ≈0.09 | ≈0.09 |

LOF's poor performance is a genuine, documented finding — see
`AUDIT_REPORT.md` §3 for the explanation, not a bug.

**Production model**: Isolation Forest is saved and used for live scoring
(`/api/score`) — it won on PR-AUC AND can score a new point cheaply
without keeping the full training set in memory (unlike LOF, whose
`novelty=True` mode still needs its fitted neighbor structure).

## 5. API contract (`app.py`)

| endpoint | method | purpose |
|---|---|---|
| `/` | GET | serves the frontend |
| `/api/health` | GET | liveness check |
| `/api/metrics` | GET | model card: all 3 methods' PR-AUC/precision/recall/F1, dataset stats |
| `/api/anomalies` | GET | top-K flagged transactions from the training set (review queue, `?limit=N`) |
| `/api/score` | POST | `{amount_usd, hour_of_day, account_age_days, transactions_last_hour, distance_from_home_km}` → `{risk_score (0-100), flagged_as_anomaly, raw_isolation_forest_score}` |

## 6. Frontend

Plain HTML/CSS/JS (no build step, no charting library): a method
comparison table (best method starred), a live transaction-scoring form
with a color-coded risk result, and a review-queue table of the
highest-scored transactions from the training set (with ground truth
shown alongside, so you can see how well the flags line up).

## 7. Verification performed

- `python src/train.py` — trains and compares all 3 methods, printed
  comparison table inspected manually; sign convention for each method's
  anomaly score double-checked directly against `sklearn` docs (see
  `AUDIT_REPORT.md` §2)
- `uvicorn app:app` + `curl` against `/api/health`, `/api/metrics`,
  `/api/anomalies`, and `/api/score` with both an obviously-normal and an
  obviously-suspicious transaction — the suspicious one scored 97/100 and
  was flagged; the normal one scored 3.6/100 and was not
- `python -m pytest tests/` — 6/6 automated tests pass, including
  dedicated normal-vs-suspicious scoring tests

## 8. Explicit non-goals (kept out of scope per "not fancy, simple frontend and backend")

- No autoencoder / deep-learning anomaly detection method
- No One-Class SVM (a 4th popular method, left out to keep the comparison to 3)
- No "AutoResearch hill-climbing" hyperparameter search
- No research-paper-matched dashboard
- No authentication, database, or multi-user admin dashboard
- No containerization/CI — this is a local, single-process demo
