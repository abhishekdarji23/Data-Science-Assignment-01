# IMPLEMENTATION_PLANS.md — Project 03: Customer Segmentation Clustering

## 1. Objective

Group customers into behaviorally meaningful segments from Age, Annual
Income, and Spending Score, so a business can target marketing differently
per segment (e.g. "high income, low spend" customers might respond to a
different offer than existing "VIP" spenders). Matches the shape of the
classic Kaggle
["Mall Customer Segmentation"](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
dataset/tutorial.

## 2. Architecture

```
┌────────────────────┐   GET /api/customers, /api/clusters   ┌──────────────────────┐
│  Browser (Chart.js  │ ──────────────────────────────────▶ │   FastAPI app.py     │
│  scatter + tables)  │ ◀────────────────────────────────── │   (uvicorn)          │
└────────────────────┘   POST /api/predict (new customer)    └──────────┬───────────┘
                                                                          │ loads
                                                               ┌──────────▼───────────┐
                                                               │ models/               │
                                                               │  kmeans_model.joblib  │
                                                               │  scaler.joblib        │
                                                               │  metrics.json         │
                                                               │  customers_with_      │
                                                               │   clusters.csv        │
                                                               └──────────▲───────────┘
                                                                          │ produced by
                                                               ┌──────────┴───────────┐
                                                               │ src/train.py          │
                                                               │  (offline, one-time)  │
                                                               └──────────▲───────────┘
                                                                          │ reads
                                                               ┌──────────┴───────────┐
                                                               │ data/*.csv            │
                                                               │ (synthetic or real)   │
                                                               └───────────────────────┘
```

No database — the persisted model, scaler, and pre-scored customer CSV
*are* the deployment unit, appropriate for a single offline-trained model.

## 3. Data schema

Matches the real Kaggle "Mall_Customers.csv" exactly, so a real download
can be dropped in without code changes:

| column | type | notes |
|---|---|---|
| CustomerID | int | arbitrary id, excluded from features |
| Gender | string | excluded from features (see `data_prep.py` docstring) |
| Age | int | 18–75 in this build |
| Annual Income (k$) | int | in thousands of dollars |
| Spending Score (1-100) | int | store-assigned behavioral score |

## 4. Feature engineering (`src/data_prep.py`)

`FEATURE_COLUMNS = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]`
— deliberately excludes `CustomerID` (arbitrary identifier, no behavioral
meaning) and `Gender` (categorical; the numeric features already separate
the archetypes cleanly, and clustering on a demographic attribute directly
risks building segments around it rather than behavior).

## 5. Model (`src/train.py`)

- **Algorithm**: `sklearn.cluster.KMeans` — the standard, interpretable choice for this kind of small tabular segmentation task.
- **Preprocessing**: `StandardScaler` fit once on the full feature set (see leakage note below), persisted to `models/scaler.joblib` so training and serving always use the identical transform.
- **Model selection (choosing k)**: fit KMeans for k = 2..8, compute the silhouette score for each, and pick `argmax(silhouette)`. Inertia (elbow-method data) is also recorded for a human-readable diagnostic, but silhouette is the deciding metric.
- **Result on this build's synthetic data**: chosen k = **4**, silhouette ≈ **0.40** (see `models/metrics.json` for the live numbers) — matching the 4 latent archetypes the synthetic generator was built from.
- **Cluster labeling**: `label_segment()` turns each cluster's centroid (in original, unscaled units) into a human-readable business label — e.g. "High Income, High Spend (VIP)" — so the dashboard doesn't force a reader to interpret raw centroid numbers.

## 6. API contract (`app.py`)

| endpoint | method | purpose |
|---|---|---|
| `/` | GET | serves the frontend |
| `/api/health` | GET | liveness check |
| `/api/metrics` | GET | model card: silhouette scores by k, elbow curve, cluster profiles |
| `/api/clusters` | GET | just the cluster-profile table |
| `/api/customers` | GET | every customer with its assigned cluster (for the scatter plot) |
| `/api/predict` | POST | `{age, annual_income_k, spending_score}` → `{cluster, segment_label, cluster_profile}` |

## 7. Frontend

Plain HTML/CSS/JS (no build step), using Chart.js (via CDN) for a scatter
plot of Income vs. Spending Score colored by segment, a segment-profile
table, and a form to assign a new hypothetical customer to a segment.

## 8. Verification performed

- `python src/train.py` — trains and writes `models/*.joblib`, `metrics.json`, `customers_with_clusters.csv`; silhouette/cluster sizes inspected manually
- `uvicorn app:app` + `curl` against `/api/health`, `/api/clusters`, `/api/predict`, `/api/customers` — all returned expected shapes and sensible cluster assignments (e.g. a young, high-income, high-spending test customer was assigned to the "VIP" segment)
- `python -m pytest tests/` — 6/6 automated tests pass (health, metrics, cluster shape, customer list, plausible prediction, boundary rejection)

## 9. Explicit non-goals (kept out of scope per "not fancy")

- No React/TypeScript build pipeline
- No AutoML / hyperparameter search / "AutoResearch hill-climbing" beyond the k=2..8 silhouette sweep
- No hierarchical/DBSCAN comparison, no dimensionality reduction (PCA/t-SNE) view
- No authentication, database, or multi-user admin dashboard
- No containerization/CI — this is a local, single-process demo
