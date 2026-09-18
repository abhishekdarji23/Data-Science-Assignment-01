# WALKTHROUGH.md — Architectural Deep-Dive

> No video walkthrough is included with this build: recording and uploading
> a narrated video (e.g. to YouTube) isn't something this assistant can do —
> there's no video/audio generation or YouTube-upload capability available.
> This document is the substitute: a code-ordered tour you (or a screen
> recorder) can follow to produce one in a few minutes if you want it, plus
> a suggested shot list at the bottom.

## Reading order (this *is* the tour)

1. **`data/generate_data.py`** — where the data comes from. Read the
   module docstring first: synthetic because there's no Kaggle access
   here. Look at `ARCHETYPES`: 4 latent customer types with distinct
   (income, spending, age) centers and weights, each perturbed with
   Gaussian noise — so genuine, separable cluster structure exists for
   KMeans to rediscover (a pure-random dataset would make clustering
   results meaningless).

2. **`src/data_prep.py`** — the single source of truth for which columns
   are model inputs. Read the docstring's explanation of why `CustomerID`
   and `Gender` are excluded from `FEATURE_COLUMNS`.

3. **`src/train.py`** — the core of this project. Read in this order:
   - `StandardScaler().fit_transform(X)` — scale once, persist the scaler
   - the `for k in K_CANDIDATES` loop — fits KMeans at each k, records
     inertia (elbow data) and silhouette score
   - `best_k = argmax(silhouette)` — model selection
   - `label_segment()` — turns a cluster centroid into a business-readable
     label
   - the final block — saves `kmeans_model.joblib`, `scaler.joblib`,
     `metrics.json`, and `customers_with_clusters.csv`

4. **`src/predict.py`** — `predict_segment()`: loads the persisted model
   *and* the persisted scaler (never refits a new scaler at serving time —
   see the data_prep.py docstring for why that would be a bug), transforms
   the new customer's features, and returns the nearest cluster's
   human-readable profile.

5. **`app.py`** — FastAPI wiring: `/api/customers` returns every row with
   its cluster for the scatter plot; `/api/clusters` returns just the
   profile table; `/api/predict` calls `predict.predict_segment()`;
   `/api/metrics` is the full model card (silhouette-by-k, elbow curve,
   profiles).

6. **`static/app.js`** — `loadCustomersAndDrawScatter()` groups customers
   by cluster and renders a Chart.js scatter (income × spending, colored
   by segment); `loadClusterProfiles()` and `loadMetrics()` populate the
   two tables; the predict form posts to `/api/predict` and shows the
   assigned segment.

## How a request flows end-to-end

```
page loads
  → static/app.js fetches /api/customers, /api/clusters, /api/metrics
    → app.py reads models/customers_with_clusters.csv + metrics.json
  ← scatter plot + segment table + model card render

user fills "which segment" form, submits
  → static/app.js POSTs {age, annual_income_k, spending_score} to /api/predict
    → app.py: CustomerRequest validates bounds
      → predict.py: predict_segment() scales via the SAVED scaler,
        predicts nearest cluster via the SAVED KMeans model
      ← cluster id + human-readable label + profile
    ← JSON response
  ← static/app.js shows "Segment N — <label>"
```

## Suggested shot list (if you want to record your own walkthrough video)

1. `python data/generate_data.py` running, show the printed row count and archetype spread
2. `python src/train.py` running, show the silhouette-by-k printout and the chosen k
3. `uvicorn app:app --reload --port 8003` starting up
4. Browser: point at the scatter plot, note the 4 visually separated color groups
5. Browser: scroll to the segment profile table, explain 1-2 segment labels
6. Browser: fill the "which segment" form with a young/high-income/high-spend customer, submit, show it lands in the VIP segment
7. Terminal: `python -m pytest tests/` passing
8. Code editor: 30 seconds each on `data_prep.py`, `train.py`, `app.py` — the reading order above
