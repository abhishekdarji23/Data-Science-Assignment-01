# WALKTHROUGH.md — Architectural Deep-Dive

> No video walkthrough is included with this build: recording and uploading
> a narrated video (e.g. to YouTube) isn't something this assistant can do —
> there's no video/audio generation or YouTube-upload capability available.
> This document is the substitute: a code-ordered tour you (or a screen
> recorder) can follow to produce one in a few minutes if you want it, plus
> a suggested shot list at the bottom.

## Reading order (this *is* the tour)

1. **`data/generate_data.py`** — where the data comes from. Read the module
   docstring first: it explains why the data is synthetic here (no Kaggle
   access) and exactly how to swap in the real dataset. Look at
   `generate()`: pickup points are drawn from a 2D Gaussian centered on
   Manhattan; dropoff points are an exponential-radius offset (mimics many
   short trips, few long ones); `trip_duration` is derived from a genuine
   distance ÷ speed physics model with rush-hour/weekend/noise terms — not
   just random numbers, so the regression task is real.

2. **`src/data_prep.py`** — the single source of truth for feature
   engineering, imported by *both* training and inference (`train.py` and
   `predict.py`) so there's no train/serve skew. Read `add_engineered_features()`
   to see the derived columns (`distance_mi`, `bearing_deg`, `hour`,
   `is_rush_hour`, …), and the module docstring's leakage note.

3. **`src/train.py`** — `time_based_split()` first (why chronological, not
   random), then `main()`: log1p target transform → fit
   `GradientBoostingRegressor` → inverse-transform predictions → compute
   MAE/RMSE/RMSLE/R² → dump `models/trip_duration_model.joblib` +
   `models/metrics.json`.

4. **`src/predict.py`** — `predict_trip()`: loads the saved model once
   (module-level cache), builds one feature row via `data_prep`, and adds a
   transparent, non-ML fare estimator (`estimate_fare()`) so the UI can show
   a fare without pretending it was learned from data.

5. **`app.py`** — FastAPI wiring: `TripRequest` (Pydantic) bounds-checks
   coordinates to the NYC bbox; `/api/predict` calls `predict.predict_trip()`;
   `/api/metrics` exposes the training metrics as a lightweight "model card";
   static files are mounted for the frontend.

6. **`static/app.js`** — `map.on("click", …)` collects pickup then dropoff
   as Leaflet `circleMarker`s; form submit posts to `/api/predict` and
   renders the response; a separate `fetch("/api/metrics")` populates the
   model-card table on page load.

## How a request flows end-to-end

```
user clicks map twice (pickup, dropoff)
  → static/app.js reads form fields, POSTs JSON to /api/predict
    → app.py: TripRequest validates bounds
      → predict.py: predict_trip() builds features via data_prep.py
        → model.predict() on the saved GradientBoostingRegressor
      ← duration (expm1 back to seconds), haversine distance, fare estimate
    ← JSON response
  ← static/app.js renders the 4 result cards
```

## Suggested shot list (if you want to record your own walkthrough video)

1. `python data/generate_data.py` running, show the printed row count
2. `python src/train.py` running, show the metrics JSON printed to terminal
3. `uvicorn app:app --reload --port 8001` starting up
4. Browser: click two points on the map, submit the form, show the result cards
5. Browser: scroll to the model-card panel, point out MAE/RMSE/R²
6. Terminal: `python -m pytest tests/` passing
7. Code editor: 30 seconds each on `data_prep.py`, `train.py`, `app.py` — the reading order above
