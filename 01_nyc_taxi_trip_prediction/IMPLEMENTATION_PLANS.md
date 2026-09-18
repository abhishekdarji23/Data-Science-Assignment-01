# IMPLEMENTATION_PLANS.md — Project 01: NYC Taxi Trip Duration Predictor

## 1. Objective

Predict a taxi trip's duration (seconds) from information known **before**
the trip starts — pickup point, requested dropoff point, passenger count,
and pickup time — and surface it through a small interactive web app with
a map-based UX, matching the shape of the Kaggle
["New York City Taxi Trip Duration"](https://www.kaggle.com/c/nyc-taxi-trip-duration)
competition.

## 2. Architecture

```
┌────────────────────┐      POST /api/predict        ┌──────────────────────┐
│  Browser (Leaflet   │ ─────────────────────────────▶│   FastAPI app.py     │
│  map + plain JS UI) │ ◀───────────────────────────── │   (uvicorn)          │
└────────────────────┘      JSON prediction            └──────────┬───────────┘
                                                                   │ loads
                                                        ┌──────────▼───────────┐
                                                        │ models/               │
                                                        │  trip_duration_model  │
                                                        │  .joblib + metrics    │
                                                        │  .json                │
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

No database is used — the model artifact on disk *is* the deployment unit,
which is appropriate at this scale (single model, batch-retrained offline).

## 3. Data schema

Matches the real Kaggle dataset exactly, so a real `train.csv` can be
dropped in without code changes:

| column | type | notes |
|---|---|---|
| id | string | row id |
| vendor_id | int | 1 or 2 |
| pickup_datetime | datetime | ISO timestamp |
| dropoff_datetime | datetime | not used as a model input (see leakage note) |
| passenger_count | int | 1–6 |
| pickup_longitude / pickup_latitude | float | NYC bounding box |
| dropoff_longitude / dropoff_latitude | float | NYC bounding box |
| store_and_fwd_flag | 'Y'/'N' | not used as a feature (near-constant, low signal) |
| trip_duration | int (seconds) | **target** |

## 4. Feature engineering (`src/data_prep.py`)

All features are derivable at request time (before the trip happens):

- `distance_mi` — haversine great-circle distance, pickup → dropoff
- `manhattan_distance_mi` — grid/L1 distance (approximates NYC street grid)
- `bearing_deg` — compass direction of travel
- `hour`, `day_of_week`, `month`, `is_weekend`, `is_rush_hour` — from `pickup_datetime`
- raw `pickup_latitude/longitude`, `dropoff_latitude/longitude`
- `passenger_count`

## 5. Model (`src/train.py`)

- **Algorithm**: `sklearn.ensemble.GradientBoostingRegressor` (250 trees, depth 4, lr 0.08) — chosen for strong tabular performance without a GPU/large dependency footprint, appropriate for "not fancy."
- **Target transform**: `log1p(trip_duration)` (duration is right-skewed); predictions are `expm1`-transformed back to seconds before evaluation.
- **Split**: time-based 80/20 (train = earliest 80% chronologically, test = most recent 20%) — a random shuffle would leak future traffic patterns backward in time.
- **Metrics** (on this build's synthetic data, see `models/metrics.json` for the live numbers): MAE ≈ 81s, RMSE ≈ 122s, RMSLE ≈ 0.30, R² ≈ 0.89.

## 6. API contract (`app.py`)

| endpoint | method | purpose |
|---|---|---|
| `/` | GET | serves the frontend |
| `/api/health` | GET | liveness check |
| `/api/metrics` | GET | model card: training metrics + feature importances |
| `/api/predict` | POST | `{pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, passenger_count, pickup_datetime}` → `{predicted_duration_seconds, predicted_duration_minutes, distance_miles, estimated_fare_usd, avg_speed_mph}` |

Request fields are bounds-checked against the NYC bounding box via Pydantic
(`app.py: TripRequest`), returning HTTP 422 on out-of-range coordinates.

## 7. Frontend

Plain HTML/CSS/JS (no build step), using Leaflet.js (via CDN) for the map.
Two clicks set pickup/dropoff; the form posts to `/api/predict`; a "model
card" panel calls `/api/metrics` so a reviewer can see training metrics
without leaving the page.

## 8. Verification performed

- `python src/train.py` — trains and writes `models/*.joblib` + `metrics.json`, printed metrics inspected manually
- `uvicorn app:app` + `curl` against `/api/health`, `/api/metrics`, `/api/predict` — all returned expected shapes/values
- `python -m pytest tests/` — 4/4 automated tests pass (health, metrics, happy-path prediction, out-of-bounds rejection)

## 9. Explicit non-goals (kept out of scope per "not fancy")

- No React/TypeScript build pipeline
- No AutoML / hyperparameter search / "AutoResearch hill-climbing"
- No authentication, database, or multi-user admin dashboard
- No containerization/CI — this is a local, single-process demo
