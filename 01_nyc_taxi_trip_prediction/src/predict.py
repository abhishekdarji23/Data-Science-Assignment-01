"""
predict.py
----------
CRISP-DM Phase 6: Deployment (inference layer).

Thin wrapper around the trained model so both the FastAPI app and any
CLI/tests can get a prediction the same way.
"""
import json
from pathlib import Path

import joblib
import numpy as np

from data_prep import build_single_feature_row, haversine_miles

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_model = None
_metrics = None


def _load():
    global _model, _metrics
    if _model is None:
        _model = joblib.load(MODELS_DIR / "trip_duration_model.joblib")
    if _metrics is None:
        with open(MODELS_DIR / "metrics.json") as f:
            _metrics = json.load(f)
    return _model, _metrics


def get_metrics() -> dict:
    _, metrics = _load()
    return metrics


# Simple, transparent fare estimator (NOT part of the ML model -- this is
# a deterministic formula loosely based on NYC TLC's published yellow-cab
# rate card, so the UI can show a "fare estimate" alongside the ML-predicted
# duration without pretending the fare itself was learned from data).
BASE_FARE = 3.00
PER_MILE = 1.75
PER_MINUTE = 0.50
MIN_FARE = 8.00


def estimate_fare(distance_mi: float, duration_sec: float) -> float:
    fare = BASE_FARE + PER_MILE * distance_mi + PER_MINUTE * (duration_sec / 60.0)
    return round(max(fare, MIN_FARE), 2)


def predict_trip(
    pickup_lat: float, pickup_lon: float,
    dropoff_lat: float, dropoff_lon: float,
    passenger_count: int, pickup_datetime: str,
) -> dict:
    model, _ = _load()
    X = build_single_feature_row(
        pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
        passenger_count, pickup_datetime,
    )
    pred_log = model.predict(X)[0]
    duration_sec = float(np.expm1(pred_log))
    duration_sec = max(duration_sec, 30.0)

    distance_mi = float(haversine_miles(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon))
    fare = estimate_fare(distance_mi, duration_sec)

    return {
        "predicted_duration_seconds": round(duration_sec, 1),
        "predicted_duration_minutes": round(duration_sec / 60.0, 1),
        "distance_miles": round(distance_mi, 2),
        "estimated_fare_usd": fare,
        "avg_speed_mph": round(distance_mi / (duration_sec / 3600.0), 1) if duration_sec > 0 else None,
    }
