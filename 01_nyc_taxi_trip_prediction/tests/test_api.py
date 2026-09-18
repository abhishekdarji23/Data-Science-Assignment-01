"""
test_api.py
-----------
Minimal smoke tests. Run with: `python -m pytest tests/` from the project root
(after training the model at least once).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metrics():
    r = client.get("/api/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "mae_seconds" in body
    assert body["n_train"] > 0


def test_predict_happy_path():
    payload = {
        "pickup_lat": 40.758, "pickup_lon": -73.9855,
        "dropoff_lat": 40.7306, "dropoff_lon": -73.9352,
        "passenger_count": 2,
        "pickup_datetime": "2024-06-14T08:30:00",
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_duration_seconds"] > 0
    assert body["distance_miles"] > 0
    assert body["estimated_fare_usd"] >= 8.0


def test_predict_out_of_bounds_rejected():
    payload = {
        "pickup_lat": 10.0, "pickup_lon": -73.9855,  # way outside NYC bbox
        "dropoff_lat": 40.7306, "dropoff_lon": -73.9352,
        "passenger_count": 2,
        "pickup_datetime": "2024-06-14T08:30:00",
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 422  # pydantic field validation error
