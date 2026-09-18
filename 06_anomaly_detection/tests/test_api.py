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
    assert "isolation_forest" in body["methods_compared"]
    assert "local_outlier_factor" in body["methods_compared"]
    assert "z_score_baseline" in body["methods_compared"]
    assert body["methods_compared"]["isolation_forest"]["pr_auc"] > 0.5


def test_anomalies_review_queue():
    r = client.get("/api/anomalies?limit=10")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 10
    for row in rows:
        assert row["flagged"] == 1


def test_score_obviously_normal_transaction_low_risk():
    payload = {
        "amount_usd": 40, "hour_of_day": 14, "account_age_days": 500,
        "transactions_last_hour": 1, "distance_from_home_km": 2,
    }
    r = client.post("/api/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["flagged_as_anomaly"] is False
    assert body["risk_score"] < 50


def test_score_obviously_suspicious_transaction_high_risk():
    payload = {
        "amount_usd": 1200, "hour_of_day": 3, "account_age_days": 3,
        "transactions_last_hour": 9, "distance_from_home_km": 300,
    }
    r = client.post("/api/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["flagged_as_anomaly"] is True
    assert body["risk_score"] > 50


def test_score_out_of_bounds_rejected():
    payload = {
        "amount_usd": -5,  # negative amount is invalid
        "hour_of_day": 14, "account_age_days": 400,
        "transactions_last_hour": 1, "distance_from_home_km": 3,
    }
    r = client.post("/api/score", json=payload)
    assert r.status_code == 422
