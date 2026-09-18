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
    assert body["chosen_k"] >= 2
    assert body["chosen_k_silhouette"] > 0


def test_clusters_shape():
    r = client.get("/api/clusters")
    assert r.status_code == 200
    profiles = r.json()
    assert len(profiles) >= 2
    for p in profiles:
        assert "label" in p and "size" in p


def test_customers_list():
    r = client.get("/api/customers")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) > 0
    assert "cluster" in rows[0]


def test_predict_high_income_high_spend_is_vip_like():
    payload = {"age": 28, "annual_income_k": 95, "spending_score": 90}
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "cluster" in body
    assert "High" in body["segment_label"] or "VIP" in body["segment_label"]


def test_predict_out_of_bounds_rejected():
    payload = {"age": 200, "annual_income_k": 60, "spending_score": 50}
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 422
