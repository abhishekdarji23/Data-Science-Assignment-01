"""
test_api.py
-----------
Minimal smoke tests. Run with: `python -m pytest tests/` from the project root
(after mining rules at least once).
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
    assert body["n_rules"] > 0
    assert body["n_transactions"] > 0


def test_items_catalog():
    r = client.get("/api/items")
    assert r.status_code == 200
    items = r.json()
    assert "bread" in items and "diapers" in items


def test_rules_shape():
    r = client.get("/api/rules?limit=5")
    assert r.status_code == 200
    rules = r.json()
    assert len(rules) <= 5
    for rule in rules:
        assert rule["lift"] >= 1.0  # min_lift filter enforced at train time


def test_recommend_classic_diapers_beer_pattern():
    """This pattern is deliberately baked into generate_data.py -- if this
    fails, either the data generator, the apriori thresholds, or the
    recommend() logic broke."""
    r = client.post("/api/recommend", json={"items": ["diapers"]})
    assert r.status_code == 200
    body = r.json()
    recommended_items = {rec["item"] for rec in body["recommendations"]}
    assert "beer" in recommended_items or "wipes" in recommended_items


def test_recommend_unknown_item_handled_gracefully():
    r = client.post("/api/recommend", json={"items": ["not_a_real_item"]})
    assert r.status_code == 200
    body = r.json()
    assert body["unknown_items"] == ["not_a_real_item"]
    assert body["recommendations"] == []


def test_recommend_empty_cart_rejected():
    r = client.post("/api/recommend", json={"items": []})
    assert r.status_code == 422
