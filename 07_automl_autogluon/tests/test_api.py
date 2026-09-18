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


def test_leaderboard_has_all_models_plus_ensemble():
    r = client.get("/api/leaderboard")
    assert r.status_code == 200
    body = r.json()
    lb = body["leaderboard"]
    assert "stacked_ensemble" in lb
    assert len(lb) == 6  # 5 base models + 1 stacked ensemble
    for model_metrics in lb.values():
        assert 0 <= model_metrics["roc_auc"] <= 1


def test_predict_high_risk_customer():
    payload = {
        "tenure_months": 2, "monthly_charges": 95, "total_charges": 190,
        "contract_type": "Month-to-month", "internet_service": "Fiber optic",
        "tech_support": "No", "payment_method": "Electronic check", "num_support_calls": 4,
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["risk_level"] == "High"
    assert body["churn_probability"] > 0.5
    assert len(body["base_model_predictions"]) == 5


def test_predict_low_risk_customer():
    payload = {
        "tenure_months": 48, "monthly_charges": 45, "total_charges": 2160,
        "contract_type": "Two year", "internet_service": "DSL",
        "tech_support": "Yes", "payment_method": "Bank transfer", "num_support_calls": 0,
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["risk_level"] == "Low"
    assert body["churn_probability"] < 0.3


def test_predict_invalid_contract_type_rejected():
    payload = {
        "tenure_months": 2, "monthly_charges": 95, "total_charges": 190,
        "contract_type": "Not a real contract", "internet_service": "Fiber optic",
        "tech_support": "No", "payment_method": "Electronic check", "num_support_calls": 4,
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 422
