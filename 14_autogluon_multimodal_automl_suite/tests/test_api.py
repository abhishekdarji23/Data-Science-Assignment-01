"""
test_api.py
-----------
Minimal smoke tests, plus the test that actually matters for this
project: that fusion genuinely outperforms both single modalities, and
that a tabular-only model is provably blind to the text signal.
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


def test_leaderboard_fusion_beats_both_single_modalities():
    r = client.get("/api/leaderboard")
    assert r.status_code == 200
    body = r.json()
    lb = body["leaderboard"]
    assert lb["fusion"]["r2"] > lb["tabular_only"]["r2"]
    assert lb["fusion"]["r2"] > lb["text_only"]["r2"]
    assert body["best_model"] == "fusion"
    assert body["fusion_beats_both_singles"] is True


def test_tabular_only_is_blind_to_text_signal():
    """Same tabular fields, different text quality signal -- tabular-only
    prediction must be IDENTICAL (it structurally cannot see the text),
    while fusion must differ."""
    base = {
        "bedrooms": 2, "bathrooms": 1, "accommodates": 4,
        "room_type": "Entire home/apt", "neighborhood": "Downtown",
    }
    luxury = client.post("/api/predict", json={
        **base, "description": "Stunning skyline view. Designer furnished. Private rooftop access.",
    }).json()
    budget = client.post("/api/predict", json={
        **base, "description": "A bit dated but functional. Basic amenities. No frills stay.",
    }).json()

    assert luxury["tabular_only_prediction"] == budget["tabular_only_prediction"]
    assert luxury["fusion_prediction"] > budget["fusion_prediction"]


def test_predict_invalid_room_type_rejected():
    r = client.post("/api/predict", json={
        "bedrooms": 2, "bathrooms": 1, "accommodates": 4,
        "room_type": "Castle", "neighborhood": "Downtown", "description": "nice place",
    })
    assert r.status_code == 422
