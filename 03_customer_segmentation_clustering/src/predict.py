"""
predict.py
----------
CRISP-DM Phase 6: Deployment (inference layer).

Assigns a NEW customer to the nearest existing cluster, using the SAME
scaler that was fit during training (loaded from disk, never refit here --
see the data_prep.py docstring for why that matters).
"""
import json
from pathlib import Path

import joblib

from data_prep import build_single_row

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_model = None
_scaler = None
_metrics = None


def _load():
    global _model, _scaler, _metrics
    if _model is None:
        _model = joblib.load(MODELS_DIR / "kmeans_model.joblib")
    if _scaler is None:
        _scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    if _metrics is None:
        with open(MODELS_DIR / "metrics.json") as f:
            _metrics = json.load(f)
    return _model, _scaler, _metrics


def get_metrics() -> dict:
    _, _, metrics = _load()
    return metrics


def get_cluster_profile(cluster_id: int) -> dict | None:
    _, _, metrics = _load()
    for p in metrics["cluster_profiles"]:
        if p["cluster"] == cluster_id:
            return p
    return None


def predict_segment(age: float, annual_income_k: float, spending_score: float) -> dict:
    model, scaler, _ = _load()
    X = build_single_row(age, annual_income_k, spending_score)
    X_scaled = scaler.transform(X)
    cluster_id = int(model.predict(X_scaled)[0])
    profile = get_cluster_profile(cluster_id)
    return {
        "cluster": cluster_id,
        "segment_label": profile["label"] if profile else "Unknown",
        "cluster_profile": profile,
    }
