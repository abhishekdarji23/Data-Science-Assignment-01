"""
predict.py
----------
CRISP-DM Phase 6: Deployment (inference layer).

Scores a NEW listing through all 3 models (tabular-only, text-only,
fusion) so the API/UI can show them side by side -- the point being to
make the value of fusion visible per-listing, not just in an aggregate
metric.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from data_prep import build_tabular_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_scaler = None
_vectorizer = None
_model_tabular = None
_model_text = None
_model_fusion = None
_metrics = None


def _load():
    global _scaler, _vectorizer, _model_tabular, _model_text, _model_fusion, _metrics
    if _scaler is None:
        _scaler = joblib.load(MODELS_DIR / "tabular_scaler.joblib")
    if _vectorizer is None:
        _vectorizer = joblib.load(MODELS_DIR / "text_vectorizer.joblib")
    if _model_tabular is None:
        _model_tabular = joblib.load(MODELS_DIR / "model_tabular_only.joblib")
    if _model_text is None:
        _model_text = joblib.load(MODELS_DIR / "model_text_only.joblib")
    if _model_fusion is None:
        _model_fusion = joblib.load(MODELS_DIR / "model_fusion.joblib")
    if _metrics is None:
        with open(MODELS_DIR / "metrics.json") as f:
            _metrics = json.load(f)
    return _scaler, _vectorizer, _model_tabular, _model_text, _model_fusion, _metrics


def get_metrics() -> dict:
    *_, metrics = _load()
    return metrics


def predict_price(
    bedrooms: int, bathrooms: int, accommodates: int,
    room_type: str, neighborhood: str, description: str,
) -> dict:
    scaler, vectorizer, model_tabular, model_text, model_fusion, _ = _load()

    row = pd.DataFrame([{
        "bedrooms": bedrooms, "bathrooms": bathrooms, "accommodates": accommodates,
        "room_type": room_type, "neighborhood": neighborhood,
    }])
    X_tab = build_tabular_features(row)
    X_tab_scaled = scaler.transform(X_tab)
    X_text = vectorizer.transform([description])
    X_fusion = hstack([csr_matrix(X_tab_scaled), X_text])

    pred_tabular = float(model_tabular.predict(X_tab_scaled)[0])
    pred_text = float(model_text.predict(X_text)[0])
    pred_fusion = float(model_fusion.predict(X_fusion)[0])

    return {
        "tabular_only_prediction": round(max(pred_tabular, 0), 2),
        "text_only_prediction": round(max(pred_text, 0), 2),
        "fusion_prediction": round(max(pred_fusion, 0), 2),
    }
