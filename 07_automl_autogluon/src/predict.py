"""
predict.py
----------
CRISP-DM Phase 6: Deployment (inference layer).

Scores a NEW customer by running it through every base model (the L1
layer), then feeding those predictions into the meta-learner (the L2
layer) -- reproducing the exact same stacking architecture used in
training, at serving time.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from data_prep import build_features, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_base_models = None
_meta_learner = None
_scaler = None
_metrics = None


def _load():
    global _base_models, _meta_learner, _scaler, _metrics
    if _metrics is None:
        with open(MODELS_DIR / "metrics.json") as f:
            _metrics = json.load(f)
    if _base_models is None:
        _base_models = {
            name: joblib.load(MODELS_DIR / f"base_{name}.joblib")
            for name in _metrics["model_zoo"]
        }
    if _meta_learner is None:
        _meta_learner = joblib.load(MODELS_DIR / "meta_learner.joblib")
    if _scaler is None:
        _scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    return _base_models, _meta_learner, _scaler, _metrics


def get_metrics() -> dict:
    _, _, _, metrics = _load()
    return metrics


def predict_churn(
    tenure_months: float, monthly_charges: float, total_charges: float,
    contract_type: str, internet_service: str, tech_support: str,
    payment_method: str, num_support_calls: int,
) -> dict:
    base_models, meta_learner, scaler, metrics = _load()

    row = pd.DataFrame([{
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "payment_method": payment_method,
        "num_support_calls": num_support_calls,
    }])
    X = build_features(row)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

    # L1: every base model's opinion
    base_predictions = {}
    for name, model in base_models.items():
        base_predictions[name] = round(float(model.predict_proba(X_scaled)[0][1]), 4)

    # L2: meta-learner combines the base models' opinions
    meta_X = pd.DataFrame([base_predictions])[list(base_models.keys())]
    stacked_proba = float(meta_learner.predict_proba(meta_X)[0][1])

    return {
        "churn_probability": round(stacked_proba, 4),
        "risk_level": "High" if stacked_proba >= 0.5 else ("Medium" if stacked_proba >= 0.3 else "Low"),
        "base_model_predictions": base_predictions,
        "leaderboard_best_model": metrics["best_model"],
    }
