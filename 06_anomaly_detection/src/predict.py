"""
predict.py
----------
CRISP-DM Phase 6: Deployment (inference layer).

Scores a NEW transaction with the production model (Isolation Forest --
chosen in train.py for both accuracy on this dataset and because it can
score a new point cheaply without needing the training set kept in memory,
unlike LOF).
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from data_prep import FEATURE_COLUMNS

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_model = None
_scaler = None
_metrics = None

# A raw Isolation Forest decision_function score is not intuitive on its own
# (its scale depends on the forest); we rescale it to a 0-100 "risk score"
# using the training set's observed score range, purely for readability in
# the UI. The underlying ranking (and the flag decision) is unaffected.
_score_min = None
_score_max = None


def _load():
    global _model, _scaler, _metrics, _score_min, _score_max
    if _model is None:
        _model = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    if _scaler is None:
        _scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    if _metrics is None:
        with open(MODELS_DIR / "metrics.json") as f:
            _metrics = json.load(f)
    if _score_min is None:
        scored = pd.read_csv(MODELS_DIR / "scored_transactions.csv")
        _score_min = float(scored["anomaly_score"].min())
        _score_max = float(scored["anomaly_score"].max())
    return _model, _scaler, _metrics


def get_metrics() -> dict:
    _, _, metrics = _load()
    return metrics


def get_top_anomalies(limit: int = 20) -> list:
    scored = pd.read_csv(MODELS_DIR / "scored_transactions.csv")
    top = scored[scored["flagged"] == 1].head(limit)
    return top.to_dict(orient="records")


def _risk_score(raw_anomaly_score: float) -> float:
    span = max(_score_max - _score_min, 1e-9)
    pct = (raw_anomaly_score - _score_min) / span
    return round(max(0.0, min(1.0, pct)) * 100, 1)


def score_transaction(
    amount_usd: float, hour_of_day: int, account_age_days: float,
    transactions_last_hour: int, distance_from_home_km: float,
) -> dict:
    model, scaler, metrics = _load()
    X = pd.DataFrame([{
        "amount_usd": amount_usd,
        "hour_of_day": hour_of_day,
        "account_age_days": account_age_days,
        "transactions_last_hour": transactions_last_hour,
        "distance_from_home_km": distance_from_home_km,
    }])[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)

    raw_score = float(-model.decision_function(X_scaled)[0])  # higher = more anomalous
    is_flagged = bool(model.predict(X_scaled)[0] == -1)  # sklearn: -1 = outlier, 1 = inlier

    return {
        "risk_score": _risk_score(raw_score),
        "flagged_as_anomaly": is_flagged,
        "raw_isolation_forest_score": round(raw_score, 4),
        "model_contamination_setting": metrics["contamination_rate"],
    }
