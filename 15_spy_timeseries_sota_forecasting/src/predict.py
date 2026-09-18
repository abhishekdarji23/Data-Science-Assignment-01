"""
predict.py
----------
CRISP-DM Phase 6: Deployment (inference layer).

Produces a live next-trading-day forecast from all 5 methods, using the
full historical dataset as "history so far." ETS and ARIMA are refit
once per request (cheap: ~0.03s and ~0.15s respectively) rather than
using a persisted, potentially-stale fit.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

from data_prep import load_dataset, build_lag_features, get_feature_columns

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_gbm = None
_metrics = None


def _load():
    global _gbm, _metrics
    if _gbm is None:
        _gbm = joblib.load(MODELS_DIR / "gradient_boosting.joblib")
    if _metrics is None:
        with open(MODELS_DIR / "metrics.json") as f:
            _metrics = json.load(f)
    return _gbm, _metrics


def get_metrics() -> dict:
    _, metrics = _load()
    return metrics


def get_chart_data() -> list[dict]:
    df = pd.read_csv(MODELS_DIR / "test_period_chart_data.csv")
    return df.to_dict(orient="records")


def forecast_next_day() -> dict:
    gbm, metrics = _load()
    df = load_dataset()
    close = df["close"].values
    last_date = df["date"].iloc[-1]
    last_close = float(close[-1])

    # 1. naive
    pred_naive = last_close

    # 2. moving average
    ma_window = metrics["ma_window"]
    pred_ma = float(close[-ma_window:].mean())

    # 3. exponential smoothing (refit on full history)
    ets_fit = ExponentialSmoothing(list(close), trend="add", damped_trend=True,
                                    initialization_method="estimated").fit(optimized=True)
    pred_ets = float(ets_fit.forecast(1)[0])

    # 4. ARIMA (refit on full history, using the order chosen during training)
    order = tuple(metrics["arima_order"])
    arima_fit = ARIMA(close, order=order).fit()
    pred_arima = float(arima_fit.forecast(1)[0])

    # 5. gradient boosting (persisted model, one new feature row from the tail of history)
    df_feat = build_lag_features(df)
    feature_cols = get_feature_columns()
    last_row = df_feat[feature_cols].iloc[[-1]]
    pred_gbm = float(gbm.predict(last_row)[0])

    return {
        "as_of_date": str(last_date.date()),
        "last_close": round(last_close, 2),
        "forecasts": {
            "naive": round(pred_naive, 2),
            "moving_average": round(pred_ma, 2),
            "exponential_smoothing": round(pred_ets, 2),
            "arima": round(pred_arima, 2),
            "gradient_boosting": round(pred_gbm, 2),
        },
    }
