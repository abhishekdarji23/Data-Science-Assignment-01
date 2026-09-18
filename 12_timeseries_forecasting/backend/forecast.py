"""
A simple, from-scratch forecaster: classical additive decomposition
(trend via centered moving average, seasonality via average day-of-week
deviation, everything else is residual), then a linear extrapolation of
the trend plus the repeating seasonal pattern for the forecast horizon.
Prediction intervals widen with the square root of the horizon (a rough
but standard way to reflect growing uncertainty further into the future).
"""

import math
from datetime import date, timedelta

SEASON_LENGTH = 7  # weekly seasonality


def _moving_average(values, window):
    half = window // 2
    result = [None] * len(values)
    for i in range(half, len(values) - half):
        result[i] = sum(values[i - half : i + half + 1]) / window
    return result


def decompose(values, dates):
    trend = _moving_average(values, SEASON_LENGTH)

    # seasonal: average (value - trend) grouped by day-of-week, over points where trend exists
    by_dow = {d: [] for d in range(7)}
    for v, t, d_str in zip(values, trend, dates):
        if t is None:
            continue
        dow = date.fromisoformat(d_str).weekday()
        by_dow[dow].append(v - t)
    seasonal_by_dow = {dow: (sum(vals) / len(vals) if vals else 0.0) for dow, vals in by_dow.items()}

    residuals = []
    for v, t, d_str in zip(values, trend, dates):
        if t is None:
            continue
        dow = date.fromisoformat(d_str).weekday()
        residuals.append(v - t - seasonal_by_dow[dow])

    return trend, seasonal_by_dow, residuals


def _fit_line(xs, ys):
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    slope = num / den if den > 0 else 0.0
    intercept = mean_y - slope * mean_x
    return slope, intercept


def forecast(records, horizon=30):
    values = [r["value"] for r in records]
    dates = [r["date"] for r in records]

    trend, seasonal_by_dow, residuals = decompose(values, dates)

    trend_points = [(i, t) for i, t in enumerate(trend) if t is not None]
    slope, intercept = _fit_line([p[0] for p in trend_points], [p[1] for p in trend_points])

    residual_std = (sum(r * r for r in residuals) / len(residuals)) ** 0.5 if residuals else 0.0

    last_date = date.fromisoformat(dates[-1])
    last_index = len(values) - 1

    predictions = []
    for h in range(1, horizon + 1):
        idx = last_index + h
        future_date = last_date + timedelta(days=h)
        point_forecast = slope * idx + intercept + seasonal_by_dow[future_date.weekday()]
        interval_width = 1.96 * residual_std * math.sqrt(h)
        predictions.append(
            {
                "date": future_date.isoformat(),
                "forecast": round(point_forecast, 2),
                "lower": round(point_forecast - interval_width, 2),
                "upper": round(point_forecast + interval_width, 2),
            }
        )

    return {
        "slope_per_day": round(slope, 4),
        "residual_std": round(residual_std, 3),
        "seasonal_by_weekday": {
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][k]: round(v, 2) for k, v in seasonal_by_dow.items()
        },
        "predictions": predictions,
    }


def backtest(records, holdout=60):
    train = records[:-holdout]
    actual_holdout = records[-holdout:]

    result = forecast(train, horizon=holdout)
    preds = result["predictions"]

    errors = [abs(p["forecast"] - a["value"]) for p, a in zip(preds, actual_holdout)]
    sq_errors = [(p["forecast"] - a["value"]) ** 2 for p, a in zip(preds, actual_holdout)]
    mae = sum(errors) / len(errors)
    rmse = (sum(sq_errors) / len(sq_errors)) ** 0.5

    return {
        "holdout_days": holdout,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "actual": actual_holdout,
        "predicted": preds,
    }
