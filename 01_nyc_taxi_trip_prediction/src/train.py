"""
train.py
--------
CRISP-DM Phase 4 (Modeling) + Phase 5 (Evaluation).

Trains a gradient-boosted regressor to predict NYC taxi `trip_duration`
(seconds) from pickup/dropoff geometry, time-of-day, and ride metadata.

Leakage guardrails applied here (see AUDIT_REPORT.md for the full checklist):
  * Split is done BEFORE any fitting -- no scaler/encoder is fit on data
    that later ends up in the test set.
  * The split is time-based (train = first 80% of the year chronologically,
    test = last 20%), not a random shuffle, because trip data is a time
    series and random shuffling would leak future traffic patterns into
    the training set.
  * The target (`trip_duration`) is log1p-transformed for training (it is
    heavily right-skewed) and predictions are inverse-transformed with
    expm1 before computing metrics, so reported MAE/RMSE are in seconds.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_prep import load_dataset, add_engineered_features, FEATURE_COLUMNS, TARGET_COLUMN

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    df = df.sort_values("pickup_datetime").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def main():
    raw = load_dataset()
    raw = add_engineered_features(raw)

    train_df, test_df = time_based_split(raw)
    print(f"[train] train rows={len(train_df):,}  test rows={len(test_df):,}")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    y_train_log = np.log1p(y_train)

    model = GradientBoostingRegressor(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train_log)

    pred_log = model.predict(X_test)
    pred_sec = np.expm1(pred_log).clip(min=30)

    mae = mean_absolute_error(y_test, pred_sec)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_sec)))
    r2 = r2_score(y_test, pred_sec)
    # Common Kaggle-competition metric for this exact dataset (RMSLE)
    rmsle = float(np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(pred_sec))))

    metrics = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "mae_seconds": round(mae, 1),
        "rmse_seconds": round(rmse, 1),
        "rmsle": round(rmsle, 4),
        "r2": round(float(r2), 4),
        "features": FEATURE_COLUMNS,
        "model": "GradientBoostingRegressor",
        "target_transform": "log1p",
        "split": "time_based_80_20",
        "random_seed": RANDOM_SEED,
    }
    print("[train] test metrics:", json.dumps(metrics, indent=2))

    feature_importance = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_.tolist()),
        key=lambda t: t[1], reverse=True,
    )
    metrics["feature_importance"] = [{"feature": f, "importance": round(i, 4)} for f, i in feature_importance]

    joblib.dump(model, MODELS_DIR / "trip_duration_model.joblib")
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train] saved model -> {MODELS_DIR / 'trip_duration_model.joblib'}")
    print(f"[train] saved metrics -> {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
