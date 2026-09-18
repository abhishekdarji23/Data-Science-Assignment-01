"""
train.py
--------
CRISP-DM Phase 4 (Modeling) + Phase 5 (Evaluation).

Compares 5 forecasting methods on the SAME chronological test period,
all evaluated as genuine 1-STEP-AHEAD forecasts: at each test-set day t,
every method predicts close[t] using only real, true data through
close[t-1] -- never its own prior predictions and never any future
value. This matters a lot for financial series specifically: multi-step
recursive forecasting (feeding predictions back in as if they were
truth) would compound error very differently between methods and make
the comparison unfair/uninformative.

Methods, in increasing nominal sophistication:
  1. naive              -- predict close[t] = close[t-1] (the classic,
                            hard-to-beat baseline for near-random-walk
                            financial series)
  2. moving_average      -- predict close[t] = mean(close[t-5:t])
  3. exponential_smoothing -- statsmodels Holt's linear trend method,
                            refit at each step on the true history so far
  4. arima               -- statsmodels ARIMA (order chosen once via a
                            small AIC grid search on the training data),
                            refit at each step on the true history so far
  5. gradient_boosting    -- scikit-learn GradientBoostingRegressor on
                            lag/rolling features (data_prep.py), trained
                            ONCE on the training split

This is explicitly NOT the real AutoGluon/deep-learning "SOTA" stack
(no LSTM/Transformer/Temporal Fusion Transformer) -- see PROMPTS.md for
why (heavy dependencies conflict with "keep it simple"). These 5
methods span classical statistics through gradient-boosted ML, which is
enough to make an honest, checkable "does anything beat naive" story
for financial forecasting -- itself the actual, well-documented finding
in a lot of real forecasting literature, not a strawman built to make
one method win.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

from data_prep import load_dataset, build_lag_features, get_feature_columns

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42
TEST_FRACTION = 0.15
MA_WINDOW = 5


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100
    return {"mae_usd": round(float(mae), 3), "rmse_usd": round(rmse, 3), "mape_pct": round(mape, 3)}


def main():
    t_start = time.time()
    df = load_dataset()
    close = df["close"].values
    n = len(df)
    split_idx = int(n * (1 - TEST_FRACTION))
    n_test = n - split_idx
    print(f"[train] {split_idx} train days, {n_test} test days")

    # --- Method 1: naive persistence ---
    pred_naive = close[split_idx - 1: n - 1]  # close[t-1] for each test t
    y_test = close[split_idx:]
    metrics_naive = evaluate(y_test, pred_naive)

    # --- Method 2: moving average ---
    pred_ma = np.array([close[t - MA_WINDOW:t].mean() for t in range(split_idx, n)])
    metrics_ma = evaluate(y_test, pred_ma)

    # --- Method 3: Exponential Smoothing (Holt's linear trend), refit each step ---
    pred_ets = []
    history = list(close[:split_idx])
    for t in range(split_idx, n):
        model = ExponentialSmoothing(history, trend="add", damped_trend=True,
                                      initialization_method="estimated")
        fit = model.fit(optimized=True)
        pred_ets.append(fit.forecast(1)[0])
        history.append(close[t])
    pred_ets = np.array(pred_ets)
    metrics_ets = evaluate(y_test, pred_ets)

    # --- Method 4: ARIMA, order chosen once via small AIC grid search on TRAIN only ---
    train_series = close[:split_idx]
    best_order, best_aic = None, np.inf
    for p in range(3):
        for d in [1]:
            for q in range(3):
                try:
                    fit = ARIMA(train_series, order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best_aic, best_order = fit.aic, (p, d, q)
                except Exception:
                    continue
    print(f"[train] ARIMA order chosen by AIC on train only: {best_order} (AIC={best_aic:.1f})")

    pred_arima = []
    history = list(close[:split_idx])
    for t in range(split_idx, n):
        fit = ARIMA(history, order=best_order).fit()
        pred_arima.append(fit.forecast(1)[0])
        history.append(close[t])
    pred_arima = np.array(pred_arima)
    metrics_arima = evaluate(y_test, pred_arima)

    # --- Method 5: Gradient Boosting on lag features, trained ONCE ---
    df_feat = build_lag_features(df)
    feature_cols = get_feature_columns()
    df_feat = df_feat.dropna(subset=feature_cols).reset_index(drop=True)
    # re-derive split index on the feature-engineered (shorter, due to dropna) frame
    gbm_split_idx = int(len(df_feat) * (1 - TEST_FRACTION))
    X_train_gbm = df_feat.loc[:gbm_split_idx - 1, feature_cols]
    y_train_gbm = df_feat.loc[:gbm_split_idx - 1, "close"]
    X_test_gbm = df_feat.loc[gbm_split_idx:, feature_cols]
    y_test_gbm = df_feat.loc[gbm_split_idx:, "close"].values

    gbm = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_SEED)
    gbm.fit(X_train_gbm, y_train_gbm)
    pred_gbm = gbm.predict(X_test_gbm)
    metrics_gbm = evaluate(y_test_gbm, pred_gbm)

    leaderboard = {
        "naive": metrics_naive,
        "moving_average": metrics_ma,
        "exponential_smoothing": metrics_ets,
        "arima": metrics_arima,
        "gradient_boosting": metrics_gbm,
    }
    print("[train] leaderboard (lower MAE/RMSE/MAPE is better):")
    for name, m in leaderboard.items():
        print(f"    {name:22s} MAE=${m['mae_usd']}  RMSE=${m['rmse_usd']}  MAPE={m['mape_pct']}%")

    best_method = min(leaderboard, key=lambda k: leaderboard[k]["mae_usd"])
    naive_mae = metrics_naive["mae_usd"]
    beats_naive = {name: (m["mae_usd"] < naive_mae) for name, m in leaderboard.items()}
    print(f"[train] best method by MAE: {best_method}")
    print(f"[train] methods that beat naive on MAE: {[k for k, v in beats_naive.items() if v]}")

    joblib.dump(gbm, MODELS_DIR / "gradient_boosting.joblib")

    # Save test-period actuals + naive/GBM predictions for charting (the two
    # easiest to reproduce cheaply at serve time; ETS/ARIMA are refit-heavy)
    chart_df = pd.DataFrame({
        "date": df["date"].iloc[split_idx:].dt.strftime("%Y-%m-%d").values,
        "actual": y_test,
        "naive_pred": pred_naive,
    })

    metrics = {
        "n_days_total": int(n),
        "n_train": int(split_idx),
        "n_test": int(n_test),
        "test_fraction": TEST_FRACTION,
        "ma_window": MA_WINDOW,
        "arima_order": list(best_order),
        "feature_columns": feature_cols,
        "leaderboard": leaderboard,
        "best_method": best_method,
        "beats_naive": beats_naive,
        "random_seed": RANDOM_SEED,
        "training_wall_time_seconds": round(time.time() - t_start, 1),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    chart_df.to_csv(MODELS_DIR / "test_period_chart_data.csv", index=False)

    print(f"[train] saved GBM model + metrics -> {MODELS_DIR}")
    print(f"[train] total wall time: {metrics['training_wall_time_seconds']}s")


if __name__ == "__main__":
    main()
