"""
data_prep.py
------------
CRISP-DM Phase 3: Data Preparation.

Loads the price series and builds lag/rolling features for the
gradient-boosting forecaster. Every feature at row t is built ONLY from
prices at t-1 and earlier -- this is what makes each test-set row's
prediction a genuine 1-step-ahead forecast using real historical data,
not a peek at the future (see AUDIT_REPORT.md for the direct check).
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REAL_CSV = DATA_DIR / "spy_real.csv"          # if the user drops in real data
SYNTHETIC_CSV = DATA_DIR / "spy_synthetic.csv"

LAGS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [5, 10]


def load_dataset() -> pd.DataFrame:
    if REAL_CSV.exists():
        df = pd.read_csv(REAL_CSV, parse_dates=["date"])
        source = "real spy_real.csv"
    elif SYNTHETIC_CSV.exists():
        df = pd.read_csv(SYNTHETIC_CSV, parse_dates=["date"])
        source = "synthetic (offline build environment)"
    else:
        raise FileNotFoundError(
            "No dataset found. Run `python data/generate_data.py` first, "
            "or place real data at data/spy_real.csv (columns: date, close)."
        )
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[data_prep] loaded {len(df):,} trading days from {source} "
          f"({df['date'].min().date()} to {df['date'].max().date()})")
    return df


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds lag/rolling features for the ML forecaster. Every feature
    for row t uses only close[t-1] and earlier -- never close[t] itself."""
    out = df.copy()
    for lag in LAGS:
        out[f"lag_{lag}"] = out["close"].shift(lag)
    for window in ROLLING_WINDOWS:
        # shift(1) first so the rolling window itself only covers t-1 and earlier
        out[f"roll_mean_{window}"] = out["close"].shift(1).rolling(window).mean()
        out[f"roll_std_{window}"] = out["close"].shift(1).rolling(window).std()
    out["day_of_week"] = out["date"].dt.dayofweek
    out["prev_return"] = out["close"].shift(1).pct_change()
    return out


def get_feature_columns() -> list[str]:
    cols = [f"lag_{lag}" for lag in LAGS]
    cols += [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
    cols += [f"roll_std_{w}" for w in ROLLING_WINDOWS]
    cols += ["day_of_week", "prev_return"]
    return cols
