"""
data_prep.py
------------
CRISP-DM Phase 3: Data Preparation.

Shared feature list used by BOTH training (train.py) and inference
(predict.py / the API), so a live-scored transaction is evaluated on the
exact same features the detectors were fit on.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REAL_CSV = DATA_DIR / "transactions_real.csv"          # if the user drops in real data
SYNTHETIC_CSV = DATA_DIR / "transactions_synthetic.csv"

FEATURE_COLUMNS = [
    "amount_usd",
    "hour_of_day",
    "account_age_days",
    "transactions_last_hour",
    "distance_from_home_km",
]
LABEL_COLUMN = "is_anomaly"  # ground truth, used ONLY for evaluation, never as a model input


def load_dataset() -> pd.DataFrame:
    if REAL_CSV.exists():
        df = pd.read_csv(REAL_CSV)
        source = "real transactions_real.csv"
    elif SYNTHETIC_CSV.exists():
        df = pd.read_csv(SYNTHETIC_CSV)
        source = "synthetic (offline build environment)"
    else:
        raise FileNotFoundError(
            "No dataset found. Run `python data/generate_data.py` first, "
            "or place real data at data/transactions_real.csv "
            f"(columns: {', '.join(FEATURE_COLUMNS)}, {LABEL_COLUMN})."
        )
    print(f"[data_prep] loaded {len(df):,} transactions from {source} "
          f"({df[LABEL_COLUMN].sum() if LABEL_COLUMN in df.columns else '?'} labeled anomalies)")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].copy()
