"""
data_prep.py
------------
CRISP-DM Phase 3: Data Preparation.

Shared code used by BOTH training (train.py) and inference (predict.py /
the API), so the model always sees features scaled the exact same way it
was fit on -- a scaler fit separately at serving time (even on the same
formula) would silently drift from the training scaler over time. Keeping
one `StandardScaler` object (persisted to disk) as the source of truth
avoids that.

CustomerID is intentionally excluded from the feature set: it is an
arbitrary identifier with no behavioral meaning, and including it would
let the model "cluster" on row order/ID rather than actual customer
behavior (a classic beginner mistake in clustering pipelines).
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REAL_CSV = DATA_DIR / "Mall_Customers.csv"                  # if the user drops in real Kaggle data
SYNTHETIC_CSV = DATA_DIR / "mall_customers_synthetic.csv"

# Deliberately excludes CustomerID (arbitrary id) and Gender (categorical,
# and using it risks building marketing segments around a protected
# attribute -- income/spending/age already separate the archetypes well).
FEATURE_COLUMNS = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def load_dataset() -> pd.DataFrame:
    """Loads real Kaggle data if the user dropped it in data/, else synthetic."""
    if REAL_CSV.exists():
        df = pd.read_csv(REAL_CSV)
        source = "real Kaggle Mall_Customers.csv"
    elif SYNTHETIC_CSV.exists():
        df = pd.read_csv(SYNTHETIC_CSV)
        source = "synthetic (offline build environment)"
    else:
        raise FileNotFoundError(
            "No dataset found. Run `python data/generate_data.py` first, "
            "or place a Kaggle Mall_Customers.csv at data/Mall_Customers.csv."
        )
    print(f"[data_prep] loaded {len(df):,} rows from {source}")

    # Basic sanity cleaning
    df = df.dropna(subset=FEATURE_COLUMNS)
    df = df[(df["Age"] > 0) & (df["Annual Income (k$)"] > 0)]
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Returns the raw (unscaled) feature matrix. Scaling is applied
    separately in train.py / predict.py using the persisted StandardScaler,
    so this function stays a pure, side-effect-free selector."""
    return df[FEATURE_COLUMNS].copy()


def build_single_row(age: float, annual_income_k: float, spending_score: float) -> pd.DataFrame:
    """Builds a one-row feature frame for a single live prediction request."""
    return pd.DataFrame([{
        "Age": age,
        "Annual Income (k$)": annual_income_k,
        "Spending Score (1-100)": spending_score,
    }])[FEATURE_COLUMNS]
