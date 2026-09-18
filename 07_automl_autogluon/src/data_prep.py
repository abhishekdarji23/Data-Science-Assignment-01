"""
data_prep.py
------------
CRISP-DM Phase 3: Data Preparation.

Shared feature-engineering code used by BOTH training (train.py) and
inference (predict.py / the API), so a live-scored customer is evaluated
on the exact same feature representation every base model was fit on.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REAL_CSV = DATA_DIR / "churn_real.csv"            # if the user drops in real Kaggle data
SYNTHETIC_CSV = DATA_DIR / "churn_synthetic.csv"

NUMERIC_COLUMNS = ["tenure_months", "monthly_charges", "total_charges", "num_support_calls"]
CATEGORICAL_COLUMNS = ["contract_type", "internet_service", "tech_support", "payment_method"]
TARGET_COLUMN = "churn"

CATEGORY_LEVELS = {
    "contract_type": ["Month-to-month", "One year", "Two year"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "tech_support": ["No", "Yes"],
    "payment_method": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
}


def load_dataset() -> pd.DataFrame:
    if REAL_CSV.exists():
        df = pd.read_csv(REAL_CSV)
        source = "real Kaggle churn_real.csv"
    elif SYNTHETIC_CSV.exists():
        df = pd.read_csv(SYNTHETIC_CSV)
        source = "synthetic (offline build environment)"
    else:
        raise FileNotFoundError(
            "No dataset found. Run `python data/generate_data.py` first, "
            "or place a Kaggle-derived CSV at data/churn_real.csv."
        )
    print(f"[data_prep] loaded {len(df):,} customers from {source}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encodes categoricals with FIXED, known category levels (not
    inferred from whatever happens to be in this particular dataframe) so a
    single live customer row encodes to the exact same columns the models
    were trained on -- a common AutoML/production bug is a category level
    missing from a single-row request silently producing a different (and
    wrong-shaped) set of dummy columns."""
    df = df.copy()
    for col, levels in CATEGORY_LEVELS.items():
        df[col] = pd.Categorical(df[col], categories=levels)
    cat_encoded = pd.get_dummies(df[CATEGORICAL_COLUMNS], drop_first=True)
    X = pd.concat([df[NUMERIC_COLUMNS].reset_index(drop=True),
                   cat_encoded.reset_index(drop=True)], axis=1)
    return X


def get_feature_names() -> list[str]:
    """The full, fixed feature column list after encoding, computed once
    from CATEGORY_LEVELS so train.py and predict.py always agree on it."""
    dummy = pd.DataFrame({col: [levels[0]] for col, levels in CATEGORY_LEVELS.items()})
    for col, levels in CATEGORY_LEVELS.items():
        dummy[col] = pd.Categorical(dummy[col], categories=levels)
    cat_encoded = pd.get_dummies(dummy[CATEGORICAL_COLUMNS], drop_first=True)
    return NUMERIC_COLUMNS + list(cat_encoded.columns)
