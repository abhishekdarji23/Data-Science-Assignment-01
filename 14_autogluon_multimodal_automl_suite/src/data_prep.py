"""
data_prep.py
------------
CRISP-DM Phase 3: Data Preparation.

Builds the tabular and text feature representations used by all 3 models
(tabular-only, text-only, and the fusion model) -- shared here so every
model sees features built the exact same way.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REAL_CSV = DATA_DIR / "listings_real.csv"          # if the user drops in real data
SYNTHETIC_CSV = DATA_DIR / "listings_synthetic.csv"

NUMERIC_COLUMNS = ["bedrooms", "bathrooms", "accommodates"]
CATEGORICAL_COLUMNS = ["room_type", "neighborhood"]
TEXT_COLUMN = "description"
TARGET_COLUMN = "price"

CATEGORY_LEVELS = {
    "room_type": ["Entire home/apt", "Private room", "Shared room"],
    "neighborhood": ["Downtown", "Uptown", "Waterfront", "Suburb"],
}


def load_dataset() -> pd.DataFrame:
    if REAL_CSV.exists():
        df = pd.read_csv(REAL_CSV)
        source = "real listings_real.csv"
    elif SYNTHETIC_CSV.exists():
        df = pd.read_csv(SYNTHETIC_CSV)
        source = "synthetic (offline build environment)"
    else:
        raise FileNotFoundError(
            "No dataset found. Run `python data/generate_data.py` first, "
            "or place a real CSV at data/listings_real.csv."
        )
    print(f"[data_prep] loaded {len(df):,} listings from {source}")
    return df


def build_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encodes categoricals with FIXED, known category levels, so
    a single live listing encodes to the exact same columns the models
    were trained on (same reasoning as project 07's data_prep.py)."""
    df = df.copy()
    for col, levels in CATEGORY_LEVELS.items():
        df[col] = pd.Categorical(df[col], categories=levels)
    cat_encoded = pd.get_dummies(df[CATEGORICAL_COLUMNS], drop_first=True)
    X = pd.concat([df[NUMERIC_COLUMNS].reset_index(drop=True),
                   cat_encoded.reset_index(drop=True)], axis=1)
    return X


def get_tabular_feature_names() -> list[str]:
    dummy = pd.DataFrame({col: [levels[0]] for col, levels in CATEGORY_LEVELS.items()})
    for col, levels in CATEGORY_LEVELS.items():
        dummy[col] = pd.Categorical(dummy[col], categories=levels)
    cat_encoded = pd.get_dummies(dummy[CATEGORICAL_COLUMNS], drop_first=True)
    return NUMERIC_COLUMNS + list(cat_encoded.columns)
