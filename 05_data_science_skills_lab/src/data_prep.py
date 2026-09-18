"""
data_prep.py
------------
Loads the working dataset for every skill. Deliberately does NOT clean or
transform the data here -- several skills (missing value analysis, outlier
detection, duplicate detection) need to see the RAW, uncleaned data to have
anything to demonstrate. Cleaning/transformation is itself one of the
skills (see skills.py), not a preprocessing step hidden before the lab.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REAL_CSV = DATA_DIR / "titanic_real.csv"           # if the user drops in real Kaggle data
SYNTHETIC_CSV = DATA_DIR / "titanic_synthetic.csv"

TARGET_COLUMN = "Survived"
NUMERIC_COLUMNS = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_COLUMNS = ["Pclass", "Sex", "Embarked"]


def load_dataset() -> pd.DataFrame:
    if REAL_CSV.exists():
        df = pd.read_csv(REAL_CSV)
        source = "real Kaggle titanic_real.csv"
    elif SYNTHETIC_CSV.exists():
        df = pd.read_csv(SYNTHETIC_CSV)
        source = "synthetic (offline build environment)"
    else:
        raise FileNotFoundError(
            "No dataset found. Run `python data/generate_data.py` first, "
            "or place a Kaggle train.csv at data/titanic_real.csv."
        )
    print(f"[data_prep] loaded {len(df):,} rows from {source}")
    return df
