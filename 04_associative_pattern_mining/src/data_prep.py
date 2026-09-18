"""
data_prep.py
------------
CRISP-DM Phase 3: Data Preparation.

Converts long-format transaction data (transaction_id, item) into the
one-hot encoded "basket matrix" (rows = transactions, columns = items,
values = item present/absent) that mlxtend's apriori() requires.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REAL_CSV = DATA_DIR / "transactions_real.csv"          # if the user drops in real data
SYNTHETIC_CSV = DATA_DIR / "transactions_synthetic.csv"


def load_transactions() -> pd.DataFrame:
    """Loads real transaction data if present, else synthetic. Long format:
    one row per (transaction_id, item)."""
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
            "(columns: transaction_id, item)."
        )
    print(f"[data_prep] loaded {df['transaction_id'].nunique():,} transactions "
          f"({len(df):,} item-rows) from {source}")
    return df


def build_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Long format -> one-hot basket matrix (rows=transactions, cols=items, bool)."""
    basket = (
        df.assign(present=1)
        .pivot_table(index="transaction_id", columns="item", values="present", fill_value=0)
        .astype(bool)
    )
    return basket
