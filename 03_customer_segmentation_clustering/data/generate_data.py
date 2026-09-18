"""
generate_data.py
-----------------
CRISP-DM Phase 2: Data Understanding (data acquisition step)

This build environment has no internet access to Kaggle, so this script
generates a SYNTHETIC dataset that mirrors the schema and statistical
shape of the classic "Mall Customer Segmentation" Kaggle dataset
(https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python),
the most commonly used beginner-to-intermediate clustering dataset.

Schema (matches the real Kaggle dataset exactly):
    CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)

Swap-in instructions for real data:
    Download `Mall_Customers.csv` from the Kaggle dataset above and drop
    it in this `data/` folder with that exact name. `src/data_prep.py`
    will use it automatically instead of the synthetic file if present.

Unlike project 01 (regression, where the target had to be derivable from
a physical model), clustering is unsupervised -- there is no "true" label
to predict, so this generator instead constructs the data from 4 latent
customer archetypes with distinct (income, spending) centers plus noise,
so that genuine, separable cluster structure exists for the model to find
(a pure-random dataset would make every clustering result meaningless).
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_ROWS = 500  # matches the real Mall_Customers.csv row count order of magnitude

OUT_PATH = Path(__file__).parent / "mall_customers_synthetic.csv"

# 4 latent customer archetypes: (mean_income_k, mean_spending, weight)
ARCHETYPES = [
    {"name": "budget_conscious",     "income": 25, "spending": 20, "age_mean": 45, "weight": 0.25},
    {"name": "high_income_low_spend", "income": 85, "spending": 18, "age_mean": 48, "weight": 0.20},
    {"name": "high_income_high_spend", "income": 88, "spending": 82, "age_mean": 32, "weight": 0.20},
    {"name": "moderate_balanced",     "income": 55, "spending": 52, "age_mean": 38, "weight": 0.35},
]


def generate(n_rows: int = N_ROWS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weights = [a["weight"] for a in ARCHETYPES]
    archetype_idx = rng.choice(len(ARCHETYPES), size=n_rows, p=weights)

    incomes, spendings, ages = [], [], []
    for idx in archetype_idx:
        a = ARCHETYPES[idx]
        incomes.append(rng.normal(a["income"], 9))
        spendings.append(rng.normal(a["spending"], 10))
        ages.append(rng.normal(a["age_mean"], 8))

    income = np.clip(np.round(incomes), 15, 140)
    spending = np.clip(np.round(spendings), 1, 100)
    age = np.clip(np.round(ages), 18, 75).astype(int)
    gender = rng.choice(["Male", "Female"], size=n_rows, p=[0.44, 0.56])

    df = pd.DataFrame({
        "CustomerID": np.arange(1, n_rows + 1),
        "Gender": gender,
        "Age": age,
        "Annual Income (k$)": income.astype(int),
        "Spending Score (1-100)": spending.astype(int),
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} synthetic customers to {OUT_PATH}")
    print(df.head())
    print(df.describe().T[["mean", "min", "max"]])
