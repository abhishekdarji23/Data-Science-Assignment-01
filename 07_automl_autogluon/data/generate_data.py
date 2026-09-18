"""
generate_data.py
-----------------
CRISP-DM Phase 2: Data Understanding (data acquisition step)

This build environment has no internet access to Kaggle, so this script
generates a SYNTHETIC dataset that mirrors the schema and statistical
shape of the classic "Telco Customer Churn" Kaggle dataset -- a common
benchmark for AutoML tools (including real AutoGluon tutorials), because
it mixes numeric and categorical features with a genuine, non-trivial
binary target.

Schema:
    customer_id, tenure_months, monthly_charges, total_charges,
    contract_type, internet_service, tech_support, payment_method,
    num_support_calls, churn

Swap-in instructions for real data:
    Download the Kaggle "Telco Customer Churn" dataset and reshape/rename
    columns to match this schema, save as data/churn_real.csv.

Realism note: churn is NOT random. It depends on contract type
(month-to-month churns far more than 2-year contracts), tenure (newer
customers churn more), monthly charges (pricier plans churn more), and
support call volume (frustrated customers churn more) -- with noise --
so the AutoML pipeline has genuine, learnable signal, and so a "which
feature matters" story is checkable against what was built in.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_ROWS = 2000

OUT_PATH = Path(__file__).parent / "churn_synthetic.csv"

CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
CONTRACT_WEIGHTS = [0.55, 0.25, 0.20]
CONTRACT_CHURN_BOOST = {"Month-to-month": 0.45, "One year": 0.10, "Two year": 0.01}

INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]
INTERNET_WEIGHTS = [0.35, 0.45, 0.20]

PAYMENT_METHODS = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
PAYMENT_WEIGHTS = [0.35, 0.20, 0.23, 0.22]


def generate(n_rows: int = N_ROWS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(0, 73, n_rows)
    contract_type = rng.choice(CONTRACT_TYPES, size=n_rows, p=CONTRACT_WEIGHTS)
    internet_service = rng.choice(INTERNET_SERVICES, size=n_rows, p=INTERNET_WEIGHTS)
    tech_support = rng.choice(["Yes", "No"], size=n_rows, p=[0.40, 0.60])
    payment_method = rng.choice(PAYMENT_METHODS, size=n_rows, p=PAYMENT_WEIGHTS)
    num_support_calls = rng.poisson(1.2, n_rows)

    base_charge = np.where(internet_service == "Fiber optic", 75,
                   np.where(internet_service == "DSL", 50, 25))
    monthly_charges = (base_charge + rng.normal(0, 12, n_rows)).clip(18, 120).round(2)
    total_charges = (monthly_charges * tenure_months + rng.normal(0, 50, n_rows)).clip(0).round(2)

    # Churn probability model: baseline + boosts from known drivers + noise
    p_churn = np.full(n_rows, 0.03)
    p_churn += np.array([CONTRACT_CHURN_BOOST[c] for c in contract_type])
    p_churn += np.where(tenure_months < 6, 0.35, np.where(tenure_months < 12, 0.15, 0.0))
    p_churn += np.where(monthly_charges > 85, 0.25, 0.0)
    p_churn += np.clip(num_support_calls - 2, 0, None) * 0.10
    p_churn += np.where(tech_support == "No", 0.08, 0.0)
    p_churn += rng.normal(0, 0.03, n_rows)
    p_churn = p_churn.clip(0.01, 0.97)
    churn = rng.binomial(1, p_churn)

    df = pd.DataFrame({
        "customer_id": [f"cust{10000+i}" for i in range(n_rows)],
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "payment_method": payment_method,
        "num_support_calls": num_support_calls,
        "churn": churn,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} customers to {OUT_PATH}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    print(df.groupby("contract_type")["churn"].mean().round(3))
