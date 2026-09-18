"""
generate_data.py
-----------------
CRISP-DM Phase 2: Data Understanding (data acquisition step)

This build environment has no internet access to Kaggle, so this script
generates a SYNTHETIC transaction dataset in the spirit of the classic
"Credit Card Fraud Detection" Kaggle dataset -- highly imbalanced binary
anomaly detection, the most common shape for this kind of demo.

Unlike the real Kaggle dataset (which anonymizes features as PCA
components V1-V28, making them uninterpretable), this generator uses
plain, interpretable transaction features so the resulting dashboard
means something to a human reviewer:

    amount_usd, hour_of_day, account_age_days,
    transactions_last_hour, distance_from_home_km, is_anomaly

Swap-in instructions for real data:
    Download the Kaggle "Credit Card Fraud Detection" dataset (or any
    transaction-level fraud dataset) and reshape it to a compatible
    schema (numeric feature columns + a binary label column named
    `is_anomaly`), save as data/transactions_real.csv.

Realism note: anomalies are NOT random outliers -- they follow a
different, genuinely separable multivariate distribution (higher amount,
odd hours, rapid repeated transactions, far from home, and newer
accounts), which is what real fraud patterns look like and what gives
Isolation Forest / LOF / z-score methods a real signal to find.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_ROWS = 5000
ANOMALY_RATE = 0.03  # 3% of transactions are anomalous, matches real-world fraud rates order of magnitude

OUT_PATH = Path(__file__).parent / "transactions_synthetic.csv"


def generate(n_rows: int = N_ROWS, anomaly_rate: float = ANOMALY_RATE, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_anomalies = int(n_rows * anomaly_rate)
    n_normal = n_rows - n_anomalies

    # --- Normal transactions ---
    normal = pd.DataFrame({
        "amount_usd": rng.lognormal(mean=3.4, sigma=0.6, size=n_normal).round(2),
        "hour_of_day": rng.choice(range(24), size=n_normal,
                                   p=_daytime_weighted_hours()),
        "account_age_days": rng.uniform(30, 2000, n_normal).round(0),
        "transactions_last_hour": rng.poisson(1.0, n_normal),
        "distance_from_home_km": rng.exponential(5.0, n_normal).round(2),
        "is_anomaly": 0,
    })

    # --- Anomalous transactions: different multivariate distribution ---
    anomaly = pd.DataFrame({
        "amount_usd": rng.lognormal(mean=6.0, sigma=0.8, size=n_anomalies).round(2),
        "hour_of_day": rng.choice(range(24), size=n_anomalies, p=_nighttime_weighted_hours()),
        "account_age_days": rng.uniform(1, 60, n_anomalies).round(0),
        "transactions_last_hour": rng.poisson(6.0, n_anomalies),
        "distance_from_home_km": rng.exponential(80.0, n_anomalies).round(2),
        "is_anomaly": 1,
    })

    df = pd.concat([normal, anomaly], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    df.insert(0, "transaction_id", [f"tx{100000+i}" for i in range(len(df))])
    return df


def _daytime_weighted_hours():
    # Most legitimate transactions happen 8am-10pm
    w = np.ones(24)
    w[8:22] = 4.0
    return w / w.sum()


def _nighttime_weighted_hours():
    # Fraud disproportionately happens overnight, 12am-5am
    w = np.ones(24)
    w[0:5] = 5.0
    return w / w.sum()


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} transactions to {OUT_PATH}")
    print(f"Anomaly rate: {df['is_anomaly'].mean():.2%} ({df['is_anomaly'].sum()} anomalies)")
    print(df.groupby("is_anomaly")[["amount_usd", "transactions_last_hour", "distance_from_home_km"]].mean().round(2))
