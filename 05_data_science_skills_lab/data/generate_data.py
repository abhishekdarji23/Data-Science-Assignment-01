"""
generate_data.py
-----------------
CRISP-DM Phase 2: Data Understanding (data acquisition step)

This build environment has no internet access to Kaggle, so this script
generates a SYNTHETIC dataset that mirrors the schema and statistical
shape of the classic "Titanic - Machine Learning from Disaster" Kaggle
dataset -- the most commonly used dataset for exactly this kind of
"demonstrate every data science skill" exercise, because it has missing
values, mixed categorical/numeric features, class imbalance, and a
genuine binary classification target.

Schema (matches the real Kaggle dataset's core columns):
    PassengerId, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Survived

Swap-in instructions for real data:
    Download `train.csv` from the Kaggle Titanic competition and drop it
    in this `data/` folder as `titanic_real.csv`. `src/data_prep.py` will
    use it automatically instead of the synthetic file if present.

Realism notes:
  * Survival is NOT random -- it depends on Sex, Pclass, and Age (women,
    higher class, and children survived at higher rates historically),
    with noise, so classification skills have real signal to find.
  * Age and Embarked have realistic missing-value rates (~20% and ~0.5%),
    matching the real dataset, so missing-data skills have something to do.
  * A handful of exact-duplicate rows are deliberately injected so the
    duplicate-detection skill has something to find.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_ROWS = 600
N_INJECTED_DUPLICATES = 8

OUT_PATH = Path(__file__).parent / "titanic_synthetic.csv"


def generate(n_rows: int = N_ROWS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pclass = rng.choice([1, 2, 3], size=n_rows, p=[0.24, 0.21, 0.55])
    sex = rng.choice(["male", "female"], size=n_rows, p=[0.65, 0.35])
    age = rng.normal(29, 13, n_rows).clip(0.5, 80).round(1)
    sibsp = rng.choice([0, 1, 2, 3, 4], size=n_rows, p=[0.68, 0.20, 0.06, 0.04, 0.02])
    parch = rng.choice([0, 1, 2, 3], size=n_rows, p=[0.76, 0.13, 0.08, 0.03])
    fare_base = {1: 84.0, 2: 20.0, 3: 13.5}
    fare = np.array([rng.exponential(fare_base[p]) for p in pclass]).round(2)
    embarked = rng.choice(["S", "C", "Q"], size=n_rows, p=[0.70, 0.19, 0.11])

    # Survival probability model: baseline + boosts, then Bernoulli draw.
    base = 0.28
    p_survive = np.full(n_rows, base)
    p_survive += np.where(sex == "female", 0.45, 0.0)
    p_survive += np.where(pclass == 1, 0.22, np.where(pclass == 2, 0.08, 0.0))
    p_survive += np.where(age < 12, 0.18, 0.0)
    p_survive += rng.normal(0, 0.08, n_rows)  # noise
    p_survive = p_survive.clip(0.03, 0.97)
    survived = rng.binomial(1, p_survive)

    df = pd.DataFrame({
        "PassengerId": np.arange(1, n_rows + 1),
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "Survived": survived,
    })

    # Inject realistic missingness (matches real Titanic dataset's rates)
    age_missing_idx = rng.choice(n_rows, size=int(n_rows * 0.20), replace=False)
    df.loc[age_missing_idx, "Age"] = np.nan
    embarked_missing_idx = rng.choice(n_rows, size=max(1, int(n_rows * 0.005)), replace=False)
    df.loc[embarked_missing_idx, "Embarked"] = np.nan

    # Inject exact duplicate rows for the duplicate-detection skill
    dup_rows = df.sample(n=N_INJECTED_DUPLICATES, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)
    df["PassengerId"] = np.arange(1, len(df) + 1)  # keep IDs unique/sequential

    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")
    print(f"Missing Age: {df['Age'].isna().sum()}, Missing Embarked: {df['Embarked'].isna().sum()}")
    print(f"Duplicate rows: {df.duplicated(subset=[c for c in df.columns if c != 'PassengerId']).sum()}")
    print(f"Survival rate: {df['Survived'].mean():.2%}")
