"""
generate_data.py
-----------------
CRISP-DM Phase 2: Data Understanding (data acquisition step)

This build environment has no internet access to Kaggle, so this script
generates a SYNTHETIC grocery basket dataset that mirrors the shape of
the classic "Groceries" / "Online Retail" market-basket-analysis datasets
used for association rule mining tutorials.

Schema:
    transaction_id, item
(one row per item-in-a-basket; a transaction_id repeats for every item in
that basket -- the standard "long format" for market basket data)

Swap-in instructions for real data:
    Download the Kaggle "Groceries dataset" (or similar) and reshape it to
    the same two-column (transaction_id, item) format, save as
    data/transactions_real.csv. `src/data_prep.py` will use it automatically
    instead of the synthetic file if present.

Realism note: items are NOT chosen independently at random. A small set of
"association rules" (e.g. bread -> butter, chips -> salsa, diapers -> beer)
is baked into the generator, each with its own baseline co-purchase
probability, so the downstream Apriori/association-rule mining has genuine
non-trivial patterns to discover -- a purely random basket generator would
make the whole exercise trivial (every itemset would have near-baseline
support and no meaningful lift).
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_TRANSACTIONS = 3000

OUT_PATH = Path(__file__).parent / "transactions_synthetic.csv"

CATALOG = [
    "bread", "butter", "milk", "eggs", "cheese", "yogurt",
    "chips", "salsa", "soda", "beer", "diapers", "wipes",
    "coffee", "sugar", "pasta", "tomato_sauce", "parmesan",
    "cereal", "bananas", "apples", "chicken", "rice",
]

# Baked-in association rules: (antecedent_items, consequent_items, lift_strength)
# lift_strength in [0,1]: probability the consequent is ALSO added, given the
# antecedent is present in the basket (on top of its own baseline rate).
BAKED_IN_RULES = [
    (["bread"], ["butter"], 0.65),
    (["pasta"], ["tomato_sauce"], 0.70),
    (["tomato_sauce"], ["parmesan"], 0.45),
    (["chips"], ["salsa"], 0.55),
    (["diapers"], ["beer"], 0.40),          # the classic textbook example
    (["diapers"], ["wipes"], 0.75),
    (["coffee"], ["sugar"], 0.50),
    (["cereal"], ["milk"], 0.60),
    (["chicken"], ["rice"], 0.35),
    (["bananas"], ["apples"], 0.25),
]

# Baseline probability each item appears in a random basket, independent of rules
BASELINE_P = {item: p for item, p in zip(
    CATALOG,
    [0.28, 0.14, 0.30, 0.22, 0.16, 0.18, 0.20, 0.10, 0.18, 0.12, 0.09, 0.08,
     0.20, 0.10, 0.15, 0.12, 0.08, 0.16, 0.25, 0.24, 0.18, 0.14],
)}


def generate(n_transactions: int = N_TRANSACTIONS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for tid in range(1, n_transactions + 1):
        basket = set()
        # Step 1: baseline independent draws
        for item, p in BASELINE_P.items():
            if rng.random() < p:
                basket.add(item)

        # Step 2: apply baked-in association boosts (antecedent present -> boost consequent)
        for antecedents, consequents, strength in BAKED_IN_RULES:
            if all(a in basket for a in antecedents):
                for c in consequents:
                    if c not in basket and rng.random() < strength:
                        basket.add(c)

        # Ensure baskets aren't empty (retail transactions always have >=1 item)
        if not basket:
            basket.add(rng.choice(CATALOG))

        for item in basket:
            rows.append({"transaction_id": tid, "item": item})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    n_tx = df["transaction_id"].nunique()
    avg_basket = len(df) / n_tx
    print(f"Wrote {len(df):,} item-rows across {n_tx:,} transactions to {OUT_PATH}")
    print(f"Average basket size: {avg_basket:.2f} items")
    print(df["item"].value_counts().head(10))
