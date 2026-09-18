"""
train.py
--------
CRISP-DM Phase 4 (Modeling) + Phase 5 (Evaluation).

Runs the Apriori algorithm to find frequent itemsets, then derives
association rules (antecedent -> consequent) with support, confidence,
and lift -- the standard market-basket-analysis pipeline.

Definitions (for the admin dashboard / anyone reading the metrics):
  * support(X)    = P(X in basket)                        -- how common is X overall
  * confidence(X->Y) = P(Y in basket | X in basket)        -- how often Y follows X
  * lift(X->Y)    = confidence(X->Y) / support(Y)          -- how much MORE likely Y is
                     given X, vs. Y's baseline rate. lift=1 means no association;
                     lift>1 means a genuine positive association (this is the key
                     metric -- confidence alone can be misleading for very popular
                     items that show up in most rules regardless of the antecedent).

Evaluation note: unlike supervised learning, there's no train/test split here
-- association rule mining describes patterns present in the transaction log
itself (like clustering). "Evaluation" means sanity-checking that mined rules
are non-trivial (lift > 1, reasonable support) rather than measuring
held-out predictive accuracy.
"""
import json
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from data_prep import load_transactions, build_basket_matrix

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MIN_SUPPORT = 0.03      # itemset must appear in >=3% of transactions
MIN_CONFIDENCE = 0.30    # rule must hold >=30% of the time when antecedent present
MIN_LIFT = 1.1            # rule must show a genuine positive association


def main():
    df = load_transactions()
    basket = build_basket_matrix(df)
    n_transactions = len(basket)

    frequent_itemsets = apriori(basket, min_support=MIN_SUPPORT, use_colnames=True)
    print(f"[train] {len(frequent_itemsets)} frequent itemsets found "
          f"(min_support={MIN_SUPPORT})")

    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE
    )
    rules = rules[rules["lift"] >= MIN_LIFT].copy()
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    # frozenset -> sorted list of strings, for clean JSON output
    rules["antecedents"] = rules["antecedents"].apply(lambda s: sorted(s))
    rules["consequents"] = rules["consequents"].apply(lambda s: sorted(s))

    rules_out = rules[[
        "antecedents", "consequents", "support", "confidence", "lift"
    ]].round(4).to_dict(orient="records")

    print(f"[train] {len(rules_out)} rules kept "
          f"(min_confidence={MIN_CONFIDENCE}, min_lift={MIN_LIFT})")

    item_frequency = (
        df["item"].value_counts() / n_transactions
    ).round(4).sort_values(ascending=False)
    item_freq_out = [{"item": k, "support": v} for k, v in item_frequency.items()]

    metrics = {
        "n_transactions": int(n_transactions),
        "n_unique_items": int(df["item"].nunique()),
        "avg_basket_size": round(len(df) / n_transactions, 2),
        "min_support": MIN_SUPPORT,
        "min_confidence": MIN_CONFIDENCE,
        "min_lift": MIN_LIFT,
        "n_frequent_itemsets": int(len(frequent_itemsets)),
        "n_rules": len(rules_out),
        "top_items_by_support": item_freq_out[:10],
        "algorithm": "Apriori (mlxtend)",
    }

    with open(MODELS_DIR / "rules.json", "w") as f:
        json.dump(rules_out, f, indent=2)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(MODELS_DIR / "item_catalog.json", "w") as f:
        json.dump(sorted(df["item"].unique().tolist()), f, indent=2)

    print(f"[train] saved rules -> {MODELS_DIR / 'rules.json'}")
    print(f"[train] saved metrics -> {MODELS_DIR / 'metrics.json'}")
    print("[train] top 5 rules by lift:")
    for r in rules_out[:5]:
        print(f"    {r['antecedents']} -> {r['consequents']}  "
              f"(support={r['support']}, confidence={r['confidence']}, lift={r['lift']})")


if __name__ == "__main__":
    main()
