"""
predict.py
----------
CRISP-DM Phase 6: Deployment (inference layer).

Given a shopping cart (set of items already added), recommend items to
add next, using the association rules mined by train.py. A rule
`antecedents -> consequents` fires when `antecedents` is a subset of the
current cart; matching rules are ranked by lift (how much more likely the
consequent is, given the antecedent, vs. its own baseline rate) and
returned with their consequent items merged/de-duplicated.

This is a lookup over precomputed rules, not a live model -- consistent
with how market basket analysis is normally deployed (rules are mined
periodically offline, then served cheaply at request time).
"""
import json
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_rules = None
_metrics = None
_catalog = None


def _load():
    global _rules, _metrics, _catalog
    if _rules is None:
        with open(MODELS_DIR / "rules.json") as f:
            _rules = json.load(f)
    if _metrics is None:
        with open(MODELS_DIR / "metrics.json") as f:
            _metrics = json.load(f)
    if _catalog is None:
        with open(MODELS_DIR / "item_catalog.json") as f:
            _catalog = json.load(f)
    return _rules, _metrics, _catalog


def get_metrics() -> dict:
    _, metrics, _ = _load()
    return metrics


def get_rules(limit: int = 50) -> list:
    rules, _, _ = _load()
    return rules[:limit]


def get_catalog() -> list:
    _, _, catalog = _load()
    return catalog


def recommend(cart_items: list[str], top_n: int = 5) -> dict:
    rules, _, catalog = _load()
    cart_set = set(cart_items)

    unknown = [i for i in cart_items if i not in catalog]

    matched_rules = []
    recommendations = {}  # item -> best lift seen recommending it
    for r in rules:
        antecedents = set(r["antecedents"])
        if antecedents and antecedents.issubset(cart_set):
            new_items = [c for c in r["consequents"] if c not in cart_set]
            if not new_items:
                continue
            matched_rules.append(r)
            for item in new_items:
                if item not in recommendations or r["lift"] > recommendations[item]["lift"]:
                    recommendations[item] = {
                        "item": item,
                        "lift": r["lift"],
                        "confidence": r["confidence"],
                        "because_of": sorted(antecedents),
                    }

    ranked = sorted(recommendations.values(), key=lambda r: r["lift"], reverse=True)[:top_n]

    return {
        "cart": sorted(cart_set),
        "unknown_items": unknown,
        "matched_rule_count": len(matched_rules),
        "recommendations": ranked,
    }
