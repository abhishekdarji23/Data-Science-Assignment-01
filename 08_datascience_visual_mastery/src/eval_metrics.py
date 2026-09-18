"""
eval_metrics.py
---------------
Topic 2: Model evaluation — confusion matrix, type I/II errors, ROC-AUC,
cost matrix, precision/recall tradeoff.

Everything here operates on the fixed set of (true_label, predicted_probability)
pairs in data/eval_test_set_synthetic.csv. The live simulation is: the user
picks a decision threshold, and every quantity below (confusion matrix,
precision, recall, type I/II error counts) recomputes from the SAME
underlying probabilities -- this is the direct, hands-on way to see the
precision/recall tradeoff, rather than being told about it abstractly.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_test_set_synthetic.csv"

_df = None


def _load() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)
    return _df


def confusion_at_threshold(threshold: float, cost_fp: float = 1.0, cost_fn: float = 1.0) -> dict:
    """cost_fp / cost_fn let the demo illustrate a COST MATRIX: in many
    real problems a false negative (e.g. missing a fraud case) is far more
    expensive than a false positive (e.g. a false alarm), and vice versa --
    changing these weights changes what the 'best' threshold even means."""
    df = _load()
    y_true = df["true_label"].values
    y_pred = (df["predicted_probability"].values >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())  # Type I error
    fn = int(((y_pred == 0) & (y_true == 1)).sum())  # Type II error

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(df)
    total_cost = fp * cost_fp + fn * cost_fn

    return {
        "threshold": round(threshold, 3),
        "confusion_matrix": {"true_positive": tp, "true_negative": tn,
                              "false_positive_type1": fp, "false_negative_type2": fn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "total_cost": round(total_cost, 2),
    }


def roc_curve_points() -> dict:
    df = _load()
    fpr, tpr, thresholds = roc_curve(df["true_label"], df["predicted_probability"])
    auc = roc_auc_score(df["true_label"], df["predicted_probability"])
    # Downsample to at most ~40 points for a clean plot (ROC curves from
    # small test sets already have a manageable number of points, but cap
    # defensively).
    idx = np.linspace(0, len(fpr) - 1, min(40, len(fpr))).astype(int)
    return {
        "fpr": [round(float(x), 4) for x in fpr[idx]],
        "tpr": [round(float(x), 4) for x in tpr[idx]],
        "auc": round(float(auc), 4),
    }


def dataset_summary() -> dict:
    df = _load()
    return {
        "n_samples": len(df),
        "n_positive": int((df["true_label"] == 1).sum()),
        "n_negative": int((df["true_label"] == 0).sum()),
    }
