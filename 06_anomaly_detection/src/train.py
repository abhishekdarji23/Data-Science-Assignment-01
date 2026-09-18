"""
train.py
--------
CRISP-DM Phase 4 (Modeling) + Phase 5 (Evaluation).

Trains and compares 3 standard, popular unsupervised anomaly detection
methods on the same feature set:

  1. Isolation Forest   -- isolates points via random recursive splits;
                            anomalies need fewer splits to isolate.
  2. Local Outlier Factor (LOF) -- density-based; flags points in
                            sparser neighborhoods than their neighbors.
  3. Z-score (statistical baseline) -- flags points far (in standard
                            deviations) from the feature-wise mean.
                            Included as a simple, transparent baseline
                            the two ML methods should beat.

All three are UNSUPERVISED (they never see `is_anomaly` while fitting) --
this matches how anomaly detection is used in practice, where you rarely
have a large set of confirmed fraud/anomaly labels to train on. The
`is_anomaly` label is used ONLY afterward, to evaluate how well each
method's unsupervised anomaly score lines up with ground truth. This is
why average precision (PR-AUC) is the headline metric: it is the
appropriate metric for a rare, imbalanced positive class (3% here), unlike
plain accuracy which a "flag nothing" model would ace by doing nothing.

The best-scoring method (by PR-AUC) is saved as the "production" scorer
used by predict.py / the API.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

from data_prep import load_dataset, build_features, FEATURE_COLUMNS, LABEL_COLUMN

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42


def evaluate(y_true: np.ndarray, anomaly_score: np.ndarray, contamination: float) -> dict:
    """anomaly_score: higher = more anomalous, for ALL methods (already normalized
    to this convention by the caller). Threshold = flag the top-K most anomalous
    points, where K = the known contamination rate * n -- a fair, consistent
    way to compare precision/recall across methods with different score scales."""
    n = len(y_true)
    k = max(1, int(round(contamination * n)))
    threshold_idx = np.argsort(anomaly_score)[::-1][:k]
    y_pred = np.zeros(n, dtype=int)
    y_pred[threshold_idx] = 1

    return {
        "pr_auc": round(float(average_precision_score(y_true, anomaly_score)), 4),
        "precision_at_k": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_at_k": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_at_k": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "k_flagged": k,
    }


def main():
    df = load_dataset()
    X = build_features(df)
    y = df[LABEL_COLUMN].values
    contamination = float(y.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    # --- Method 1: Isolation Forest ---
    iso = IsolationForest(contamination=contamination, random_state=RANDOM_SEED, n_estimators=200)
    iso.fit(X_scaled)
    iso_score = -iso.decision_function(X_scaled)  # flip sign: higher = more anomalous
    results["isolation_forest"] = evaluate(y, iso_score, contamination)

    # --- Method 2: Local Outlier Factor (novelty=True so it can score new points later) ---
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
    lof.fit(X_scaled)
    lof_score = -lof.score_samples(X_scaled)  # flip sign: higher = more anomalous
    results["local_outlier_factor"] = evaluate(y, lof_score, contamination)

    # --- Method 3: Z-score statistical baseline ---
    z_scores = np.abs(X_scaled)  # X_scaled is already (x - mean)/std per feature
    z_score_max = z_scores.max(axis=1)  # max |z| across features = simple multivariate outlier score
    results["z_score_baseline"] = evaluate(y, z_score_max, contamination)

    print("[train] method comparison (PR-AUC is the headline metric):")
    for method, m in results.items():
        print(f"    {method:22s} PR-AUC={m['pr_auc']:.4f}  "
              f"precision@k={m['precision_at_k']:.4f}  recall@k={m['recall_at_k']:.4f}  f1@k={m['f1_at_k']:.4f}")

    best_method = max(results, key=lambda m: results[m]["pr_auc"])
    print(f"[train] best method by PR-AUC: {best_method}")

    # Save the production scorer (Isolation Forest -- fast, no need to store training data
    # for scoring new points, unlike LOF which needs its neighbor index kept in memory)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(iso, MODELS_DIR / "isolation_forest.joblib")

    # Save every transaction's Isolation Forest anomaly score, for the review-queue dashboard
    df_scored = df.copy()
    df_scored["anomaly_score"] = iso_score
    df_scored["flagged"] = 0
    top_k_idx = np.argsort(iso_score)[::-1][:results["isolation_forest"]["k_flagged"]]
    df_scored.loc[df_scored.index[top_k_idx], "flagged"] = 1
    df_scored.sort_values("anomaly_score", ascending=False).to_csv(
        MODELS_DIR / "scored_transactions.csv", index=False
    )

    metrics = {
        "n_transactions": int(len(df)),
        "n_anomalies_labeled": int(y.sum()),
        "contamination_rate": round(contamination, 4),
        "features": FEATURE_COLUMNS,
        "methods_compared": results,
        "best_method": best_method,
        "production_method": "isolation_forest",
        "random_seed": RANDOM_SEED,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train] saved production model -> {MODELS_DIR / 'isolation_forest.joblib'}")
    print(f"[train] saved metrics -> {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
