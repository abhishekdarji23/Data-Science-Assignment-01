"""
train.py
--------
CRISP-DM Phase 4 (Modeling) + Phase 5 (Evaluation).

Fits a KMeans clustering model to segment customers by Age, Annual Income,
and Spending Score. Since clustering is unsupervised, "evaluation" here
means model *selection* (choosing k) via internal validity metrics rather
than held-out accuracy:

  * Elbow method (inertia vs k) -- for a human-readable diagnostic plot's
    worth of data (saved to metrics.json), not used alone to pick k.
  * Silhouette score (the primary metric) -- how well-separated & compact
    are the resulting clusters. Chosen k = argmax(silhouette) over a
    candidate range, which is a more principled tie-breaker than eyeballing
    an elbow plot alone.

Leakage note: fitting a StandardScaler on the FULL dataset before clustering
is standard practice for unsupervised learning (unlike supervised learning,
there is no future/held-out label being leaked -- clustering describes the
data you have). What *would* be a mistake is fitting a fresh scaler at
serving time instead of reusing this one, which is why the scaler is
persisted alongside the model (see predict.py).
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from data_prep import load_dataset, build_features, FEATURE_COLUMNS

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42
K_CANDIDATES = range(2, 9)


def label_segment(row) -> str:
    """Human-readable business label for a cluster centroid, in ORIGINAL
    (unscaled) units -- makes the admin dashboard readable without forcing
    a marketer to interpret raw centroid numbers."""
    income, spend, age = row["Annual Income (k$)"], row["Spending Score (1-100)"], row["Age"]
    if income >= 65 and spend >= 55:
        return "High Income, High Spend (VIP)"
    if income >= 65 and spend < 55:
        return "High Income, Low Spend (Save-to-convert)"
    if income < 45 and spend < 35:
        return "Budget-Conscious"
    if spend >= 55:
        return "Moderate Income, High Spend (Value-driven)"
    return "Moderate, Balanced Spender"


def main():
    raw = load_dataset()
    X = build_features(raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    elbow = []
    silhouette_by_k = {}
    best_k, best_score, best_model = None, -1.0, None
    for k in K_CANDIDATES:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(X_scaled, labels))
        elbow.append({"k": k, "inertia": round(inertia, 2)})
        silhouette_by_k[k] = round(sil, 4)
        if sil > best_score:
            best_k, best_score, best_model = k, sil, km

    print(f"[train] silhouette by k: {silhouette_by_k}")
    print(f"[train] chosen k={best_k} (silhouette={best_score:.4f})")

    final_labels = best_model.predict(X_scaled)
    raw = raw.copy()
    raw["cluster"] = final_labels

    # Cluster profiles in ORIGINAL units (business-readable), for the admin dashboard
    profiles = []
    for c in sorted(raw["cluster"].unique()):
        sub = raw[raw["cluster"] == c]
        centroid = {
            "Age": round(sub["Age"].mean(), 1),
            "Annual Income (k$)": round(sub["Annual Income (k$)"].mean(), 1),
            "Spending Score (1-100)": round(sub["Spending Score (1-100)"].mean(), 1),
        }
        profiles.append({
            "cluster": int(c),
            "label": label_segment(centroid),
            "size": int(len(sub)),
            "pct_of_customers": round(100 * len(sub) / len(raw), 1),
            "avg_age": centroid["Age"],
            "avg_income_k": centroid["Annual Income (k$)"],
            "avg_spending_score": centroid["Spending Score (1-100)"],
        })

    metrics = {
        "n_customers": int(len(raw)),
        "features": FEATURE_COLUMNS,
        "k_candidates_tried": list(K_CANDIDATES),
        "silhouette_by_k": silhouette_by_k,
        "chosen_k": int(best_k),
        "chosen_k_silhouette": round(best_score, 4),
        "elbow_inertia_by_k": elbow,
        "cluster_profiles": profiles,
        "model": "KMeans",
        "random_seed": RANDOM_SEED,
    }
    print("[train] cluster profiles:", json.dumps(profiles, indent=2))

    joblib.dump(best_model, MODELS_DIR / "kmeans_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    raw.to_csv(MODELS_DIR / "customers_with_clusters.csv", index=False)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train] saved model -> {MODELS_DIR / 'kmeans_model.joblib'}")
    print(f"[train] saved metrics -> {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
