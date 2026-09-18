"""
train.py
--------
CRISP-DM Phase 4 (Modeling) + Phase 5 (Evaluation).

Trains and compares 3 models on the SAME train/test split, all using
Ridge regression (kept identical across all 3 so the comparison isolates
the effect of WHICH FEATURES each model sees, not a difference in
algorithm):

  1. tabular_only  -- bedrooms/bathrooms/accommodates/room_type/neighborhood
  2. text_only     -- TF-IDF of the listing description
  3. fusion        -- tabular features AND TF-IDF text features, concatenated
                       into one feature matrix for a single model

This is the actual technique "multimodal AutoML" refers to (feature-level
fusion of heterogeneous modalities into one model), implemented directly
with scikit-learn rather than depending on the real AutoGluon Multimodal
package -- see PROMPTS.md for why (torch/transformers/timm dependency
weight conflicts with "keep it simple").

Because the synthetic data's price genuinely depends on BOTH the tabular
fields AND the hidden "quality" signal only expressed in the text (see
generate_data.py's docstring), an honest evaluation should show:
tabular_only and text_only each capture PART of the price variance,
and fusion captures MORE than either alone -- this is checked directly
in AUDIT_REPORT.md, not just claimed.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from data_prep import (
    load_dataset, build_tabular_features, get_tabular_feature_names,
    TEXT_COLUMN, TARGET_COLUMN,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42
RIDGE_ALPHA = 5.0
MAX_TEXT_FEATURES = 300


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae_usd": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main():
    df = load_dataset()
    X_tabular = build_tabular_features(df)
    y = df[TARGET_COLUMN].values
    texts = df[TEXT_COLUMN].astype(str)

    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=RANDOM_SEED
    )

    y_train, y_test = y[idx_train], y[idx_test]

    # --- Tabular features: scaled ---
    scaler = StandardScaler()
    X_tab_train = scaler.fit_transform(X_tabular.iloc[idx_train])
    X_tab_test = scaler.transform(X_tabular.iloc[idx_test])

    # --- Text features: TF-IDF, fit ONLY on the training split ---
    vectorizer = TfidfVectorizer(max_features=MAX_TEXT_FEATURES, stop_words="english")
    X_text_train = vectorizer.fit_transform(texts.iloc[idx_train])
    X_text_test = vectorizer.transform(texts.iloc[idx_test])

    # --- Model 1: tabular only ---
    model_tabular = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)
    model_tabular.fit(X_tab_train, y_train)
    pred_tabular = model_tabular.predict(X_tab_test)
    metrics_tabular = evaluate(y_test, pred_tabular)

    # --- Model 2: text only ---
    model_text = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)
    model_text.fit(X_text_train, y_train)
    pred_text = model_text.predict(X_text_test)
    metrics_text = evaluate(y_test, pred_text)

    # --- Model 3: fusion (tabular + text concatenated) ---
    X_fusion_train = hstack([csr_matrix(X_tab_train), X_text_train])
    X_fusion_test = hstack([csr_matrix(X_tab_test), X_text_test])
    model_fusion = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)
    model_fusion.fit(X_fusion_train, y_train)
    pred_fusion = model_fusion.predict(X_fusion_test)
    metrics_fusion = evaluate(y_test, pred_fusion)

    leaderboard = {
        "tabular_only": metrics_tabular,
        "text_only": metrics_text,
        "fusion": metrics_fusion,
    }
    print("[train] leaderboard (Ridge regression, same algorithm, different features):")
    for name, m in leaderboard.items():
        print(f"    {name:15s} R2={m['r2']:.4f}  MAE=${m['mae_usd']}")

    best_model = max(leaderboard, key=lambda k: leaderboard[k]["r2"])
    fusion_beats_both_singles = (
        metrics_fusion["r2"] > metrics_tabular["r2"] and metrics_fusion["r2"] > metrics_text["r2"]
    )
    print(f"[train] best model: {best_model}  |  fusion beats both single modalities: {fusion_beats_both_singles}")

    joblib.dump(scaler, MODELS_DIR / "tabular_scaler.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "text_vectorizer.joblib")
    joblib.dump(model_tabular, MODELS_DIR / "model_tabular_only.joblib")
    joblib.dump(model_text, MODELS_DIR / "model_text_only.joblib")
    joblib.dump(model_fusion, MODELS_DIR / "model_fusion.joblib")

    metrics = {
        "n_listings": int(len(df)),
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "tabular_feature_names": get_tabular_feature_names(),
        "text_max_features": MAX_TEXT_FEATURES,
        "ridge_alpha": RIDGE_ALPHA,
        "leaderboard": leaderboard,
        "best_model": best_model,
        "fusion_beats_both_singles": fusion_beats_both_singles,
        "random_seed": RANDOM_SEED,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train] saved 3 models + vectorizer + scaler -> {MODELS_DIR}")
    print(f"[train] saved metrics -> {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
