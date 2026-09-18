"""
train.py
--------
CRISP-DM Phase 4 (Modeling) + Phase 5 (Evaluation).

Implements the core idea the original project asked for -- AutoML-style
multi-model stacking -- WITHOUT depending on the actual AutoGluon package.
See PROMPTS.md for why: AutoGluon pulls in PyTorch, LightGBM, CatBoost,
and Ray as dependencies (often several GB), which directly conflicts with
"keep it simple." This reimplements the same underlying technique
(Caruana-style stacking) with plain scikit-learn:

  1. Train a small "model zoo" of different algorithm families.
  2. For each, generate OUT-OF-FOLD (OOF) predictions on the training set
     via 5-fold cross-validation -- critical: a model's OOF predictions
     never come from a fold it was trained on, or the meta-learner would
     be trained on leaked, overly-confident predictions.
  3. Stack the OOF predictions into a meta-feature matrix and train a
     meta-learner (logistic regression) on top -- this is "stacking":
     the L2 model learns how to weigh/combine the L1 models' opinions.
  4. Refit every base model on the FULL training set (for serving), and
     evaluate every base model AND the stacked ensemble on a genuinely
     held-out test set that was never used for OOF generation or
     meta-learner training.

This mirrors AutoGluon's own documented approach (a model zoo + k-fold
stacking + a meta-learner) at a scale that trains in seconds.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_prep import load_dataset, build_features, TARGET_COLUMN, get_feature_names

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42
N_FOLDS = 5

# The "model zoo" -- a handful of different algorithm families, the same
# spirit as AutoGluon's default model set (linear, tree ensembles, a
# distance-based method).
BASE_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_SEED),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=RANDOM_SEED),
    "k_nearest_neighbors": KNeighborsClassifier(n_neighbors=15),
    "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED),
}


def evaluate(y_true, y_prob, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def main():
    df = load_dataset()
    X = build_features(df)
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    print(f"[train] model zoo: {list(BASE_MODELS.keys())}")
    oof_predictions = {}   # model_name -> OOF predict_proba on TRAIN (for meta-learner training)
    test_predictions = {}  # model_name -> predict_proba on TEST (for meta-learner + leaderboard)
    leaderboard = {}

    for name, model in BASE_MODELS.items():
        # Step 1: out-of-fold predictions on TRAIN via cross-validation (no leakage)
        oof_proba = cross_val_predict(
            model, X_train_scaled, y_train, cv=skf, method="predict_proba", n_jobs=-1
        )[:, 1]
        oof_predictions[name] = oof_proba

        # Step 2: refit on the FULL training set (for serving + test evaluation)
        model.fit(X_train_scaled, y_train)
        test_proba = model.predict_proba(X_test_scaled)[:, 1]
        test_predictions[name] = test_proba
        test_pred = (test_proba >= 0.5).astype(int)

        leaderboard[name] = evaluate(y_test, test_proba, test_pred)
        joblib.dump(model, MODELS_DIR / f"base_{name}.joblib")
        print(f"    {name:22s} test ROC-AUC={leaderboard[name]['roc_auc']:.4f}  "
              f"F1={leaderboard[name]['f1']:.4f}")

    # Step 3: stack -- train a meta-learner on the OOF predictions
    meta_X_train = pd.DataFrame(oof_predictions)
    meta_X_test = pd.DataFrame(test_predictions)

    meta_learner = LogisticRegression(random_state=RANDOM_SEED)
    meta_learner.fit(meta_X_train, y_train)

    stacked_proba = meta_learner.predict_proba(meta_X_test)[:, 1]
    stacked_pred = (stacked_proba >= 0.5).astype(int)
    leaderboard["stacked_ensemble"] = evaluate(y_test, stacked_proba, stacked_pred)
    print(f"    {'stacked_ensemble':22s} test ROC-AUC={leaderboard['stacked_ensemble']['roc_auc']:.4f}  "
          f"F1={leaderboard['stacked_ensemble']['f1']:.4f}")

    best_model = max(leaderboard, key=lambda m: leaderboard[m]["roc_auc"])
    print(f"[train] best model by ROC-AUC: {best_model}")

    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(meta_learner, MODELS_DIR / "meta_learner.joblib")

    metrics = {
        "n_customers": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
        "feature_names": get_feature_names(),
        "model_zoo": list(BASE_MODELS.keys()),
        "n_folds_for_stacking": N_FOLDS,
        "leaderboard": leaderboard,
        "best_model": best_model,
        "meta_learner_weights": {
            name: round(float(w), 3)
            for name, w in zip(BASE_MODELS.keys(), meta_learner.coef_[0])
        },
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train] saved {len(BASE_MODELS)} base models + meta-learner -> {MODELS_DIR}")
    print(f"[train] saved metrics -> {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
