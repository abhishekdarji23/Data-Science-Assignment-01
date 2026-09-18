"""
skills.py
---------
CRISP-DM Phases 2-6, implemented as a catalog of independently-executable
"skills" -- the point of this project (per the original repo's own
follow-up prompt: "execute skill live... make that visual interactive and
look like a proper dashboard, not raw JSON").

Each skill is a function `df -> SkillResult` registered in SKILL_REGISTRY.
Every result carries a `display_type` so the frontend renders it as an
actual table/metric-grid/bar-chart -- never a JSON dump -- without the
frontend needing to know anything skill-specific.

display_type contract:
  "metrics"    -> data: [{"label": str, "value": str}, ...]
  "table"      -> data: {"columns": [str,...], "rows": [[...], ...]}
  "bar_chart"  -> data: {"labels": [str,...], "values": [float,...]}
  "list"       -> data: [str, ...]
"""
from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

TARGET = "Survived"
NUMERIC_COLS = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_COLS = ["Pclass", "Sex", "Embarked"]
RANDOM_SEED = 42


@dataclass
class SkillResult:
    summary: str
    display_type: str
    data: Any


@dataclass
class Skill:
    id: str
    title: str
    crisp_dm_phase: str
    description: str
    fn: Callable[[pd.DataFrame], SkillResult]


SKILL_REGISTRY: list[Skill] = []


def skill(id: str, title: str, crisp_dm_phase: str, description: str):
    def decorator(fn):
        SKILL_REGISTRY.append(Skill(id=id, title=title, crisp_dm_phase=crisp_dm_phase,
                                     description=description, fn=fn))
        return fn
    return decorator


def _metrics(*pairs):
    return [{"label": k, "value": v} for k, v in pairs]


# ---------------------------------------------------------------------------
# DATA UNDERSTANDING
# ---------------------------------------------------------------------------

@skill("dataset_overview", "Dataset Overview", "Data Understanding",
       "Row/column counts, memory footprint, and column types at a glance.")
def dataset_overview(df: pd.DataFrame) -> SkillResult:
    return SkillResult(
        summary=f"{len(df):,} rows × {df.shape[1]} columns.",
        display_type="metrics",
        data=_metrics(
            ("Rows", f"{len(df):,}"),
            ("Columns", str(df.shape[1])),
            ("Numeric columns", str(len(NUMERIC_COLS))),
            ("Categorical columns", str(len(CATEGORICAL_COLS))),
            ("Target column", TARGET),
            ("Memory usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"),
        ),
    )


@skill("missing_values", "Missing Value Summary", "Data Understanding",
       "Which columns have missing data, and how much.")
def missing_values(df: pd.DataFrame) -> SkillResult:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    rows = [[col, int(n), f"{100*n/len(df):.1f}%"] for col, n in miss.items()]
    summary = (f"{len(rows)} column(s) have missing values." if len(rows)
               else "No missing values found.")
    return SkillResult(summary=summary, display_type="table",
                        data={"columns": ["Column", "Missing count", "Missing %"], "rows": rows})


@skill("duplicate_rows", "Duplicate Row Check", "Data Understanding",
       "Exact-duplicate rows (ignoring the row ID), which inflate any model trained on them.")
def duplicate_rows(df: pd.DataFrame) -> SkillResult:
    check_cols = [c for c in df.columns if c != "PassengerId"]
    dup_mask = df.duplicated(subset=check_cols, keep=False)
    n_dup = int(dup_mask.sum())
    ids = df.loc[dup_mask, "PassengerId"].astype(str).tolist()
    return SkillResult(
        summary=f"{n_dup} duplicate row(s) found ({100*n_dup/len(df):.1f}% of dataset).",
        display_type="list",
        data=[f"PassengerId {i}" for i in ids] if ids else ["No duplicates found."],
    )


@skill("descriptive_stats", "Descriptive Statistics", "Data Understanding",
       "Mean, std, min, max for every numeric column.")
def descriptive_stats(df: pd.DataFrame) -> SkillResult:
    desc = df[NUMERIC_COLS].describe().T[["mean", "std", "min", "max"]].round(2)
    rows = [[idx, row["mean"], row["std"], row["min"], row["max"]] for idx, row in desc.iterrows()]
    return SkillResult(
        summary=f"Summary statistics computed for {len(NUMERIC_COLS)} numeric columns.",
        display_type="table",
        data={"columns": ["Column", "Mean", "Std Dev", "Min", "Max"], "rows": rows},
    )


@skill("outlier_detection", "Outlier Detection (Tukey IQR)", "Data Understanding",
       "Flags values outside [Q1-1.5*IQR, Q3+1.5*IQR] for each numeric column.")
def outlier_detection(df: pd.DataFrame) -> SkillResult:
    rows = []
    for col in NUMERIC_COLS:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lower) | (s > upper)).sum())
        rows.append([col, n_out, f"{100*n_out/len(s):.1f}%", round(lower, 1), round(upper, 1)])
    total = sum(r[1] for r in rows)
    return SkillResult(
        summary=f"{total} total outlier value(s) flagged across {len(NUMERIC_COLS)} numeric columns.",
        display_type="table",
        data={"columns": ["Column", "Outliers", "% of column", "Lower bound", "Upper bound"], "rows": rows},
    )


@skill("categorical_counts", "Categorical Value Counts", "Data Understanding",
       "Distribution of each categorical column's values.")
def categorical_counts(df: pd.DataFrame) -> SkillResult:
    rows = []
    for col in CATEGORICAL_COLS:
        counts = df[col].value_counts(dropna=False)
        top = ", ".join(f"{k}={v}" for k, v in counts.items())
        rows.append([col, df[col].nunique(dropna=True), top])
    return SkillResult(
        summary=f"Value distributions computed for {len(CATEGORICAL_COLS)} categorical columns.",
        display_type="table",
        data={"columns": ["Column", "Unique values", "Counts"], "rows": rows},
    )


@skill("correlation_matrix", "Correlation Matrix", "Data Understanding",
       "Pearson correlation between numeric columns (and the target).")
def correlation_matrix(df: pd.DataFrame) -> SkillResult:
    cols = NUMERIC_COLS + [TARGET]
    corr = df[cols].corr().round(2)
    rows = [[idx] + [corr.loc[idx, c] for c in cols] for idx in corr.index]
    strongest = corr[TARGET].drop(TARGET).abs().idxmax()
    return SkillResult(
        summary=f"Strongest correlation with {TARGET} is {strongest} (r={corr.loc[strongest, TARGET]}).",
        display_type="table",
        data={"columns": [""] + cols, "rows": rows},
    )


@skill("class_balance", "Target Class Balance", "Data Understanding",
       "How imbalanced is the prediction target? Affects which evaluation metrics matter.")
def class_balance(df: pd.DataFrame) -> SkillResult:
    counts = df[TARGET].value_counts().sort_index()
    labels = [f"Survived={i}" for i in counts.index]
    ratio = counts.min() / counts.max()
    return SkillResult(
        summary=f"Class ratio (minority:majority) = {ratio:.2f}. "
                + ("Reasonably balanced." if ratio > 0.5 else "Notably imbalanced — accuracy alone would be misleading."),
        display_type="bar_chart",
        data={"labels": labels, "values": [int(v) for v in counts.values]},
    )


# ---------------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------------

@skill("missing_imputation", "Missing Value Imputation", "Data Preparation",
       "Fills Age with the median and Embarked with the mode — before/after comparison.")
def missing_imputation(df: pd.DataFrame) -> SkillResult:
    before_age = int(df["Age"].isna().sum())
    before_emb = int(df["Embarked"].isna().sum())
    age_filled = df["Age"].fillna(df["Age"].median())
    emb_filled = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])
    return SkillResult(
        summary=f"Age: {before_age} → 0 missing (filled with median {df['Age'].median():.1f}). "
                f"Embarked: {before_emb} → 0 missing (filled with mode '{df['Embarked'].mode().iloc[0]}').",
        display_type="metrics",
        data=_metrics(
            ("Age missing before", str(before_age)),
            ("Age missing after", "0"),
            ("Age fill value (median)", f"{df['Age'].median():.1f}"),
            ("Embarked missing before", str(before_emb)),
            ("Embarked missing after", "0"),
            ("Embarked fill value (mode)", str(df["Embarked"].mode().iloc[0])),
        ),
    )


@skill("categorical_encoding", "Categorical Encoding Preview", "Data Preparation",
       "One-hot encodes Sex, Pclass, and Embarked — shows how column count expands.")
def categorical_encoding(df: pd.DataFrame) -> SkillResult:
    sub = df[CATEGORICAL_COLS].fillna("missing").astype(str)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(sub)
    new_cols = encoder.get_feature_names_out(CATEGORICAL_COLS)
    preview_rows = [[sub.iloc[i][c] for c in CATEGORICAL_COLS] +
                     [int(v) for v in encoded[i]] for i in range(min(5, len(df)))]
    return SkillResult(
        summary=f"{len(CATEGORICAL_COLS)} categorical columns → {len(new_cols)} one-hot columns.",
        display_type="table",
        data={"columns": CATEGORICAL_COLS + list(new_cols), "rows": preview_rows},
    )


@skill("feature_scaling", "Feature Scaling (StandardScaler)", "Data Preparation",
       "Standardizes numeric columns to mean=0, std=1 — before/after comparison.")
def feature_scaling(df: pd.DataFrame) -> SkillResult:
    X = df[NUMERIC_COLS].fillna(df[NUMERIC_COLS].median())
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=NUMERIC_COLS)
    rows = []
    for col in NUMERIC_COLS:
        rows.append([col, round(X[col].mean(), 2), round(X[col].std(), 2),
                     round(X_scaled[col].mean(), 2), round(X_scaled[col].std(), 2)])
    return SkillResult(
        summary="After scaling, every column has mean≈0 and std≈1 — required for distance/gradient-based models.",
        display_type="table",
        data={"columns": ["Column", "Mean (before)", "Std (before)", "Mean (after)", "Std (after)"], "rows": rows},
    )


# ---------------------------------------------------------------------------
# shared modeling helper
# ---------------------------------------------------------------------------

def _prepare_model_features(df: pd.DataFrame):
    X_num = df[NUMERIC_COLS].fillna(df[NUMERIC_COLS].median())
    X_cat = pd.get_dummies(df[CATEGORICAL_COLS].fillna("missing").astype(str), drop_first=True)
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    y = df[TARGET].reset_index(drop=True)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y


# ---------------------------------------------------------------------------
# MODELING
# ---------------------------------------------------------------------------

@skill("train_test_split", "Train/Test Split", "Modeling",
       "80/20 stratified split — preserves the target's class balance in both halves.")
def train_test_split_skill(df: pd.DataFrame) -> SkillResult:
    X, y = _prepare_model_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    return SkillResult(
        summary=f"{len(X_train)} train rows, {len(X_test)} test rows — stratified so survival rate matches in both.",
        display_type="metrics",
        data=_metrics(
            ("Train rows", str(len(X_train))),
            ("Test rows", str(len(X_test))),
            ("Train survival rate", f"{y_train.mean():.1%}"),
            ("Test survival rate", f"{y_test.mean():.1%}"),
            ("Split type", "80/20 stratified"),
        ),
    )


@skill("baseline_vs_model", "Baseline vs. Logistic Regression", "Modeling",
       "Compares a majority-class baseline against a real trained classifier.")
def baseline_vs_model(df: pd.DataFrame) -> SkillResult:
    X, y = _prepare_model_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    baseline_pred = np.full(len(y_test), y_train.mode().iloc[0])
    baseline_acc = accuracy_score(y_test, baseline_pred)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_acc = accuracy_score(y_test, model_pred)

    lift = model_acc - baseline_acc
    return SkillResult(
        summary=f"Logistic regression beats the majority-class baseline by {lift:+.1%} accuracy.",
        display_type="bar_chart",
        data={"labels": ["Majority-class baseline", "Logistic Regression"],
              "values": [round(float(baseline_acc), 4), round(float(model_acc), 4)]},
    )


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

@skill("cross_validation", "5-Fold Cross-Validation", "Evaluation",
       "Checks that the model's accuracy is stable across different data splits, not a lucky single split.")
def cross_validation_skill(df: pd.DataFrame) -> SkillResult:
    X, y = _prepare_model_features(df)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    return SkillResult(
        summary=f"Mean accuracy {scores.mean():.1%} ± {scores.std():.1%} across 5 folds.",
        display_type="bar_chart",
        data={"labels": [f"Fold {i+1}" for i in range(5)], "values": [round(float(s), 4) for s in scores]},
    )


@skill("confusion_matrix", "Confusion Matrix & Metrics", "Evaluation",
       "Precision, recall, F1, and the confusion matrix on the held-out test set.")
def confusion_matrix_skill(df: pd.DataFrame) -> SkillResult:
    X, y = _prepare_model_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    cm = confusion_matrix(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    return SkillResult(
        summary=f"Precision {precision:.1%}, Recall {recall:.1%}, F1 {f1:.1%} on the held-out test set.",
        display_type="table",
        data={
            "columns": ["", "Predicted: Did not survive", "Predicted: Survived"],
            "rows": [
                ["Actual: Did not survive", int(cm[0][0]), int(cm[0][1])],
                ["Actual: Survived", int(cm[1][0]), int(cm[1][1])],
            ],
        },
    )


@skill("feature_importance", "Feature Importance", "Evaluation",
       "Which features move the model's prediction the most (logistic regression coefficient magnitude).")
def feature_importance_skill(df: pd.DataFrame) -> SkillResult:
    X, y = _prepare_model_features(df)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X, y)
    importance = pd.Series(np.abs(model.coef_[0]), index=X.columns).sort_values(ascending=False).head(8)
    top_feature = importance.index[0]
    return SkillResult(
        summary=f"'{top_feature}' has the strongest influence on the model's survival prediction.",
        display_type="bar_chart",
        data={"labels": list(importance.index), "values": [round(float(v), 3) for v in importance.values]},
    )


def get_skill(skill_id: str) -> Skill | None:
    for s in SKILL_REGISTRY:
        if s.id == skill_id:
            return s
    return None


def list_skills() -> list[dict]:
    return [{"id": s.id, "title": s.title, "crisp_dm_phase": s.crisp_dm_phase,
             "description": s.description} for s in SKILL_REGISTRY]
