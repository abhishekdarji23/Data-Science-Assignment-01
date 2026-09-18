"""
pipeline.py
-----------
This is the SUBJECT being audited, not the audit platform itself (that's
audit_checks.py / audit_runner.py). It builds and trains a small NYC-taxi
trip-duration model -- CRISP-DM Phases 2-5 -- and can run in two modes:

  * "clean"  -- follows correct practice at every point this audit checks
  * "flawed" -- deliberately violates 3 specific, real practices:
      1. target leakage: includes a feature derived directly from the
         target (trip_duration) as a model input
      2. random (shuffled) train/test split instead of time-based, for
         data that is genuinely a time series
      3. duplicate rows are NOT removed before training

The "flawed" mode exists so the audit platform's checks can be run
against a subject KNOWN to violate each rule, and be shown to actually
catch it -- rather than only ever being demonstrated against data that
already passes, which would prove nothing about whether the checks work.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "taxi_trips_synthetic.csv"
RANDOM_SEED = 42

BASE_FEATURE_COLUMNS = ["passenger_count", "distance_mi", "hour", "is_rush_hour"]
LEAKY_FEATURE_COLUMN = "duration_bucket"  # derived directly from the target -- leakage if included

PipelineVariant = Literal["clean", "flawed"]


@dataclass
class PipelineArtifacts:
    variant: PipelineVariant
    raw_df: pd.DataFrame
    working_df: pd.DataFrame
    feature_columns: list[str]
    duplicates_found: int
    duplicates_removed: int
    split_method: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    model: GradientBoostingRegressor
    baseline_mae: float
    model_mae: float
    model_r2: float
    random_seed_used: int | None


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["is_rush_hour"] = (((df["hour"] >= 7) & (df["hour"] <= 10)) |
                           ((df["hour"] >= 16) & (df["hour"] <= 19))).astype(int)
    return df


def run_pipeline(variant: PipelineVariant = "clean") -> PipelineArtifacts:
    raw_df = pd.read_csv(DATA_PATH)
    df = _add_time_features(raw_df)

    check_cols = [c for c in df.columns if c != "trip_id"]
    duplicates_found = int(df.duplicated(subset=check_cols).sum())

    if variant == "flawed":
        # Violation 3: duplicates NOT removed
        duplicates_removed = 0
        working_df = df.copy()
        # Violation 1: a feature derived directly from the target
        working_df[LEAKY_FEATURE_COLUMN] = working_df["trip_duration"] // 300
        feature_columns = BASE_FEATURE_COLUMNS + [LEAKY_FEATURE_COLUMN]
    else:
        # Clean: duplicates removed before anything else touches the data
        working_df = df.drop_duplicates(subset=check_cols).reset_index(drop=True)
        duplicates_removed = duplicates_found
        feature_columns = list(BASE_FEATURE_COLUMNS)

    X = working_df[feature_columns]
    y = working_df["trip_duration"]

    if variant == "flawed":
        # Violation 2: random shuffle split on time-series data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=None, shuffle=True
        )
        split_method = "random_shuffle (no seed)"
        random_seed_used = None
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3)  # unseeded
    else:
        working_df_sorted = working_df.sort_values("pickup_datetime").reset_index(drop=True)
        X_sorted = working_df_sorted[feature_columns]
        y_sorted = working_df_sorted["trip_duration"]
        split_idx = int(len(working_df_sorted) * 0.8)
        X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
        y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
        split_method = "time_based_80_20"
        random_seed_used = RANDOM_SEED
        model = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=RANDOM_SEED)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))
    model_mae = float(mean_absolute_error(y_test, pred))
    model_r2 = float(r2_score(y_test, pred))

    return PipelineArtifacts(
        variant=variant,
        raw_df=df,
        working_df=working_df,
        feature_columns=feature_columns,
        duplicates_found=duplicates_found,
        duplicates_removed=duplicates_removed,
        split_method=split_method,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        model=model,
        baseline_mae=baseline_mae,
        model_mae=model_mae,
        model_r2=model_r2,
        random_seed_used=random_seed_used,
    )
