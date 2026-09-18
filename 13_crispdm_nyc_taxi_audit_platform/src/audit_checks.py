"""
audit_checks.py
----------------
Each check is a function `PipelineArtifacts -> AuditCheckResult`, run
against whatever pipeline variant was actually executed (clean or
flawed). Nothing here is a canned pass/fail -- every check inspects the
real feature list, split method, data, and trained model produced by
pipeline.py moments earlier.
"""
from dataclasses import dataclass
from typing import Callable

from pipeline import PipelineArtifacts, LEAKY_FEATURE_COLUMN


@dataclass
class AuditCheckResult:
    id: str
    title: str
    crisp_dm_phase: str
    passed: bool
    detail: str
    evidence: dict


@dataclass
class AuditCheck:
    id: str
    title: str
    crisp_dm_phase: str
    description: str
    fn: Callable[[PipelineArtifacts], AuditCheckResult]


CHECK_REGISTRY: list[AuditCheck] = []


def check(id: str, title: str, crisp_dm_phase: str, description: str):
    def decorator(fn):
        CHECK_REGISTRY.append(AuditCheck(id=id, title=title, crisp_dm_phase=crisp_dm_phase,
                                          description=description, fn=fn))
        return fn
    return decorator


@check("no_target_leakage", "No target leakage in features", "Data Preparation",
       "The known-leaky column (derived directly from trip_duration) must not be a model input.")
def no_target_leakage(a: PipelineArtifacts) -> AuditCheckResult:
    leaked = LEAKY_FEATURE_COLUMN in a.feature_columns
    return AuditCheckResult(
        id="no_target_leakage", title="No target leakage in features", crisp_dm_phase="Data Preparation",
        passed=not leaked,
        detail=(f"Feature list includes '{LEAKY_FEATURE_COLUMN}', which is computed directly from "
                f"trip_duration (the target). This is target leakage." if leaked
                else "No feature is derived from the target."),
        evidence={"feature_columns": a.feature_columns, "leaky_column_present": leaked},
    )


@check("time_based_split", "Time-based train/test split", "Modeling",
       "Trip data is a time series; a random shuffle split would leak future patterns into training.")
def time_based_split(a: PipelineArtifacts) -> AuditCheckResult:
    passed = a.split_method.startswith("time_based")
    return AuditCheckResult(
        id="time_based_split", title="Time-based train/test split", crisp_dm_phase="Modeling",
        passed=passed,
        detail=f"Split method used: '{a.split_method}'.",
        evidence={"split_method": a.split_method},
    )


@check("duplicates_removed", "Duplicate rows removed before training", "Data Preparation",
       "Exact-duplicate rows inflate whatever split they land in and bias the model toward them.")
def duplicates_removed(a: PipelineArtifacts) -> AuditCheckResult:
    passed = a.duplicates_removed == a.duplicates_found
    return AuditCheckResult(
        id="duplicates_removed", title="Duplicate rows removed before training", crisp_dm_phase="Data Preparation",
        passed=passed,
        detail=f"{a.duplicates_found} duplicate row(s) found in raw data; {a.duplicates_removed} removed before training.",
        evidence={"duplicates_found": a.duplicates_found, "duplicates_removed": a.duplicates_removed},
    )


@check("reproducibility_seeded", "Training is reproducible (seeded)", "Modeling",
       "An unseeded model or split means re-running the pipeline produces different results every time.")
def reproducibility_seeded(a: PipelineArtifacts) -> AuditCheckResult:
    passed = a.random_seed_used is not None
    return AuditCheckResult(
        id="reproducibility_seeded", title="Training is reproducible (seeded)", crisp_dm_phase="Modeling",
        passed=passed,
        detail=(f"Random seed {a.random_seed_used} used for split and model." if passed
                else "No random seed was set for the split or model -- results will differ on every run."),
        evidence={"random_seed_used": a.random_seed_used},
    )


@check("model_beats_baseline", "Model meaningfully beats a naive baseline", "Evaluation",
       "A model that barely beats predicting the mean duration for every trip isn't learning much.")
def model_beats_baseline(a: PipelineArtifacts) -> AuditCheckResult:
    improvement = (a.baseline_mae - a.model_mae) / a.baseline_mae if a.baseline_mae > 0 else 0
    passed = improvement > 0.10  # at least 10% MAE improvement over baseline
    return AuditCheckResult(
        id="model_beats_baseline", title="Model meaningfully beats a naive baseline", crisp_dm_phase="Evaluation",
        passed=passed,
        detail=f"Baseline MAE={a.baseline_mae:.1f}s, model MAE={a.model_mae:.1f}s "
               f"({improvement:.1%} improvement).",
        evidence={"baseline_mae": a.baseline_mae, "model_mae": a.model_mae, "improvement_pct": round(improvement, 4)},
    )


@check("no_missing_values", "No missing values in model features", "Data Preparation",
       "An unhandled NaN in a feature column would either crash training or silently corrupt it.")
def no_missing_values(a: PipelineArtifacts) -> AuditCheckResult:
    na_counts = a.working_df[a.feature_columns].isna().sum()
    total_na = int(na_counts.sum())
    return AuditCheckResult(
        id="no_missing_values", title="No missing values in model features", crisp_dm_phase="Data Preparation",
        passed=total_na == 0,
        detail=f"{total_na} missing value(s) across feature columns." if total_na else "No missing values found.",
        evidence={"na_by_column": {k: int(v) for k, v in na_counts.items() if v > 0}},
    )


@check("feature_target_correlation_sane", "Distance is positively correlated with duration", "Data Understanding",
       "A basic domain sanity check: farther trips should generally take longer. If not, something upstream is broken.")
def feature_target_correlation_sane(a: PipelineArtifacts) -> AuditCheckResult:
    corr = float(a.working_df["distance_mi"].corr(a.working_df["trip_duration"]))
    passed = corr > 0.3
    return AuditCheckResult(
        id="feature_target_correlation_sane", title="Distance is positively correlated with duration",
        crisp_dm_phase="Data Understanding",
        passed=passed,
        detail=f"Correlation(distance_mi, trip_duration) = {corr:.3f}.",
        evidence={"correlation": round(corr, 4)},
    )


@check("suspiciously_high_r2", "Test R2 is not suspiciously close to perfect", "Evaluation",
       "An R2 above ~0.95 on a noisy real-world quantity like trip duration is itself a leakage smell, independent of the feature-list check.")
def suspiciously_high_r2(a: PipelineArtifacts) -> AuditCheckResult:
    passed = a.model_r2 < 0.95
    return AuditCheckResult(
        id="suspiciously_high_r2", title="Test R2 is not suspiciously close to perfect", crisp_dm_phase="Evaluation",
        passed=passed,
        detail=f"Test R2 = {a.model_r2:.4f}." + ("" if passed else " This is unusually high for noisy real-world trip duration data -- investigate for leakage."),
        evidence={"r2": round(a.model_r2, 4)},
    )


@check("prediction_determinism", "Live predictions are deterministic", "Deployment",
       "The same input should produce the same prediction every time -- a live, executable check, not just a code-review claim.")
def prediction_determinism(a: PipelineArtifacts) -> AuditCheckResult:
    row = a.X_test.iloc[[0]]
    pred1 = float(a.model.predict(row)[0])
    pred2 = float(a.model.predict(row)[0])
    passed = pred1 == pred2
    return AuditCheckResult(
        id="prediction_determinism", title="Live predictions are deterministic", crisp_dm_phase="Deployment",
        passed=passed,
        detail=f"Same input predicted twice: {pred1:.4f} and {pred2:.4f}.",
        evidence={"prediction_1": pred1, "prediction_2": pred2},
    )


@check("no_duplicate_feature_names", "No duplicate entries in the feature list", "Data Preparation",
       "A duplicated feature name would silently double-count that signal.")
def no_duplicate_feature_names(a: PipelineArtifacts) -> AuditCheckResult:
    passed = len(a.feature_columns) == len(set(a.feature_columns))
    return AuditCheckResult(
        id="no_duplicate_feature_names", title="No duplicate entries in the feature list", crisp_dm_phase="Data Preparation",
        passed=passed,
        detail=f"Feature list: {a.feature_columns}.",
        evidence={"feature_columns": a.feature_columns},
    )


def run_all_checks(artifacts: PipelineArtifacts) -> list[AuditCheckResult]:
    return [c.fn(artifacts) for c in CHECK_REGISTRY]


def list_checks() -> list[dict]:
    return [{"id": c.id, "title": c.title, "crisp_dm_phase": c.crisp_dm_phase, "description": c.description}
            for c in CHECK_REGISTRY]
