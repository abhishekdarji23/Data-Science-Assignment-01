"""
audit_runner.py
----------------
Ties pipeline.py (the subject) and audit_checks.py (the checks) together:
runs the pipeline for a given variant, runs every registered check against
the result, and produces a compliance report.
"""
from dataclasses import asdict

from pipeline import run_pipeline, PipelineVariant
from audit_checks import run_all_checks


def run_audit(variant: PipelineVariant = "clean") -> dict:
    artifacts = run_pipeline(variant)
    results = run_all_checks(artifacts)

    n_passed = sum(1 for r in results if r.passed)
    n_total = len(results)

    by_phase: dict[str, list[dict]] = {}
    for r in results:
        by_phase.setdefault(r.crisp_dm_phase, []).append(asdict(r))

    return {
        "variant": variant,
        "compliance_score": round(100 * n_passed / n_total, 1) if n_total else 0.0,
        "checks_passed": n_passed,
        "checks_total": n_total,
        "pipeline_summary": {
            "n_rows_raw": len(artifacts.raw_df),
            "n_rows_used_for_training": len(artifacts.working_df),
            "feature_columns": artifacts.feature_columns,
            "split_method": artifacts.split_method,
            "model_mae_seconds": round(artifacts.model_mae, 1),
            "model_r2": round(artifacts.model_r2, 4),
            "baseline_mae_seconds": round(artifacts.baseline_mae, 1),
        },
        "checks_by_phase": by_phase,
        "checks": [asdict(r) for r in results],
    }


def run_comparison() -> dict:
    """Runs both variants and returns them side by side -- the 'comparison
    report' proving the checks behave differently on a subject that's
    known to violate real practices vs. one that doesn't."""
    return {
        "clean": run_audit("clean"),
        "flawed": run_audit("flawed"),
    }
