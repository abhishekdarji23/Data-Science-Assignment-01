"""
test_api.py
-----------
Minimal smoke tests, plus the tests that actually matter for an audit
platform: that it correctly discriminates a clean pipeline from a
deliberately flawed one, check by check.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200


def test_checks_catalog_has_10_checks_across_multiple_phases():
    r = client.get("/api/audit/checks")
    assert r.status_code == 200
    checks = r.json()
    assert len(checks) == 10
    phases = {c["crisp_dm_phase"] for c in checks}
    assert len(phases) >= 4


def test_clean_pipeline_passes_every_check():
    r = client.post("/api/audit/run?variant=clean")
    assert r.status_code == 200
    body = r.json()
    assert body["compliance_score"] == 100.0
    assert body["checks_passed"] == body["checks_total"]


def test_flawed_pipeline_fails_the_4_deliberate_violations_plus_the_corroborating_signal():
    """4 deliberately-introduced violations, plus the independent
    'suspiciously high R2' check that fires as a side effect of the
    leakage -- 5 failures total. The 4 structural checks are deterministic;
    the R2 check depends on an unseeded model fit (by design -- that's
    itself one of the violations), so it's asserted separately with a
    comfortable margin rather than folded into a strict set-equality
    check that a borderline R2 could occasionally flip."""
    r = client.post("/api/audit/run?variant=flawed")
    assert r.status_code == 200
    body = r.json()
    failed_ids = {c["id"] for c in body["checks"] if not c["passed"]}
    deterministic_failures = {
        "no_target_leakage",
        "time_based_split",
        "duplicates_removed",
        "reproducibility_seeded",
    }
    assert deterministic_failures.issubset(failed_ids), f"missing expected failures: {deterministic_failures - failed_ids}"
    # Not folded into the set above because it depends on the unseeded fit's
    # exact R2 (see pipeline.py) -- checked with a comfortable margin instead
    # (30 trials observed R2 in [0.967, 0.978], well clear of the 0.95 threshold).
    assert body["pipeline_summary"]["model_r2"] > 0.95


def test_flawed_pipeline_scores_lower_than_clean():
    clean = client.post("/api/audit/run?variant=clean").json()
    flawed = client.post("/api/audit/run?variant=flawed").json()
    assert flawed["compliance_score"] < clean["compliance_score"]


def test_compare_endpoint_returns_both_variants():
    r = client.post("/api/audit/compare")
    assert r.status_code == 200
    body = r.json()
    assert body["clean"]["compliance_score"] > body["flawed"]["compliance_score"]


def test_flawed_pipeline_leakage_inflates_apparent_r2():
    """The whole point of including this check: leakage doesn't just
    violate methodology, it produces MISLEADINGLY BETTER metrics -- so an
    audit that only looked at 'is R2 high' without structural checks
    would be fooled in the wrong direction."""
    clean = client.post("/api/audit/run?variant=clean").json()
    flawed = client.post("/api/audit/run?variant=flawed").json()
    assert flawed["pipeline_summary"]["model_r2"] > clean["pipeline_summary"]["model_r2"]
