# SKILLS.md — Skills Demonstrated / Applied in This Build

| # | Skill | Where applied |
|---|---|---|
| 1 | Building an audit that can be proven to work, not just run | `src/pipeline.py`'s `flawed` variant + `tests/test_api.py` — the audit is demonstrated against a subject with KNOWN violations, and tested to confirm it actually catches them |
| 2 | Catching the same problem two independent ways | `src/audit_checks.py`: `no_target_leakage` (structural) and `suspiciously_high_r2` (behavioral) both target leakage from different angles — see `WALKTHROUGH.md` for why neither alone is sufficient |
| 3 | A self-registering check pattern | `src/audit_checks.py`: `CHECK_REGISTRY` + the `@check(...)` decorator — adding a new check requires no change to `audit_runner.py` or the frontend |
| 4 | CRISP-DM phase tagging on every check | Every `AuditCheck` carries a `crisp_dm_phase`, used to group the dashboard and confirm coverage spans multiple phases (`tests/test_api.py: test_checks_catalog_has_10_checks_across_multiple_phases`) |
| 5 | Live execution over static reporting | `audit_runner.py: run_audit()` trains a fresh model on every call — "audit platform," not "audit report" |
| 6 | Empirically verifying a threshold rather than guessing it | `AUDIT_REPORT.md` section 2 — the `suspiciously_high_r2` threshold was checked against 30 repeated runs of the unseeded flawed variant, and moved from 0.97 to 0.95 after finding a real edge case, not chosen once and left untested |
| 7 | Reliable tests for a system with an intentionally unseeded component | `tests/test_api.py` — the R2-dependent assertion uses a comfortable margin instead of strict equality, specifically because one of the 4 violations being audited is "unseeded," so the test itself has to tolerate that on purpose |
| 8 | API input validation via typed literals | `app.py`: `variant: Literal["clean", "flawed"]` — an invalid variant value is a clean Pydantic 422, not a runtime KeyError |
| 9 | Honest sourcing when a prompt can't be verified | `PROMPTS.md` — states plainly that the original prompt for this specific project couldn't be retrieved, rather than presenting a fabricated quote as verbatim |
| 10 | Documentation-as-artifact | This file, plus `PROMPTS.md`, `IMPLEMENTATION_PLANS.md`, `WALKTHROUGH.md`, `AUDIT_REPORT.md`, `README.md` |

## Explicitly not attempted here

Persisted audit history across runs, authentication/multi-user support,
and a recorded video walkthrough were out of scope — per your explicit
instruction to keep this "not fancy," with the main aim being a simple,
working frontend and backend, and per this assistant's genuine
capability limits (no video/audio generation).
