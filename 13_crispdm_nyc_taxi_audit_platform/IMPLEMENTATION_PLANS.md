# IMPLEMENTATION_PLANS.md — Project 13: CRISP-DM NYC Taxi Audit Platform

## 1. Objective

Provide a live, executable audit of a taxi trip-duration ML pipeline
against CRISP-DM/methodology best practices — and prove the audit
actually catches real violations, not just that it runs.

## 2. Architecture

```
┌────────────────────┐   POST /api/audit/run?variant=clean|flawed  ┌──────────────────────┐
│  Browser (dashboard, │ ──────────────────────────────────────▶ │   FastAPI app.py     │
│  score + checks by   │ ◀────────────────────────────────────── │   (uvicorn)          │
│  CRISP-DM phase)     │   POST /api/audit/compare                  └──────────┬───────────┘
└────────────────────┘                                                        │ calls
                                                                     ┌──────────▼───────────┐
                                                                     │ src/audit_runner.py    │
                                                                     └──────────┬───────────┘
                                                              runs────┴────runs
                                                    ┌──────────▼───────────┐  ┌──────────▼───────────┐
                                                    │ src/pipeline.py        │  │ src/audit_checks.py    │
                                                    │ (the SUBJECT: clean or │  │ (10 independent checks │
                                                    │  flawed variant)       │  │  against the subject)  │
                                                    └──────────▲───────────┘  └───────────────────────┘
                                                               │ reads
                                                    ┌──────────┴───────────┐
                                                    │ data/*.csv             │
                                                    └───────────────────────┘
```

Every audit run trains a fresh model (fast at this data size) rather
than auditing a pre-trained, persisted one — this is what "live audit
platform" means here, matching the pattern the source description
implied (as opposed to a static report checked once and filed away).

## 3. The subject pipeline (`src/pipeline.py`) — two variants

- **`clean`**: duplicates removed before training, no leaky features,
  chronological 80/20 split, seeded model.
- **`flawed`**: 4 deliberate, specific violations —
  1. a feature (`duration_bucket`) derived directly from the target
  2. a random shuffle split instead of time-based, on genuinely
     time-series data
  3. duplicate rows left in the training data
  4. no random seed set (model and split both unseeded)

The flawed variant exists so the audit's checks can be run against a
subject *known* to violate each rule — proof the checks work, not just
that they execute without error.

## 4. The audit checks (`src/audit_checks.py`) — 10 checks, 4 CRISP-DM phases

| Phase | Checks |
|---|---|
| Data Understanding | distance/duration correlation sanity |
| Data Preparation | no target leakage, duplicates removed, no missing values, no duplicate feature names |
| Modeling | time-based split, reproducibility (seeded) |
| Evaluation | model beats baseline, R2 not suspiciously close to perfect |
| Deployment | live predictions are deterministic |

Each check is a small, independent function taking the pipeline's actual
artifacts (trained model, feature list, split method, data) and
returning `{passed, detail, evidence}` — nothing is a canned verdict.

## 5. Verified result (see `AUDIT_REPORT.md` for the full investigation)

- **Clean pipeline**: 10/10 checks pass, 100% compliance.
- **Flawed pipeline**: 5/10 checks fail — all 4 deliberate violations,
  plus "suspiciously high R2" firing as a corroborating signal (the
  leaked feature makes the model look BETTER, not worse — R2≈0.97 vs.
  the clean model's R2≈0.89 — a real, worth-highlighting property of
  leakage: it doesn't just violate methodology, it produces misleadingly
  optimistic metrics).

## 6. API contract (`app.py`)

| endpoint | method | purpose |
|---|---|---|
| `/` | GET | serves the frontend |
| `/api/health` | GET | liveness check |
| `/api/audit/checks` | GET | catalog of the 10 checks (metadata only) |
| `/api/audit/run?variant=clean\|flawed` | POST | runs the full pipeline + all checks, returns the compliance report |
| `/api/audit/compare` | POST | runs both variants, returns them side by side |

## 7. Frontend

Plain HTML/CSS/JS (no build step): a compliance-score circle, a pipeline
summary table, checks grouped by CRISP-DM phase with pass/fail cards, and
a side-by-side clean-vs-flawed comparison view.

## 8. Verification performed

- Direct script run of `run_audit()` against both variants — confirmed
  clean=100%, flawed=50%, with the failing checks matching exactly the 4
  deliberate violations plus the R2 corroborating signal
- Confirmed via 30 repeated runs that the flawed pipeline's R2 (unseeded
  by design) stays safely above the "suspicious" threshold every time
  (observed range 0.967-0.978 vs. a 0.95 threshold) -- this was checked
  empirically, not assumed, after an early threshold choice (0.97) was
  found to occasionally flip on a borderline run
- `uvicorn app:app` + `curl` against every endpoint -- correct live results
- `python -m pytest tests/` -- 7/7 passing, stable across 5 repeated runs
  (verifying no flakiness from the unseeded flawed variant)

## 9. Explicit non-goals

- No persisted audit history / trend-over-time view
- No authentication or multi-user support
- No containerization/CI -- this is a local, single-process demo
- No recorded video (see `PROMPTS.md` and `WALKTHROUGH.md`)
