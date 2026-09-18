# WALKTHROUGH.md — Architectural Deep-Dive

> No video walkthrough is included with this build: recording and uploading
> a narrated video (e.g. to YouTube) isn't something this assistant can do —
> there's no video/audio generation or YouTube-upload capability available.
> This document is the substitute: a code-ordered tour you (or a screen
> recorder) can follow to produce one in a few minutes if you want it, plus
> a suggested shot list at the bottom (this also stands in for the
> "video script" mentioned in the third-party description of this
> project — see `PROMPTS.md`).

## Reading order (this *is* the tour)

1. **`data/generate_data.py`** — a compact NYC-taxi-style generator, with
   12 deliberately injected duplicate rows for the duplicate-detection
   check to find. Same spirit as project 01's generator, kept
   self-contained to this project.

2. **`src/pipeline.py`** — the audit SUBJECT, not the audit itself. Read
   the module docstring, then compare the `if variant == "flawed":` and
   `else:` branches side by side in `run_pipeline()` — this is where all
   4 deliberate violations live: the leaky `duration_bucket` column, the
   unseeded `train_test_split(..., shuffle=True)`, skipping
   `drop_duplicates()`, and the unseeded `GradientBoostingRegressor()`.

3. **`src/audit_checks.py`** — 10 independent check functions, each
   registered via the `@check(...)` decorator with its CRISP-DM phase.
   Read `no_target_leakage()` and `suspiciously_high_r2()` together —
   they check the SAME underlying problem (leakage) from two angles: one
   structural (is the leaky column in the feature list), one behavioral
   (does the resulting R2 look too good to be true). Both fire on the
   flawed variant; only the structural one would catch leakage a
   behavioral-only audit might miss (e.g. mild leakage that doesn't push
   R2 above any fixed threshold), and vice versa.

4. **`src/audit_runner.py`** — `run_audit()`: runs the pipeline, runs
   every check, groups results by phase, computes the compliance score.
   `run_comparison()` just calls `run_audit()` twice — the "comparison
   report."

5. **`app.py`** — thin FastAPI wiring; `variant` is a `Literal["clean",
   "flawed"]` query parameter, so an invalid value is rejected by
   Pydantic before it reaches the pipeline.

6. **`static/app.js`** — `runAudit()` and `runCompare()` both call
   `scoreHtml()` and `checksByPhaseHtml()` — the same two render
   functions used for both single-run and comparison views, so the
   dashboard and the comparison panel never visually drift apart.

## How a request flows end-to-end

```
user clicks "Run audit: flawed pipeline"
  → static/app.js POSTs to /api/audit/run?variant=flawed
    → app.py validates variant via Literal["clean","flawed"]
      → audit_runner.py: run_audit("flawed")
        → pipeline.py: run_pipeline("flawed") -- trains a FRESH model
          with all 4 violations active
        → audit_checks.py: run_all_checks() -- 10 checks run against
          the artifacts just produced
      ← {compliance_score: 50.0, checks: [...], ...}
    ← JSON response
  ← static/app.js renders the score circle + 10 check cards, 5 red
```

## Suggested shot list (if you want to record your own walkthrough video)

1. `python data/generate_data.py` running, show the duplicate-count printout
2. `uvicorn app:app --reload --port 8013` starting up
3. Browser: click "Run audit: clean pipeline" — show the green 100% score and all 10 checks passing
4. Browser: click "Run audit: flawed pipeline" — show the score drop to 50%, point out which 5 checks turned red and why
5. Browser: point specifically at the R2 numbers — the flawed model's R2 is HIGHER, not lower, and explain why that's itself suspicious
6. Browser: click "Compare both, side by side" — show the two columns
7. Terminal: `python -m pytest tests/` passing, especially the test asserting the flawed pipeline's R2 is misleadingly inflated
8. Code editor: 30 seconds on `pipeline.py`'s `if variant == "flawed":` branch — the actual violations, in one place
