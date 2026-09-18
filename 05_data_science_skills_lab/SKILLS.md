# SKILLS.md — Skills Demonstrated / Applied in This Build

This project is unusual among this repo's projects: its whole purpose
*is* a skills catalog (`src/skills.py`), so this file has two parts —
the 16 data-science skills the app itself demonstrates (the product),
and the engineering practices used to build it (same format as the
other projects' SKILLS.md).

## Part A: the 16 data science skills the app demonstrates

See `IMPLEMENTATION_PLANS.md` §4 for the full table with CRISP-DM phases.
Each one is independently verifiable by running the app and clicking
"Execute skill," or directly via `POST /api/skills/{id}/execute`.

## Part B: engineering practices used to build the lab itself

| # | Skill | Where applied |
|---|---|---|
| 1 | CRISP-DM structuring, applied recursively | The lab's own architecture follows CRISP-DM (Data Understanding → Preparation → Modeling → Evaluation), AND each individual skill is tagged with which phase it belongs to (`src/skills.py: @skill(..., crisp_dm_phase=...)`) |
| 2 | Self-registering plugin pattern | `src/skills.py: SKILL_REGISTRY` + the `@skill` decorator — adding a new skill requires no change to `app.py` or the frontend, it just appears in the catalog |
| 3 | A display contract that decouples backend and frontend | `SkillResult.display_type` (`metrics`/`table`/`bar_chart`/`list`) — the frontend's `renderResult()` handles all 16 skills with zero skill-specific code |
| 4 | Consistent feature engineering across related skills | `src/skills.py: _prepare_model_features()` — every modeling/evaluation skill uses the identical feature set, so results are comparable across skills |
| 5 | Defensive error handling at the API boundary | `app.py: execute_skill()` — wraps skill execution in try/except, returns HTTP 404 for an unknown skill id and HTTP 500 (with message) rather than crashing on a broken skill |
| 6 | Verifiable synthetic data (ground-truth signal + known defects) | `data/generate_data.py` — survival genuinely depends on Sex/Pclass/Age (so modeling skills have signal), and missing values/duplicates are injected on purpose (so the corresponding Data Understanding skills have something to find) |
| 7 | Automated, registry-driven testing | `tests/test_api.py: test_execute_every_registered_skill` — loops over the live `/api/skills` catalog rather than hardcoding 16 test functions, so a newly added skill is automatically tested |
| 8 | Reproducibility | Fixed `RNG_SEED = 42` in `data/generate_data.py`, `RANDOM_SEED = 42` in `src/skills.py` for every train/test split and model fit |
| 9 | Documentation-as-artifact | This file, plus `PROMPTS.md`, `IMPLEMENTATION_PLANS.md`, `WALKTHROUGH.md`, `AUDIT_REPORT.md`, `README.md` |

## Explicitly not attempted here (see `IMPLEMENTATION_PLANS.md` §8)

Installing the two external GitHub skill packages named in the original
prompt, covering 5 separate Kaggle datasets, and a charting library for
the bar charts were all present in the source repo's larger vision but
were intentionally left out — per your explicit instruction to keep this
"not fancy," with the main aim being a simple, working frontend and
backend.
