# IMPLEMENTATION_PLANS.md — Project 05: Data Science Skills Mastery Lab

## 1. Objective

Provide a catalog of individually-executable data science "skills" —
missing value analysis, outlier detection, feature scaling, model
training, cross-validation, etc. — each runnable live against a real
dataset, with results rendered as an actual dashboard (metric cards,
tables, bar charts) rather than raw JSON. This directly answers the
source repo's own follow-up complaint about raw JSON output (see
`PROMPTS.md`).

## 2. Architecture

```
┌────────────────────┐   GET /api/skills, /api/dataset/summary   ┌──────────────────────┐
│  Browser (skill      │ ─────────────────────────────────────▶ │   FastAPI app.py     │
│  cards by CRISP-DM   │ ◀───────────────────────────────────── │   (uvicorn)          │
│  phase)              │   POST /api/skills/{id}/execute          └──────────┬───────────┘
└────────────────────┘                                                       │ calls
                                                                    ┌──────────▼───────────┐
                                                                    │ src/skills.py         │
                                                                    │  SKILL_REGISTRY        │
                                                                    │  (16 functions)        │
                                                                    └──────────▲───────────┘
                                                                               │ reads (live, every call)
                                                                    ┌──────────┴───────────┐
                                                                    │ src/data_prep.py      │
                                                                    │  load_dataset()       │
                                                                    └──────────▲───────────┘
                                                                               │
                                                                    ┌──────────┴───────────┐
                                                                    │ data/*.csv             │
                                                                    │ (synthetic or real)    │
                                                                    └───────────────────────┘
```

Unlike projects 01/03/04, there is **no persisted model artifact and no
separate `train.py`**. Every skill (including the modeling and
evaluation ones) is computed fresh on every API call — that's the whole
premise of a "live" skills lab: clicking "Execute" actually re-runs real
pandas/scikit-learn code against the current dataset, not a cached
result. On this dataset size (~600 rows), every skill executes in well
under a second.

## 3. Data schema

Matches the real Kaggle Titanic competition's core columns, so a real
`train.csv` can be dropped in without code changes:

| column | type | notes |
|---|---|---|
| PassengerId | int | row id |
| Pclass | int (1/2/3) | ticket class |
| Sex | string | male/female |
| Age | float | ~20% missing, matching the real dataset |
| SibSp, Parch | int | family-size features |
| Fare | float | ticket fare |
| Embarked | string | S/C/Q, a handful missing |
| Survived | int (0/1) | **target** |

## 4. Skill registry (`src/skills.py`)

Each skill is a `df -> SkillResult` function registered via the `@skill`
decorator with `id`, `title`, `crisp_dm_phase`, and `description`
metadata. A `SkillResult` always carries a `display_type` (`metrics` |
`table` | `bar_chart` | `list`) plus a plain-English `summary` sentence
— this contract is what lets one generic frontend renderer
(`static/app.js: renderResult()`) display all 16 skills correctly
without any skill-specific frontend code.

**16 skills across 4 CRISP-DM phases:**

| Phase | Skills |
|---|---|
| Data Understanding (8) | dataset_overview, missing_values, duplicate_rows, descriptive_stats, outlier_detection (Tukey IQR), categorical_counts, correlation_matrix, class_balance |
| Data Preparation (3) | missing_imputation, categorical_encoding (one-hot), feature_scaling (StandardScaler) |
| Modeling (2) | train_test_split (stratified 80/20), baseline_vs_model (majority-class baseline vs. logistic regression) |
| Evaluation (3) | cross_validation (5-fold), confusion_matrix (+ precision/recall/F1), feature_importance (logistic regression coefficients) |

All modeling/evaluation skills share one feature-preparation helper
(`_prepare_model_features()`) so every skill that trains a model uses
the identical feature set — median-imputed numerics, one-hot categoricals,
standardized — avoiding subtly-different results between skills.

## 5. API contract (`app.py`)

| endpoint | method | purpose |
|---|---|---|
| `/` | GET | serves the frontend |
| `/api/health` | GET | liveness check |
| `/api/skills` | GET | catalog metadata for all 16 skills (no execution) |
| `/api/dataset/summary` | GET | row/column counts for the page header |
| `/api/skills/{skill_id}/execute` | POST | runs that skill live, returns `{skill_id, title, crisp_dm_phase, summary, display_type, data}` |

Unknown `skill_id` → HTTP 404. A skill that raises internally → HTTP 500
with the exception message (defensive; no skill currently does this on
the shipped dataset — see `AUDIT_REPORT.md`).

## 6. Frontend

Plain HTML/CSS/JS (no build step, no charting library — bar charts are
plain `<div>` width bars, which is enough for this use case and avoids
a CDN dependency). Skill cards are grouped into 4 sections by CRISP-DM
phase; each has an "Execute skill" button that calls the API and renders
the response via the shared `renderResult()` function, dispatching on
`display_type`.

## 7. Verification performed

- A standalone script ran every registered skill directly against the
  dataset (no HTTP layer) and printed each result's summary — confirmed
  all 16 execute without error and return sensible, dataset-consistent
  values (e.g. `Sex_male` came out as the top feature importance, matching
  the sex-driven survival boost baked into `generate_data.py`).
- `uvicorn app:app` + a script that calls `/api/skills` then POSTs to
  `/api/skills/{id}/execute` for every returned skill — same 16/16 pass,
  through the real HTTP layer this time.
- `python -m pytest tests/` — 7/7 automated tests pass, including one
  that loops over the live `/api/skills` catalog and executes every one
  through the API (so a newly-added skill is automatically covered).

## 8. Explicit non-goals (kept out of scope per "not fancy, simple frontend and backend")

- No installation of external GitHub skill packages (`param087/agent-ml-skills`, `nimrodfisher/data-analytics-skills`) — reimplemented a representative, self-contained subset instead
- Only 1 dataset (Titanic-style), not 5
- No charting library (Chart.js etc.) — plain CSS bar charts instead
- No persisted/cached skill results — every execution is live, by design
- No authentication, database, or multi-user admin dashboard
- No containerization/CI — this is a local, single-process demo
