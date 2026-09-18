# WALKTHROUGH.md — Architectural Deep-Dive

> No video walkthrough is included with this build: recording and uploading
> a narrated video (e.g. to YouTube) isn't something this assistant can do —
> there's no video/audio generation or YouTube-upload capability available.
> This document is the substitute: a code-ordered tour you (or a screen
> recorder) can follow to produce one in a few minutes if you want it, plus
> a suggested shot list at the bottom.

## Reading order (this *is* the tour)

1. **`data/generate_data.py`** — where the data comes from. Read the
   module docstring first, then the survival-probability model: baseline
   + boosts for `Sex`, `Pclass`, and young `Age`, plus noise, then a
   Bernoulli draw — so classification skills have real signal, not random
   noise. Note the deliberate missing-value injection and the 8 injected
   duplicate rows near the bottom — both exist specifically so the
   corresponding skills have something to find.

2. **`src/data_prep.py`** — intentionally thin: `load_dataset()` does
   *not* clean anything. Read the docstring's explanation of why — several
   skills need the raw, uncleaned data to have something to demonstrate.

3. **`src/skills.py`** — the core of this project. Read in this order:
   - the `SkillResult` / `Skill` dataclasses and the `display_type`
     contract in the module docstring — this is what makes one generic
     frontend renderer work for all 16 skills
   - the `@skill(...)` decorator and `SKILL_REGISTRY` — skills
     self-register; `app.py` never has a manually-maintained list
   - pick 2-3 skill functions to read closely, e.g. `outlier_detection`
     (Tukey IQR bounds), `baseline_vs_model` (majority-class baseline vs.
     a real trained classifier — the comparison that makes "the model is
     actually learning something" checkable), and `feature_importance`
   - `_prepare_model_features()` — the one shared feature-prep helper
     every modeling/evaluation skill uses, so results are consistent
     across skills

4. **`app.py`** — FastAPI wiring: `/api/skills` returns catalog metadata
   only (fast, no computation); `/api/skills/{id}/execute` is where a
   skill actually runs, wrapped in try/except so a broken skill returns a
   clean HTTP 500 instead of crashing the server.

5. **`static/app.js`** — `loadSkills()` groups the catalog by
   `crisp_dm_phase` and renders a card per skill; `executeSkill()` POSTs
   to the execute endpoint; `renderResult()` is the generic renderer that
   dispatches on `display_type` to draw metric cards, a table, plain-CSS
   bar chart rows, or a list — this single function is what replaces the
   original "unfriendly raw JSON" the source repo's own follow-up prompt
   complained about.

## How a request flows end-to-end

```
page loads
  → static/app.js fetches /api/skills (catalog) and /api/dataset/summary
  ← 16 skill cards render, grouped into 4 CRISP-DM phase sections

user clicks "Execute skill" on e.g. "Confusion Matrix & Metrics"
  → static/app.js POSTs to /api/skills/confusion_matrix/execute
    → app.py looks up the skill in SKILL_REGISTRY, loads the dataset fresh
      → skills.py: confusion_matrix_skill() splits, trains, predicts, scores
    ← {display_type: "table", summary: "...", data: {columns, rows}}
  ← static/app.js renders an actual HTML table, not the raw JSON
```

## Suggested shot list (if you want to record your own walkthrough video)

1. `python data/generate_data.py` running, show the missing-value/duplicate/survival-rate printout
2. `uvicorn app:app --reload --port 8005` starting up
3. Browser: page load, show the 4 CRISP-DM phase sections with skill cards
4. Browser: click "Execute skill" on "Missing Value Summary" — show the table render
5. Browser: click "Baseline vs. Logistic Regression" — show the bar chart, point out the accuracy lift over baseline
6. Browser: click "Confusion Matrix & Metrics" — show precision/recall/F1 and the 2×2 table
7. Terminal: `python -m pytest tests/` passing, especially the test that loops over every skill
8. Code editor: 30 seconds each on the `SkillResult`/`display_type` contract in `skills.py`, and `renderResult()` in `app.js` — the two ends of the contract that make this work
