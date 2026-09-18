# PROMPTS.md — Verbatim Prompt Catalog (Project 07: AutoML Stacking)

This document records the prompt this project was originally generated from
(source: [`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples),
`PROMPTS.md`, Project 7), plus the prompts actually used to reproduce a
scoped-down, working version of it in this build.

## Original source prompt

> /teamwork-preview Now lets do another project - illustrate automl with
> autogluon on various data science tasks - make sure you follow crisp-dm
> framework and also include nice data science admin dashboard. you can
> research the papers and implement autoresearch to do hill climbing and
> match the dashboard details with research paper. include all details a
> data scientist and ai engineer will care.

## Reproduction prompts used in this build

1. (Earlier in this conversation) `Replicate the data science experiments in your favorite coding assistant — prompts are in the repo dlmastery/data_science_examples. Publish to GitHub, walk through in a YouTube video, link in README.` — clarified up front that GitHub push and YouTube upload aren't things this assistant can do; scope was narrowed to code + docs delivered as a zip.
2. `Don't make it very fancy. Create project 01_nyc_taxi_prediction, complete folder in a zip file. Use PROMPTS.md, IMPLEMENTATION_PLANS.md, WALKTHROUGH.md, SKILLS.md, AUDIT_REPORT.md.` (established the pattern all subsequent projects follow)
3. `I don't want it to be very fancy. The main aim is that a simple frontend and backend would work.` (reaffirmed for project 04, carried forward here)
4. `[AutoML AutoGluon Stacking](.../07_automl_autogluon)` (this build)

## The one deliberate substitution in this build, and why

**The actual AutoGluon package was NOT used or installed.** AutoGluon's
dependency footprint includes PyTorch, LightGBM, CatBoost, and Ray —
often several gigabytes of installs, with a build/import time that
directly conflicts with "keep it simple" and "not fancy." Rather than
either failing partway through a heavy install or silently skipping the
AutoML angle entirely, this build reimplements the specific technique the
project's title names — **stacking** — using plain scikit-learn:

1. Train a small "model zoo" (logistic regression, random forest, gradient
   boosting, k-NN, decision tree) — the same spirit as AutoGluon's default
   model set (linear / tree ensembles / distance-based).
2. Generate out-of-fold (OOF) predictions for each via k-fold
   cross-validation (no leakage).
3. Train a meta-learner (logistic regression) on the OOF predictions —
   this is the actual "stacking" AutoGluon itself is famous for
   (Caruana-style ensembling), just implemented directly rather than via
   the AutoGluon API.
4. Compare every base model AND the stacked ensemble on a held-out test
   set via a leaderboard, exactly like AutoGluon's own `leaderboard()`
   output.

## What else changed from the original spec, and why

| Original ask | This build | Reason |
|---|---|---|
| "various data science tasks" (implies multiple task types: regression, classification, etc.) | One task: binary classification (customer churn) | Scoped down per "not fancy, simple frontend and backend." One well-executed task beats a shallow multi-task demo. |
| Real Kaggle dataset | Physically-modeled **synthetic** telco-churn-style dataset (`data/generate_data.py`), matching the schema of the classic Kaggle "Telco Customer Churn" dataset — itself a common real AutoGluon tutorial dataset | No internet access to Kaggle in this environment. `src/data_prep.py` will auto-prefer a real `data/churn_real.csv` if you drop one in. |
| "AutoResearch hill-climbing," research-paper-matched admin dashboard | A plain leaderboard table + bar chart, a live scoring form showing every base model's individual opinion alongside the final stacked prediction | Explicitly requested: "I don't want it to be very fancy. The main aim is that a simple frontend and backend would work." |
| Publish to GitHub, YouTube walkthrough | Zip file for you to `git push` yourself; text-based `WALKTHROUGH.md` instead of a recorded video | No GitHub write credentials or video/audio recording & YouTube upload capability are available to this assistant |

## An honest result worth flagging up front

The stacked ensemble beats the best individual base model by a small
margin (ROC-AUC ≈0.7385 vs. ≈0.7382 for logistic regression alone) — not
a dramatic win. That's realistic: stacking typically gives a modest lift
over the best single model, not a transformative one, especially with a
small model zoo and moderate-sized data. See `AUDIT_REPORT.md` for the
full comparison and why this is expected, not a sign anything is broken.
