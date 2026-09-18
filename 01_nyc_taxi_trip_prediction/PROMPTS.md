# PROMPTS.md — Verbatim Prompt Catalog (Project 01: NYC Taxi Trip Duration Predictor)

This document records the prompt this project was originally generated from
(source: [`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples),
`PROMPTS.md`, Project 1), plus the prompts actually used to reproduce a
scoped-down, working version of it in this build.

## Original source prompt

> Now on to another project - You will do end2end a data science project of
> including data, training, deployment, crisp-dm framework, and an awesome
> front end. you can use Kaggle NYC taxi challenge. make sure that frontend
> includes interactive map and trip estimation.

**Original follow-ups (from the source repo):**
- `make the admin dashboard even better with more data scientist related details. prepare a research report with crisp-dm methodology for the same as well.`
- `please follow the autoresearch repo skills deligently for hill climbing. also match the dashboard details with research paper.`

## Reproduction prompts used in this build

1. `Replicate the data science experiments in your favorite coding assistant — prompts are in the repo dlmastery/data_science_examples. Publish to GitHub, walk through in a YouTube video, link in README.`
2. Follow-up scoping (after clarifying real constraints — no GitHub push access, no video/YouTube upload capability): `Don't make it very fancy. Create project 01_nyc_taxi_prediction, complete folder in a zip file. Use PROMPTS.md, IMPLEMENTATION_PLANS.md, WALKTHROUGH.md, SKILLS.md, AUDIT_REPORT.md.`

## What changed from the original spec, and why

| Original ask | This build | Reason |
|---|---|---|
| Real Kaggle NYC taxi dataset | Physically-modeled **synthetic** dataset (`data/generate_data.py`), same schema as the Kaggle competition | No internet access to Kaggle in this environment. `src/data_prep.py` will auto-prefer a real `data/nyc_taxi_train.csv` if you drop one in. |
| "Awesome" React admin dashboard, AutoResearch hill-climbing, published research paper | Plain HTML/CSS/JS frontend + a `/api/metrics` model-card endpoint | Explicitly scoped down ("don't make it very fancy") to something that reliably runs in one pass |
| Publish to GitHub, YouTube walkthrough | Zip file for you to `git push` yourself; text-based `WALKTHROUGH.md` instead of a recorded video | No GitHub write credentials or video/audio recording & YouTube upload capability are available to this assistant |
