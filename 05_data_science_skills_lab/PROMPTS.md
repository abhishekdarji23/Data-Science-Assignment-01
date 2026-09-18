# PROMPTS.md — Verbatim Prompt Catalog (Project 05: Data Science Skills Mastery Lab)

This document records the prompts this project was originally generated
from (source: [`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples),
`PROMPTS.md`, Project 5), plus the prompts actually used to reproduce a
scoped-down, working version of it in this build.

## Original source prompt

> /teamwork-preview install param087 github agent-ml-skills and install
> nimrodfisher data-analytics-skills and demonstrate every skill on
> appropriate popular kaggle data set. also include crisp-dm steps in it.

**Original follow-up (from the source repo) — this is the important one:**

> /teamwork-preview the data science skills mastery lab - when i say
> execute skill live - it shows a unfriendly raw json output on screen.
> can you make that visual interactive and look like a proper dashboard?

That follow-up is effectively the spec this build was built to satisfy
directly: every skill result here is rendered as metric cards, a table,
a bar chart, or a list — never a raw JSON dump — from the very first
version (see `src/skills.py`'s `display_type` contract).

## Reproduction prompts used in this build

1. (Earlier in this conversation) `Replicate the data science experiments in your favorite coding assistant — prompts are in the repo dlmastery/data_science_examples. Publish to GitHub, walk through in a YouTube video, link in README.` — clarified up front that GitHub push and YouTube upload aren't things this assistant can do; scope was narrowed to code + docs delivered as a zip.
2. `Don't make it very fancy. Create project 01_nyc_taxi_prediction, complete folder in a zip file. Use PROMPTS.md, IMPLEMENTATION_PLANS.md, WALKTHROUGH.md, SKILLS.md, AUDIT_REPORT.md.` (established the pattern all subsequent projects follow)
3. `I don't want it to be very fancy. The main aim is that a simple frontend and backend would work.` (reaffirmed for project 04, carried forward here)
4. `05_data_science_skills_lab` (this build)

## What changed from the original spec, and why

| Original ask | This build | Reason |
|---|---|---|
| Install two external GitHub skill repos (`param087/agent-ml-skills`, `nimrodfisher/data-analytics-skills`), 54 skills across 5 Kaggle datasets | 16 self-contained skills, implemented directly in `src/skills.py`, on 1 synthetic dataset | This assistant cannot `pip install` or clone arbitrary GitHub skill packages into a running product for someone else to rely on, and 5 full datasets × dozens of skills is out of scope for "not fancy, simple frontend and backend." 16 skills across all 4 data-facing CRISP-DM phases is a genuine, representative sample. |
| Real Kaggle dataset | Physically-modeled **synthetic** Titanic-style dataset (`data/generate_data.py`), same schema as the real competition, with realistic missing values, duplicates, and a genuine (non-random) survival signal baked in | No internet access to Kaggle in this environment. `src/data_prep.py` will auto-prefer a real `data/titanic_real.csv` if you drop one in. |
| Elaborate "AutoResearch"-matched admin dashboard | Plain HTML/JS skill cards grouped by CRISP-DM phase, each with an "Execute skill" button | Explicitly requested: "I don't want it to be very fancy. The main aim is that a simple frontend and backend would work." |
| Publish to GitHub, YouTube walkthrough | Zip file for you to `git push` yourself; text-based `WALKTHROUGH.md` instead of a recorded video | No GitHub write credentials or video/audio recording & YouTube upload capability are available to this assistant |
