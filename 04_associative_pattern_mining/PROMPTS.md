# PROMPTS.md — Verbatim Prompt Catalog (Project 04: Market Basket Pattern Mining)

This document records the prompt this project was originally generated from
(source: [`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples),
`PROMPTS.md`, Project 4), plus the prompts actually used to reproduce a
scoped-down, working version of it in this build.

## Original source prompt

> /teamwork-preview Now lets do another project - associative pattern
> mining using popular kaggle data set - make sure you follow crisp-dm
> framework and also include nice data science admin dashboard. you can
> research the papers and implement autoresearch to do hill climbing and
> match the dashboard details with research paper. include all details a
> data scientist and ai engineer will care.

## Reproduction prompts used in this build

1. (Earlier in this conversation) `Replicate the data science experiments in your favorite coding assistant — prompts are in the repo dlmastery/data_science_examples. Publish to GitHub, walk through in a YouTube video, link in README.` — clarified up front that GitHub push and YouTube upload aren't things this assistant can do; scope was narrowed to code + docs delivered as a zip.
2. `Don't make it very fancy. Create project 01_nyc_taxi_prediction, complete folder in a zip file. Use PROMPTS.md, IMPLEMENTATION_PLANS.md, WALKTHROUGH.md, SKILLS.md, AUDIT_REPORT.md.` (established the pattern all subsequent projects follow)
3. `03_customer_segmentation_clustering — next project`
4. `I don't want it to be very fancy. The main aim is that a simple frontend and backend would work. Now make project code for 04_associative_pattern_mining — next this project` (this build)

## What changed from the original spec, and why

| Original ask | This build | Reason |
|---|---|---|
| Real Kaggle market-basket dataset | Synthetic grocery-basket dataset (`data/generate_data.py`) with **explicitly baked-in association rules** (bread→butter, diapers→beer/wipes, chips→salsa, etc.) | No internet access to Kaggle in this environment. Baking in known patterns lets the audit verify Apriori actually *rediscovers* them (see `AUDIT_REPORT.md`), which a purely random basket generator couldn't support. |
| "AutoResearch hill-climbing," research-paper-matched admin dashboard | Apriori (mlxtend) with fixed, documented support/confidence/lift thresholds; a plain HTML/JS cart-builder + rules table + model card | Explicitly requested: "I don't want it to be very fancy. The main aim is that a simple frontend and backend would work." |
| Publish to GitHub, YouTube walkthrough | Zip file for you to `git push` yourself; text-based `WALKTHROUGH.md` instead of a recorded video | No GitHub write credentials or video/audio recording & YouTube upload capability are available to this assistant |
