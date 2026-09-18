# PROMPTS.md — Verbatim Prompt Catalog (Project 03: Customer Segmentation Clustering)

This document records the prompt this project was originally generated from
(source: [`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples),
`PROMPTS.md`, Project 3), plus the prompts actually used to reproduce a
scoped-down, working version of it in this build.

## Original source prompt

> /teamwork-preview Now lets do another project - clustering using popular
> kaggle data set - make sure you follow crisp-dm framework and also
> include nice data science admin dashboard. you can research the papers
> and implement autoresearch to do hill climbing and match the dashboard
> details with research paper. include all details a data scientist and
> ai engineer will care.

## Reproduction prompts used in this build

1. (Earlier in this conversation) `Replicate the data science experiments in your favorite coding assistant — prompts are in the repo dlmastery/data_science_examples. Publish to GitHub, walk through in a YouTube video, link in README.` — clarified up front that GitHub push and YouTube upload aren't things this assistant can do; scope was narrowed to code + docs delivered as a zip.
2. `Don't make it very fancy. Create project 01_nyc_taxi_prediction, complete folder in a zip file. Use PROMPTS.md, IMPLEMENTATION_PLANS.md, WALKTHROUGH.md, SKILLS.md, AUDIT_REPORT.md.` (established the pattern this project follows)
3. `03_customer_segmentation_clustering — next project` (this build, following the same pattern)

## What changed from the original spec, and why

| Original ask | This build | Reason |
|---|---|---|
| Real Kaggle clustering dataset | Physically-modeled **synthetic** dataset (`data/generate_data.py`), same schema as the classic "Mall Customer Segmentation" Kaggle dataset | No internet access to Kaggle in this environment. `src/data_prep.py` will auto-prefer a real `data/Mall_Customers.csv` if you drop one in. |
| "AutoResearch hill-climbing," matched to a research paper, elaborate admin dashboard | KMeans with k chosen by silhouette score across k=2..8, a plain HTML/JS scatter-plot + segment-profile dashboard | Explicitly scoped down ("not fancy," consistent with project 01) to something that reliably runs in one pass |
| Publish to GitHub, YouTube walkthrough | Zip file for you to `git push` yourself; text-based `WALKTHROUGH.md` instead of a recorded video | No GitHub write credentials or video/audio recording & YouTube upload capability are available to this assistant |
