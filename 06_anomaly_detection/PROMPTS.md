# PROMPTS.md — Verbatim Prompt Catalog (Project 06: Anomaly Detection)

This document records the prompt this project was originally generated from
(source: [`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples),
`PROMPTS.md`, Project 6), plus the prompts actually used to reproduce a
scoped-down, working version of it in this build.

## Original source prompt

> /teamwork-preview Now lets do another project - anomaly detection using
> popular kaggle data set and popular methods - make sure you follow
> crisp-dm framework and also include nice data science admin dashboard.
> you can research the papers and implement autoresearch to do hill
> climbing and match the dashboard details with research paper. include
> all details a data scientist and ai engineer will care.

## Reproduction prompts used in this build

1. (Earlier in this conversation) `Replicate the data science experiments in your favorite coding assistant — prompts are in the repo dlmastery/data_science_examples. Publish to GitHub, walk through in a YouTube video, link in README.` — clarified up front that GitHub push and YouTube upload aren't things this assistant can do; scope was narrowed to code + docs delivered as a zip.
2. `Don't make it very fancy. Create project 01_nyc_taxi_prediction, complete folder in a zip file. Use PROMPTS.md, IMPLEMENTATION_PLANS.md, WALKTHROUGH.md, SKILLS.md, AUDIT_REPORT.md.` (established the pattern all subsequent projects follow)
3. `I don't want it to be very fancy. The main aim is that a simple frontend and backend would work.` (reaffirmed for project 04, carried forward here)
4. `06_anomaly_detection` (this build)

## What changed from the original spec, and why

| Original ask | This build | Reason |
|---|---|---|
| Real Kaggle anomaly-detection dataset | Physically-modeled **synthetic** transaction dataset (`data/generate_data.py`) with interpretable features (amount, hour, account age, transaction velocity, distance from home) and a genuinely separable, multivariate anomaly pattern | No internet access to Kaggle in this environment. The real Kaggle "Credit Card Fraud Detection" dataset also anonymizes its features as PCA components (V1-V28), which would make a "here's why this was flagged" dashboard meaningless — interpretable synthetic features serve the actual goal better here. `src/data_prep.py` will auto-prefer a real `data/transactions_real.csv` if you drop one in (with the documented schema). |
| "Popular methods" (open-ended), "AutoResearch hill-climbing," research-paper-matched dashboard | 3 named, standard, popular unsupervised methods (Isolation Forest, Local Outlier Factor, Z-score baseline), compared head-to-head by PR-AUC/precision/recall — a plain HTML/JS dashboard, no research paper | Explicitly requested: "I don't want it to be very fancy. The main aim is that a simple frontend and backend would work." Isolation Forest and LOF are the two most widely used unsupervised anomaly detectors in practice; z-score is the standard statistical baseline they're normally compared against. |
| Publish to GitHub, YouTube walkthrough | Zip file for you to `git push` yourself; text-based `WALKTHROUGH.md` instead of a recorded video | No GitHub write credentials or video/audio recording & YouTube upload capability are available to this assistant |

## An honest result worth flagging up front

Local Outlier Factor performed poorly on this dataset (PR-AUC ≈ 0.10,
barely above the ~0.03 random baseline) while Isolation Forest and the
z-score baseline both performed very well (PR-AUC ≈ 0.99 and 0.95). This
was NOT smoothed over or hidden — see `AUDIT_REPORT.md` §3 for why this
happens (LOF assumes anomalies are isolated points; this dataset's
anomalies form their own dense cluster, which is a known weakness of
density-based methods). A real "popular methods comparison" should show
this kind of result, not just report whichever number sounds best.
