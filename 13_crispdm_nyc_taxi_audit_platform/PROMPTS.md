# PROMPTS.md — Prompt Catalog (Project 13: CRISP-DM NYC Taxi Audit Platform)

## Important note on this project's source prompt

Unlike every other project in this series, **the exact original prompt
for Project 13 could not be retrieved.** GitHub's access restrictions
blocked direct retrieval of the source repo's `PROMPTS.md` file for this
specific project at build time. The only information available was a
third-party description (from a fork's own README) calling it an
*"Integrated CRISP-DM/taxi audit platform"* with a *"Prompt record,
Reproduction log, Comparison report, Video script."*

Rather than fabricate a plausible-sounding verbatim quote, this file
states that plainly. Everything below is this build's own interpretation
of that short description, applied to the same "not fancy, simple
frontend and backend" standard used throughout this series.

## Reproduction prompts used in this build

1. (Earlier in this conversation) `Replicate the data science experiments in your favorite coding assistant — prompts are in the repo dlmastery/data_science_examples. Publish to GitHub, walk through in a YouTube video, link in README.` — established that GitHub push and YouTube upload aren't things this assistant can do.
2. `Don't make it very fancy. Create project 01_nyc_taxi_prediction, complete folder in a zip file. Use PROMPTS.md, IMPLEMENTATION_PLANS.md, WALKTHROUGH.md, SKILLS.md, AUDIT_REPORT.md.` — established the pattern all subsequent projects follow.
3. `I don't want it to be very fancy. The main aim is that a simple frontend and backend would work.` — reaffirmed and carried forward.
4. `[13_crispdm_nyc_taxi_audit_platform](.../13_crispdm_nyc_taxi_audit_platform)` then `Don't continue with 9, continue with 13.` (this build; project 09/FlowForge DAG Engine was left mid-build per this instruction)

## Interpretation of "integrated CRISP-DM/taxi audit platform"

Rather than a static markdown audit report (as project 01's
`AUDIT_REPORT.md` is), this build makes the audit itself a **live,
running platform**: every check in `src/audit_checks.py` executes
against a freshly-trained model on every request, not a cached or
hand-written verdict. The "comparison report" the third-party
description mentions is implemented as a genuine A/B: the same audit
checks are run against both a `clean` pipeline (correct practice
throughout) and a deliberately `flawed` one (4 real, specific
violations), so the platform can be shown catching real problems it
would otherwise only ever be demonstrated against passing data.

## What's deliberately scoped down, and why

| Aspect | This build | Reason |
|---|---|---|
| Data | Synthetic NYC-taxi-style data (`data/generate_data.py`), self-contained to this project | No internet access to Kaggle in this environment |
| Scope | 10 audit checks across 4 CRISP-DM phases, 1 subject pipeline | "Not fancy, simple frontend and backend" |
| "Video script" | Not produced — see `WALKTHROUGH.md`'s shot list instead | No video/audio generation or YouTube upload capability is available to this assistant |
| Publish to GitHub | Zip file for you to `git push` yourself | No GitHub write credentials are available to this assistant |
