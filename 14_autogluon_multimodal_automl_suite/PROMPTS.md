# PROMPTS.md — Prompt Catalog (Project 14: Multimodal AutoML Suite)

## Note on this project's source prompt

As with project 13, GitHub's access restrictions prevented retrieving
the exact verbatim prompt for this specific project from the source
repo. The project name itself
(`14_autogluon_multimodal_automl_suite`) is a strong, specific signal of
intent — AutoGluon's Multimodal offering, applied to an AutoML-style
suite — and this build follows that literally, with one deliberate
substitution explained below. Nothing here is presented as a verbatim
quote.

## Reproduction prompts used in this build

1. (Earlier in this conversation) `Replicate the data science experiments in your favorite coding assistant — prompts are in the repo dlmastery/data_science_examples. Publish to GitHub, walk through in a YouTube video, link in README.` — established that GitHub push and YouTube upload aren't things this assistant can do.
2. `Don't make it very fancy. Create project 01_nyc_taxi_prediction, complete folder in a zip file. Use PROMPTS.md, IMPLEMENTATION_PLANS.md, WALKTHROUGH.md, SKILLS.md, AUDIT_REPORT.md.` — established the pattern all subsequent projects follow.
3. `I don't want it to be very fancy. The main aim is that a simple frontend and backend would work.` — reaffirmed and carried forward.
4. `[14_autogluon_multimodal_automl_suite](.../14_autogluon_multimodal_automl_suite)` (this build)

## The one deliberate substitution, and why (same reasoning as Project 07)

**The actual AutoGluon Multimodal package was NOT used.** Real AutoGluon
Multimodal depends on PyTorch, HuggingFace Transformers, and timm (for
image models) — a very heavy install (often several GB, with GPU-class
build times) that directly conflicts with "keep it simple." Rather than
skip the multimodal angle entirely, this build reimplements the actual
underlying technique — **feature-level fusion of heterogeneous
modalities into one model** — directly with scikit-learn:

1. A tabular feature set (bedrooms, bathrooms, accommodates, room_type,
   neighborhood) — standard structured-data encoding.
2. A text feature set (TF-IDF of the listing description).
3. A fusion model trained on BOTH feature sets concatenated into one
   matrix — the literal definition of feature-level multimodal fusion,
   just without a deep-learning backbone doing the text encoding.

All 3 models use the identical algorithm (Ridge regression) so the
leaderboard isolates the effect of WHICH FEATURES each model sees,
rather than confounding it with a difference in algorithm.

## What else changed from the original spec, and why

| Original ask (inferred from the project name) | This build | Reason |
|---|---|---|
| Multiple modalities, possibly including images | 2 modalities: tabular + text | Image modeling adds a third heavy dependency (a vision backbone) for a "not fancy" build; tabular+text is the most common and clearly demonstrable multimodal pairing, and is itself one of real AutoGluon Multimodal's own tutorial datasets (Airbnb-style listings) |
| Real dataset | Physically-modeled **synthetic** listings dataset (`data/generate_data.py`), engineered so price genuinely depends on BOTH modalities (see the module docstring) | No internet access to Kaggle in this environment. Critically, this also makes "fusion beats either modality alone" a checkable claim rather than an assumption — see `AUDIT_REPORT.md`. |
| "AutoML suite" (implies broader automation) | 3 fixed models compared on one leaderboard, no hyperparameter search | Explicitly requested: "I don't want it to be very fancy. The main aim is that a simple frontend and backend would work." |
| Publish to GitHub, YouTube walkthrough | Zip file for you to `git push` yourself; text-based `WALKTHROUGH.md` instead of a recorded video | No GitHub write credentials or video/audio recording & YouTube upload capability are available to this assistant |

## The headline result

Fusion (R2≈0.955) dramatically outperforms both tabular-only (R2≈0.616)
and text-only (R2≈0.252) on held-out data. More concretely: two listings
with IDENTICAL tabular fields but different description text get the
EXACT SAME tabular-only prediction (it's structurally blind to the
text) but different fusion predictions — a $132 gap in one verified
example, entirely attributable to text the tabular-only model never
saw. See `AUDIT_REPORT.md` for the full verification.
