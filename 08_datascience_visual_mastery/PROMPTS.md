# PROMPTS.md — Verbatim Prompt Catalog (Project 08: Data Science Visual Foundations)

This document records the prompt this project was originally generated from
(source: [`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples),
`PROMPTS.md`, Project 8), plus the prompts actually used to reproduce a
scoped-down, working version of it in this build.

## Original source prompt

> /teamwork-preview lets do another project - teach beginner data science
> students in an excellent way with deep intuition and rigorous math and
> visual intuition and live simulation on:
> 1. naive bayes
> 2. evaluation of model - confusion matrix, type 1 and type 2 errors,
>    roc-auc, cost matrix, tradeoff between precision and recall
> 3. differential calculus, derivatives and how they connect to gradient
>    descent
> 4. chain rule and how it connects to backpropagation
> include quizzes for each concept and interview prep questions. also
> create a github.io ready page for this project.

This is the one project so far in this scoped-down series whose core ask
— 4 specific teaching topics, each with live simulation + quiz + interview
prep — was reproduced essentially as specified, not narrowed. See below
for the couple of adjustments that were made.

## Reproduction prompts used in this build

1. (Earlier in this conversation) `Replicate the data science experiments in your favorite coding assistant — prompts are in the repo dlmastery/data_science_examples. Publish to GitHub, walk through in a YouTube video, link in README.` — clarified up front that GitHub push and YouTube upload aren't things this assistant can do; scope was narrowed to code + docs delivered as a zip.
2. `Don't make it very fancy. Create project 01_nyc_taxi_prediction, complete folder in a zip file. Use PROMPTS.md, IMPLEMENTATION_PLANS.md, WALKTHROUGH.md, SKILLS.md, AUDIT_REPORT.md.` (established the pattern all subsequent projects follow)
3. `I don't want it to be very fancy. The main aim is that a simple frontend and backend would work.` (reaffirmed for project 04, carried forward here)
4. `[DS Visual Foundations](.../08_datascience_visual_mastery)` (this build)

## What changed from the original spec, and why

| Original ask | This build | Reason |
|---|---|---|
| "a github.io ready page" (a static site, typically hosted via GitHub Pages) | A FastAPI backend + a plain HTML/JS frontend that calls it | Every simulation here is a REAL computation (a trained Naive Bayes classifier, live confusion-matrix recomputation, actual gradient descent, hand-verified backprop) — a github.io static page has no server to run Python/sklearn on, so it would need the same logic reimplemented in JavaScript, doubling the surface area to keep correct. A small backend keeps one source of truth for the math. If you want a purely static version later, the simulations could be ported to JS, but that's a larger, separate task than "not fancy." |
| Quizzes | Quizzes graded server-side (`POST /api/quiz/{id}/grade`) — correct answers are never sent to the browser before grading | A quiz whose answer key ships in the page source isn't really being "tested" — this is a small, deliberate improvement in the spirit of the ask, not scope creep. |
| "rigorous math" | Naive Bayes: real per-word log-likelihood-ratio explanation (the actual quantity the algorithm computes, not an approximation). Gradient descent: hand-derived analytic derivatives (not autograd). Backprop: hand-derived chain rule, verified against numerical (finite-difference) gradients to ~5-6 decimal places — see `AUDIT_REPORT.md`. | Directly honors "rigorous math," verified rather than asserted. |
| Publish to GitHub, YouTube walkthrough | Zip file for you to `git push` yourself; text-based `WALKTHROUGH.md` instead of a recorded video | No GitHub write credentials or video/audio recording & YouTube upload capability are available to this assistant |
