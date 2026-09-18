# PROMPTS.md — Prompt Catalog (Project 15: SPY Time Series SOTA Forecasting)

## Note on this project's source prompt

As with projects 13 and 14, GitHub's access restrictions prevented
retrieving the exact verbatim prompt for this specific project from the
source repo. Earlier search results did surface one concrete detail:
the source repo describes a *"Project 15 SPY TimeSeries skill"* among
its 60 packaged agent skills, confirming the subject (SPY = the S&P 500
ETF) and the time-series/forecasting framing in the project's own
folder name. Beyond that, nothing here is presented as a verbatim quote.

## Reproduction prompts used in this build

1. (Earlier in this conversation) `Replicate the data science experiments in your favorite coding assistant — prompts are in the repo dlmastery/data_science_examples. Publish to GitHub, walk through in a YouTube video, link in README.` — established that GitHub push and YouTube upload aren't things this assistant can do.
2. `Don't make it very fancy. Create project 01_nyc_taxi_prediction, complete folder in a zip file. Use PROMPTS.md, IMPLEMENTATION_PLANS.md, WALKTHROUGH.md, SKILLS.md, AUDIT_REPORT.md.` — established the pattern all subsequent projects follow.
3. `I don't want it to be very fancy. The main aim is that a simple frontend and backend would work.` — reaffirmed and carried forward.
4. `[15_spy_timeseries_sota_forecasting](.../15_spy_timeseries_sota_forecasting)` (this build)

## The one deliberate substitution, and why (same reasoning as Projects 07 and 14)

**No deep-learning forecasting stack (LSTM/GRU/Temporal Fusion
Transformer) was used**, even though "SOTA" (state-of-the-art) in
time-series forecasting today usually implies one. Those all need
PyTorch or TensorFlow as a dependency — heavy installs that conflict
with "keep it simple." Instead, this build compares 5 real methods
spanning classical statistics through gradient-boosted ML:

1. **Naive persistence** — the standard, genuinely hard-to-beat baseline
   for near-random-walk financial series.
2. **Moving average** (5-day)
3. **Exponential smoothing** (statsmodels, Holt's damped linear trend)
4. **ARIMA** (statsmodels, order chosen via AIC grid search on training
   data only)
5. **Gradient boosting** (scikit-learn, on lag/rolling features)

This is a legitimate, real comparison of "SOTA-in-spirit" classical and
ML forecasting techniques, not a deep-learning benchmark suite — see the
honest result below for why that framing matters.

## What else changed from the original spec, and why

| Aspect | This build | Reason |
|---|---|---|
| Real SPY market data | Physically-modeled **synthetic** daily price series (`data/generate_data.py`), engineered to have realistic near-random-walk structure PLUS a small, genuine, checkable mean-reversion signal | No internet/market-data-API access in this environment. A pure random walk would make "which method wins" untestable; a signal too strong would be unrealistic for real markets — this build's signal strength was empirically tuned (see the module docstring and `AUDIT_REPORT.md`) to be small but reliably detectable. |
| "SOTA" (deep learning implied) | 5 classical/ML methods, no neural forecasting model | Explicitly requested: "I don't want it to be very fancy. The main aim is that a simple frontend and backend would work." |
| Publish to GitHub, YouTube walkthrough | Zip file for you to `git push` yourself; text-based `WALKTHROUGH.md` instead of a recorded video | No GitHub write credentials or video/audio recording & YouTube upload capability are available to this assistant |

## The honest headline result

On this build's data: **naive persistence (MAE $3.80) is beaten only
narrowly by exponential smoothing ($3.78) and ARIMA ($3.78)** — both by
well under 1%. **Moving average ($4.98) and gradient boosting ($3.89)
both do WORSE than naive.** This is not a disappointing result to hide —
it's the well-documented, real finding in financial forecasting: daily
price levels are close enough to a random walk that naive persistence
is extremely hard to beat, and a "fancier" ML method doesn't
automatically win just because it's more sophisticated. See
`AUDIT_REPORT.md` for the full verification, including confirming this
data's mean-reversion signal is real (not zero) and that the small edge
ETS/ARIMA get is attributable to detecting it, not noise.
