# WALKTHROUGH.md — Architectural Deep-Dive

> No video walkthrough is included with this build: recording and uploading
> a narrated video (e.g. to YouTube) isn't something this assistant can do —
> there's no video/audio generation or YouTube-upload capability available.
> This document is the substitute: a code-ordered tour you (or a screen
> recorder) can follow to produce one in a few minutes if you want it, plus
> a suggested shot list at the bottom.

## Reading order (this *is* the tour)

1. **`data/generate_data.py`** — read the module docstring first, then
   the generation loop: `mean_reversion = -MEAN_REVERSION_STRENGTH *
   prev_return` is the one line responsible for making ETS/ARIMA able
   to extract any signal at all; everything else (`DAILY_DRIFT`,
   `vol[t]`, `DOW_EFFECT`) is realism/noise around it.

2. **`src/data_prep.py`** — `build_lag_features()`: note every feature
   uses `.shift(1)` or `.shift(lag)` — never the current row's `close`
   itself. This is what makes the gradient boosting model's test-set
   predictions genuine 1-step-ahead forecasts rather than leaked ones.

3. **`src/train.py`** — the core of this project. Read the module
   docstring's explanation of "honest 1-step-ahead evaluation" first,
   then walk through the 5 method blocks in order:
   - `pred_naive = close[split_idx - 1 : n - 1]` — a one-line baseline
   - the moving-average list comprehension
   - the ETS refit loop — note `history.append(close[t])` happens
     AFTER computing that step's forecast, so the forecast never sees
     `close[t]` before predicting it
   - the ARIMA AIC grid search (on TRAIN data only) followed by its own
     refit loop, same discipline
   - the GBM block — trained once, since its features already encode
     the "no future peeking" rule
   - the printed leaderboard — this is where you see the honest result

4. **`src/predict.py`** — `forecast_next_day()`: the live-serving
   equivalent of the same 5 methods, using the full dataset as history.
   Compare its ETS/ARIMA calls to `train.py`'s — same models, just fit
   once on all available data instead of once per backtest step.

5. **`app.py`** — thin FastAPI wiring; no request bodies needed for
   `/api/forecast` since it always forecasts from the latest data.

6. **`static/app.js`** — `drawChart()` is a plain Canvas 2D line chart
   (no library) plotting actual vs. naive-forecast over the test period;
   the leaderboard table sorts by MAE and tags each non-naive method
   with a "beats naive" / "does not beat naive" badge, so the honest
   result is visible at a glance, not buried in a metrics table.

## How a request flows end-to-end

```
user clicks "Get tomorrow's forecast"
  → static/app.js POSTs to /api/forecast
    → app.py calls predict.py: forecast_next_day()
      → loads the full price history
      → computes naive, moving average directly
      → refits ETS and ARIMA once each on full history
      → builds one feature row (lag/rolling features from the tail of
        history) and scores it with the persisted GBM model
    ← {as_of_date, last_close, forecasts: {5 methods}}
  ← static/app.js renders 5 cards with price + delta from last close
```

## Suggested shot list (if you want to record your own walkthrough video)

1. `python data/generate_data.py` running, show the printed mean/vol/autocorrelation stats
2. `python src/train.py` running (~30s) — show the leaderboard printout, point out moving average and gradient boosting losing to naive
3. `uvicorn app:app --reload --port 8015` starting up
4. Browser: point at the leaderboard, the "beats naive" / "does not beat naive" badges
5. Browser: point at the actual-vs-naive chart — note how closely the red naive line tracks the black actual line (visually demonstrating why it's hard to beat)
6. Browser: click "get tomorrow's forecast," show the 5 cards
7. Terminal: `python -m pytest tests/` passing, especially the "naive is not a strawman" test
8. Code editor: 30 seconds on the ETS/ARIMA refit loops in `train.py` — the actual mechanism behind "honest 1-step-ahead"
