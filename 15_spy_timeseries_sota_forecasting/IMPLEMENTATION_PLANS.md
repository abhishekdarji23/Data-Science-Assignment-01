# IMPLEMENTATION_PLANS.md — Project 15: SPY Time Series Forecasting

## 1. Objective

Compare 5 forecasting methods on a daily SPY-style price series,
honestly, via genuine 1-step-ahead evaluation on a held-out chronological
test period — and report whatever the result actually is, including if
"simpler" beats "fancier."

## 2. Architecture

```
┌────────────────────┐   GET /api/leaderboard, /api/chart   ┌──────────────────────┐
│  Browser (leaderboard│ ─────────────────────────────────▶ │   FastAPI app.py     │
│  + price chart +     │ ◀───────────────────────────────── │   (uvicorn)          │
│  live forecast)       │   POST /api/forecast                 └──────────┬───────────┘
└────────────────────┘                                                   │ loads
                                                                ┌──────────▼───────────┐
                                                                │ models/                │
                                                                │  gradient_boosting     │
                                                                │   .joblib              │
                                                                │  metrics.json           │
                                                                │  test_period_chart_data│
                                                                └──────────▲───────────┘
                                                                           │ produced by
                                                                ┌──────────┴───────────┐
                                                                │ src/train.py            │
                                                                │  (5 methods, offline)   │
                                                                └──────────▲───────────┘
                                                                           │ reads
                                                                ┌──────────┴───────────┐
                                                                │ data/*.csv              │
                                                                │ (synthetic or real)     │
                                                                └───────────────────────┘
```

Only the gradient boosting model is persisted — ETS and ARIMA are cheap
enough (~0.03s / ~0.15s per fit) to refit live on request rather than
serialize, which also sidesteps statsmodels' more involved model
persistence story.

## 3. Data schema

| column | type | notes |
|---|---|---|
| date | date | trading day (business days only) |
| close | float | daily closing price — **target** |

`data/generate_data.py`'s docstring explains the deliberate construction:
a dominant near-random-walk component, a small genuine mean-reversion
term in returns, a mild day-of-week effect, and volatility clustering.

## 4. Modeling (`src/train.py`) — 5 methods, honest 1-step-ahead evaluation

Every method's prediction for test-set day `t` uses only real historical
data through `close[t-1]` — never its own prior forecast and never any
future value:

1. **Naive**: `pred[t] = close[t-1]`
2. **Moving average**: `pred[t] = mean(close[t-5:t])`
3. **Exponential smoothing**: statsmodels `ExponentialSmoothing` (Holt's
   damped linear trend), **refit at every test step** on the true
   history accumulated so far
4. **ARIMA**: statsmodels `ARIMA`, order chosen ONCE via AIC grid search
   over `p,q in {0,1,2}`, `d=1` on the training data only, then refit at
   every test step (same order, new data) on the true history so far
5. **Gradient boosting**: scikit-learn `GradientBoostingRegressor` on
   lag/rolling features (`data_prep.py`), trained ONCE on the training
   split (each test row's features are already built from true past
   prices, so no refitting is needed for honest 1-step-ahead evaluation)

**Results on this build's synthetic data** (see `models/metrics.json`
for the live numbers): naive MAE~$3.80; moving average MAE~$4.98
(worse); exponential smoothing MAE~$3.78 and ARIMA MAE~$3.78 (both
narrowly better, <1%); gradient boosting MAE~$3.89 (worse). See
`AUDIT_REPORT.md` for why this is a real, checked result, and
`PROMPTS.md` for why it's the expected outcome for financial data, not
a disappointing one.

## 5. API contract (`app.py`)

| endpoint | method | purpose |
|---|---|---|
| `/` | GET | serves the frontend |
| `/api/health` | GET | liveness check |
| `/api/leaderboard` | GET | all 5 methods' MAE/RMSE/MAPE, best method, which beat naive |
| `/api/chart` | GET | test-period actual vs. naive-forecast series, for plotting |
| `/api/forecast` | POST | live next-trading-day forecast from all 5 methods |

## 6. Frontend

Plain HTML/CSS/JS (no build step, no charting library — a Canvas 2D line
chart is enough for this): a leaderboard table with "beats naive"
badges, an actual-vs-naive price chart for the test period, and a
"get tomorrow's forecast" button showing all 5 methods' next-day
predictions as cards with the price delta from today's close.

## 7. Verification performed

- Timing check before running the full refit loop: single ARIMA fit
  ~0.13s, single ETS fit ~0.03s, confirming the ~225-step refit loop for
  each would complete in well under a minute (it took 29s total)
- `python src/train.py` — trains and compares all 5 methods, printed
  leaderboard inspected manually
- Statistical check on the data generator: an initial `MEAN_REVERSION_STRENGTH`
  value produced a POSITIVE sample lag-1 autocorrelation despite a
  negative population parameter (within 1 standard error of the
  estimate's sampling noise) — the strength was increased and re-verified
  until the sample statistic reliably matched the intended sign (see
  `AUDIT_REPORT.md` section 1 for the full investigation)
- `uvicorn app:app` + `curl` against every endpoint — correct results
- `python -m pytest tests/` — 5/5 passing, including a test that
  explicitly checks the naive baseline is NOT artificially weakened
  (moving average must lose to it) and that any method beating naive
  does so by a small, plausible margin rather than a suspiciously large one

## 8. Explicit non-goals (kept out of scope per "not fancy, simple frontend and backend")

- No deep-learning forecasting model (LSTM/GRU/Transformer) — see `PROMPTS.md`
- No real market data (no internet/API access in this environment)
- No multi-step-ahead / recursive forecasting evaluation
- No walk-forward periodic refitting of the GBM model
- No authentication, database, or multi-user admin dashboard
- No containerization/CI — this is a local, single-process demo
