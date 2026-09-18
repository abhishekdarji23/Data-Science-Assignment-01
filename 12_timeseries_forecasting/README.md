# 12 — TimePulse Forecasting

A small time-series forecasting tool: 40-lag ACF/PACF, a multi-horizon
forecast with a widening uncertainty "fan", and a backtest against real
held-out days. Everything — the decomposition, the ACF/PACF math, the
forecaster — is implemented from scratch in plain Python, no
statsmodels/scikit-learn. Plain FastAPI backend, plain HTML/JS frontend
with hand-rolled SVG charts (no charting library), no build step.

## Stack

- **Backend:** FastAPI, pure-Python statistics and forecasting
- **Frontend:** one `index.html` + `app.js`, SVG charts built by string
  templates — no chart library, no framework

## Running it

```bash
# Terminal 1 — backend (port 8012)
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8012

# Terminal 2 — frontend
cd frontend
python3 -m http.server 5185
# open http://localhost:5185
```

## What's actually happening

- **`data.py`** — a synthetic 2-year daily series: linear trend + a 7-day
  weekly pattern (busier on weekends) + a slow annual wave + noise.
- **`stats.py`** — `acf()` is the textbook lag-k sample autocorrelation.
  `pacf()` uses the **Durbin-Levinson recursion**: each order's partial
  autocorrelation is built from the previous order's AR coefficients
  instead of running a separate regression per lag. On this data, ACF
  shows a clear spike every 7 lags (the weekly seasonality); PACF cuts off
  sharply after lag 1-2, which is the expected signature of a mostly-AR(1)-
  like process with a seasonal component layered on top.
- **`forecast.py`** — classical additive decomposition: a centered
  7-day moving average for trend, average day-of-week deviation for
  seasonality, everything left over is residual. The trend gets a linear
  extrapolation forward, the seasonal pattern repeats, and the 95%
  interval widens with `sqrt(horizon)` — reflecting that day 30's forecast
  is genuinely less certain than tomorrow's, not just decorated to look
  that way.
- **Backtest** — holds out the last 60 days, fits on everything before
  that, forecasts forward, and reports MAE/RMSE against what actually
  happened. Comes out to roughly MAE ≈ 2.5 / RMSE ≈ 3.1 on this dataset,
  which is close to the injected noise's own standard deviation (3) — i.e.
  the model isn't leaving much predictable structure on the table.

## Frontend tabs

1. **Series & Forecast** — last 90 days of history, 30-day forecast fan
2. **ACF / PACF** — bar charts with a dashed ~95% significance band
3. **Backtest** — actual vs. predicted for the held-out window, plus MAE/RMSE
