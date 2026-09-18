"""
Minimal FastAPI server for the time-series forecasting demo.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8012
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from data import generate_series
from stats import acf, pacf
from forecast import forecast as run_forecast, backtest as run_backtest

app = FastAPI(title="TimePulse Forecasting API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SERIES = generate_series(num_days=730, seed=11)
VALUES = [r["value"] for r in SERIES]


@app.get("/api/series")
def get_series():
    return {"series": SERIES}


@app.get("/api/acf")
def get_acf(max_lag: int = 40):
    return {"max_lag": max_lag, "values": acf(VALUES, max_lag)}


@app.get("/api/pacf")
def get_pacf(max_lag: int = 40):
    return {"max_lag": max_lag, "values": pacf(VALUES, max_lag)}


@app.get("/api/forecast")
def get_forecast(horizon: int = 30):
    return run_forecast(SERIES, horizon=horizon)


@app.get("/api/backtest")
def get_backtest(holdout: int = 60):
    return run_backtest(SERIES, holdout=holdout)
