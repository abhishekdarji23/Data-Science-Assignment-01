"""
app.py
------
CRISP-DM Phase 6: Deployment.

FastAPI backend that:
  1. Serves the leaderboard (`/api/leaderboard`) comparing 5 forecasting
     methods on honest 1-step-ahead held-out performance.
  2. Serves chart data (`/api/chart`) for the test-period actual-vs-naive plot.
  3. Serves a live next-day forecast (`/api/forecast`) from all 5 methods.
  4. Serves the plain HTML/JS/CSS frontend from ./static.

Run:
    pip install -r requirements.txt
    python data/generate_data.py     # first time only
    cd src && python train.py        # first time only (~30s, refits ARIMA/ETS ~225x each)
    cd ..
    uvicorn app:app --reload --port 8015
Then open http://127.0.0.1:8015
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import forecast_next_day, get_metrics, get_chart_data  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="SPY Time Series Forecasting Comparison",
    description="CRISP-DM demo: honest 1-step-ahead comparison of 5 forecasting methods on a synthetic SPY-style price series.",
    version="1.0.0",
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/leaderboard")
def leaderboard():
    try:
        m = get_metrics()
        return {
            "leaderboard": m["leaderboard"],
            "best_method": m["best_method"],
            "beats_naive": m["beats_naive"],
            "n_train": m["n_train"],
            "n_test": m["n_test"],
            "arima_order": m["arima_order"],
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")


@app.get("/api/chart")
def chart():
    try:
        return get_chart_data()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")


@app.post("/api/forecast")
def forecast():
    try:
        return forecast_next_day()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")


# --- static frontend (plain HTML/CSS/JS, no build step) ---
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
