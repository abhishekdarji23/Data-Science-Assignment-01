"""
app.py
------
CRISP-DM Phase 6: Deployment.

FastAPI backend that:
  1. Serves the anomaly detection API (`/api/score`, `/api/anomalies`, `/api/metrics`)
  2. Serves the plain HTML/JS/CSS frontend from ./static (no build step --
     kept intentionally simple, per project scope).

Run:
    pip install -r requirements.txt
    python data/generate_data.py     # first time only
    cd src && python train.py        # first time only
    cd ..
    uvicorn app:app --reload --port 8006
Then open http://127.0.0.1:8006
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import score_transaction, get_metrics, get_top_anomalies  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Anomaly Detection — Transaction Risk Scoring",
    description="CRISP-DM demo: Isolation Forest / LOF / z-score anomaly detection on transaction data.",
    version="1.0.0",
)


class TransactionRequest(BaseModel):
    amount_usd: float = Field(..., ge=0, le=100000)
    hour_of_day: int = Field(..., ge=0, le=23)
    account_age_days: float = Field(..., ge=0, le=20000)
    transactions_last_hour: int = Field(..., ge=0, le=1000)
    distance_from_home_km: float = Field(..., ge=0, le=20000)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics():
    """Model card / admin-dashboard data: comparison of all 3 methods, PR-AUC, precision/recall."""
    try:
        return get_metrics()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")


@app.get("/api/anomalies")
def anomalies(limit: int = 20):
    """Top flagged transactions from the training set, for a review-queue dashboard view."""
    try:
        return get_top_anomalies(limit)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")


@app.post("/api/score")
def score(req: TransactionRequest):
    try:
        return score_transaction(
            req.amount_usd, req.hour_of_day, req.account_age_days,
            req.transactions_last_hour, req.distance_from_home_km,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")


# --- static frontend (plain HTML/CSS/JS, no build step) ---
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
