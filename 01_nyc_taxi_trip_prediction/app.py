"""
app.py
------
CRISP-DM Phase 6: Deployment.

FastAPI backend that:
  1. Serves the prediction API (`/api/predict`, `/api/metrics`, `/api/health`)
  2. Serves the plain HTML/JS/CSS frontend from ./static (no build step --
     kept intentionally simple, per project scope).

Run:
    pip install -r requirements.txt
    python data/generate_data.py     # first time only
    cd src && python train.py        # first time only
    cd ..
    uvicorn app:app --reload --port 8001
Then open http://127.0.0.1:8001
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import predict_trip, get_metrics  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="NYC Taxi Trip Duration Predictor",
    description="CRISP-DM demo: predicts NYC taxi trip duration & estimates fare from pickup/dropoff points.",
    version="1.0.0",
)


class TripRequest(BaseModel):
    pickup_lat: float = Field(..., ge=40.4, le=41.0)
    pickup_lon: float = Field(..., ge=-74.3, le=-73.6)
    dropoff_lat: float = Field(..., ge=40.4, le=41.0)
    dropoff_lon: float = Field(..., ge=-74.3, le=-73.6)
    passenger_count: int = Field(1, ge=1, le=6)
    pickup_datetime: str = Field(..., description="ISO 8601, e.g. 2024-06-14T08:30:00")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics():
    """Model card / admin-dashboard data: evaluation metrics from training."""
    try:
        return get_metrics()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")


@app.post("/api/predict")
def predict(req: TripRequest):
    try:
        result = predict_trip(
            req.pickup_lat, req.pickup_lon,
            req.dropoff_lat, req.dropoff_lon,
            req.passenger_count, req.pickup_datetime,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")
    return result


# --- static frontend (plain HTML/CSS/JS, no build step) ---
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
