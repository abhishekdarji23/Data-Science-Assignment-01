"""
app.py
------
CRISP-DM Phase 6: Deployment.

FastAPI backend that:
  1. Serves the leaderboard (`/api/leaderboard`) comparing tabular-only,
     text-only, and fusion models.
  2. Serves live price prediction (`/api/predict`) through all 3 models
     for side-by-side comparison.
  3. Serves the plain HTML/JS/CSS frontend from ./static.

Run:
    pip install -r requirements.txt
    python data/generate_data.py     # first time only
    cd src && python train.py        # first time only
    cd ..
    uvicorn app:app --reload --port 8014
Then open http://127.0.0.1:8014
"""
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import predict_price, get_metrics  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Multimodal AutoML Suite — Listing Price",
    description="CRISP-DM demo: compares tabular-only, text-only, and fusion models to show the real value of multimodal fusion.",
    version="1.0.0",
)


class ListingRequest(BaseModel):
    bedrooms: int = Field(..., ge=1, le=10)
    bathrooms: int = Field(..., ge=1, le=6)
    accommodates: int = Field(..., ge=1, le=16)
    room_type: Literal["Entire home/apt", "Private room", "Shared room"]
    neighborhood: Literal["Downtown", "Uptown", "Waterfront", "Suburb"]
    description: str = Field(..., min_length=1, max_length=1000)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/leaderboard")
def leaderboard():
    try:
        m = get_metrics()
        return {
            "leaderboard": m["leaderboard"],
            "best_model": m["best_model"],
            "fusion_beats_both_singles": m["fusion_beats_both_singles"],
            "n_train": m["n_train"],
            "n_test": m["n_test"],
            "n_listings": m["n_listings"],
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")


@app.post("/api/predict")
def predict(req: ListingRequest):
    try:
        return predict_price(
            req.bedrooms, req.bathrooms, req.accommodates,
            req.room_type, req.neighborhood, req.description,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")


# --- static frontend (plain HTML/CSS/JS, no build step) ---
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
