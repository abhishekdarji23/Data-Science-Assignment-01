"""
app.py
------
CRISP-DM Phase 6: Deployment.

FastAPI backend that:
  1. Serves the clustering API (`/api/predict`, `/api/clusters`, `/api/customers`, `/api/metrics`)
  2. Serves the plain HTML/JS/CSS frontend from ./static (no build step --
     kept intentionally simple, per project scope).

Run:
    pip install -r requirements.txt
    python data/generate_data.py     # first time only
    cd src && python train.py        # first time only
    cd ..
    uvicorn app:app --reload --port 8003
Then open http://127.0.0.1:8003
"""
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import predict_segment, get_metrics  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="Customer Segmentation Clustering",
    description="CRISP-DM demo: KMeans clustering of customers by age, income, and spending score.",
    version="1.0.0",
)


class CustomerRequest(BaseModel):
    age: float = Field(..., ge=15, le=100)
    annual_income_k: float = Field(..., ge=0, le=500)
    spending_score: float = Field(..., ge=1, le=100)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics():
    """Model card / admin-dashboard data: silhouette scores, elbow curve, cluster profiles."""
    try:
        return get_metrics()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")


@app.get("/api/clusters")
def clusters():
    """Just the cluster profile table (subset of /api/metrics), for the segments view."""
    try:
        return get_metrics()["cluster_profiles"]
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")


@app.get("/api/customers")
def customers(limit: int = 500):
    """Every customer with its assigned cluster, for the scatter plot."""
    path = MODELS_DIR / "customers_with_clusters.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")
    df = pd.read_csv(path).head(limit)
    return df.to_dict(orient="records")


@app.post("/api/predict")
def predict(req: CustomerRequest):
    try:
        return predict_segment(req.age, req.annual_income_k, req.spending_score)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")


# --- static frontend (plain HTML/CSS/JS, no build step) ---
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
