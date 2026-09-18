"""
app.py
------
CRISP-DM Phase 6: Deployment.

FastAPI backend that:
  1. Serves the leaderboard (`/api/leaderboard`) and live churn scoring
     (`/api/predict`) through the full stacking pipeline.
  2. Serves the plain HTML/JS/CSS frontend from ./static (no build step --
     kept intentionally simple, per project scope).

Run:
    pip install -r requirements.txt
    python data/generate_data.py     # first time only
    cd src && python train.py        # first time only
    cd ..
    uvicorn app:app --reload --port 8007
Then open http://127.0.0.1:8007
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import predict_churn, get_metrics  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="AutoML Stacking — Customer Churn",
    description="CRISP-DM demo: a scikit-learn model zoo + out-of-fold stacking ensemble (AutoGluon-style, without the AutoGluon dependency).",
    version="1.0.0",
)


class CustomerRequest(BaseModel):
    tenure_months: float = Field(..., ge=0, le=100)
    monthly_charges: float = Field(..., ge=0, le=300)
    total_charges: float = Field(..., ge=0, le=50000)
    contract_type: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    tech_support: Literal["Yes", "No"]
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    num_support_calls: int = Field(..., ge=0, le=50)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/leaderboard")
def leaderboard():
    """Model card / admin-dashboard data: every base model + the stacked
    ensemble, ranked by ROC-AUC on the held-out test set."""
    try:
        m = get_metrics()
        return {
            "leaderboard": m["leaderboard"],
            "best_model": m["best_model"],
            "meta_learner_weights": m["meta_learner_weights"],
            "n_train": m["n_train"],
            "n_test": m["n_test"],
            "churn_rate": m["churn_rate"],
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")


@app.post("/api/predict")
def predict(req: CustomerRequest):
    try:
        return predict_churn(
            req.tenure_months, req.monthly_charges, req.total_charges,
            req.contract_type, req.internet_service, req.tech_support,
            req.payment_method, req.num_support_calls,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run `python src/train.py` first.")


# --- static frontend (plain HTML/CSS/JS, no build step) ---
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
