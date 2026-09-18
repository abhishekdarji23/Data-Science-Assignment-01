"""
app.py
------
CRISP-DM Phase 6: Deployment.

FastAPI backend that:
  1. Serves the recommendation API (`/api/recommend`, `/api/rules`, `/api/items`, `/api/metrics`)
  2. Serves the plain HTML/JS/CSS frontend from ./static (no build step --
     kept intentionally simple, per project scope).

Run:
    pip install -r requirements.txt
    python data/generate_data.py     # first time only
    cd src && python train.py        # first time only
    cd ..
    uvicorn app:app --reload --port 8004
Then open http://127.0.0.1:8004
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import recommend, get_metrics, get_rules, get_catalog  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Market Basket Pattern Mining",
    description="CRISP-DM demo: Apriori association-rule mining over synthetic grocery baskets.",
    version="1.0.0",
)


class CartRequest(BaseModel):
    items: list[str] = Field(..., min_length=1, max_length=20)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics():
    """Model card / admin-dashboard data: dataset stats, mining thresholds, top items."""
    try:
        return get_metrics()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Rules not mined yet. Run `python src/train.py` first.")


@app.get("/api/rules")
def rules(limit: int = 50):
    """Top association rules, sorted by lift."""
    try:
        return get_rules(limit)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Rules not mined yet. Run `python src/train.py` first.")


@app.get("/api/items")
def items():
    """Full item catalog, for building the cart-picker UI."""
    try:
        return get_catalog()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Rules not mined yet. Run `python src/train.py` first.")


@app.post("/api/recommend")
def recommend_endpoint(req: CartRequest):
    try:
        return recommend(req.items)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Rules not mined yet. Run `python src/train.py` first.")


# --- static frontend (plain HTML/CSS/JS, no build step) ---
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
