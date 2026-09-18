"""
app.py
------
CRISP-DM Phase 6: Deployment.

FastAPI backend that:
  1. Runs the audit platform live (`/api/audit/run`, `/api/audit/compare`)
  2. Serves the plain HTML/JS/CSS dashboard from ./static

Run:
    pip install -r requirements.txt
    python data/generate_data.py     # first time only
    uvicorn app:app --reload --port 8013
Then open http://127.0.0.1:8013
"""
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from audit_runner import run_audit, run_comparison  # noqa: E402
from audit_checks import list_checks  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="CRISP-DM NYC Taxi Audit Platform",
    description="Runs a real, executable CRISP-DM/leakage audit against a taxi trip-duration pipeline, live.",
    version="1.0.0",
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/audit/checks")
def audit_checks_catalog():
    return list_checks()


@app.post("/api/audit/run")
def audit_run(variant: Literal["clean", "flawed"] = "clean"):
    try:
        return run_audit(variant)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Data not generated yet. Run `python data/generate_data.py` first.")


@app.post("/api/audit/compare")
def audit_compare():
    try:
        return run_comparison()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Data not generated yet. Run `python data/generate_data.py` first.")


# --- static frontend (plain HTML/CSS/JS, no build step) ---
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
