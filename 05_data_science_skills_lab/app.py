"""
app.py
------
CRISP-DM Phase 6: Deployment.

FastAPI backend that:
  1. Serves the skills catalog (`/api/skills`) and executes any skill live
     on request (`/api/skills/{skill_id}/execute`), returning a friendly,
     display-typed result -- never a raw dataframe dump.
  2. Serves the plain HTML/JS/CSS frontend from ./static (no build step --
     kept intentionally simple, per project scope).

Run:
    pip install -r requirements.txt
    python data/generate_data.py     # first time only
    uvicorn app:app --reload --port 8005
Then open http://127.0.0.1:8005

Note: unlike projects 01/03/04, there is no separate `train.py` here --
skills are genuinely computed live, per request, on the current dataset
(that's the point of a "skills lab": each click re-runs real pandas/
scikit-learn code, not a cached result).
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data_prep import load_dataset  # noqa: E402
from skills import list_skills, get_skill, SKILL_REGISTRY  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Data Science Skills Mastery Lab",
    description="CRISP-DM demo: a catalog of live-executable data science skills on a Titanic-style dataset.",
    version="1.0.0",
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/skills")
def skills_catalog():
    """The catalog of available skills, grouped by CRISP-DM phase (metadata only, no execution)."""
    return list_skills()


@app.get("/api/dataset/summary")
def dataset_summary():
    """Quick dataset stats for the page header, independent of any single skill."""
    df = load_dataset()
    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "columns": list(df.columns),
    }


@app.post("/api/skills/{skill_id}/execute")
def execute_skill(skill_id: str):
    """Runs one skill LIVE on the current dataset and returns a friendly,
    display-typed result (metrics / table / bar_chart / list) -- this is
    the endpoint that replaces the original repo's raw-JSON output."""
    s = get_skill(skill_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Unknown skill_id '{skill_id}'")
    try:
        df = load_dataset()
        result = s.fn(df)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Skill execution failed: {e}")

    return {
        "skill_id": s.id,
        "title": s.title,
        "crisp_dm_phase": s.crisp_dm_phase,
        "description": s.description,
        "summary": result.summary,
        "display_type": result.display_type,
        "data": result.data,
    }


# --- static frontend (plain HTML/CSS/JS, no build step) ---
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
