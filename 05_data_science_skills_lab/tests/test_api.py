"""
test_api.py
-----------
Minimal smoke tests. Run with: `python -m pytest tests/` from the project root.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_skills_catalog_nonempty():
    r = client.get("/api/skills")
    assert r.status_code == 200
    skills = r.json()
    assert len(skills) >= 10
    for s in skills:
        assert "id" in s and "crisp_dm_phase" in s and "description" in s


def test_dataset_summary():
    r = client.get("/api/dataset/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["n_rows"] > 0
    assert "Survived" in body["columns"]


def test_execute_unknown_skill_404():
    r = client.post("/api/skills/not_a_real_skill/execute")
    assert r.status_code == 404


def test_execute_every_registered_skill():
    """Every skill in the catalog must actually execute without error and
    return one of the four supported display types."""
    skills = client.get("/api/skills").json()
    assert len(skills) > 0
    valid_types = {"metrics", "table", "bar_chart", "list"}
    for s in skills:
        r = client.post(f"/api/skills/{s['id']}/execute")
        assert r.status_code == 200, f"{s['id']} failed: {r.text}"
        body = r.json()
        assert body["display_type"] in valid_types
        assert body["summary"]
        assert body["data"] is not None


def test_confusion_matrix_shape():
    r = client.post("/api/skills/confusion_matrix/execute")
    assert r.status_code == 200
    body = r.json()
    assert body["display_type"] == "table"
    assert len(body["data"]["rows"]) == 2
    assert len(body["data"]["columns"]) == 3


def test_duplicate_detection_finds_injected_duplicates():
    """generate_data.py deliberately injects 8 duplicate rows -- this
    should find at least that many (16, since keep=False counts both copies)."""
    r = client.post("/api/skills/duplicate_rows/execute")
    assert r.status_code == 200
    body = r.json()
    assert "No duplicates" not in body["data"][0]
