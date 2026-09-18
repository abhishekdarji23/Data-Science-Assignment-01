"""
Minimal FastAPI server exposing the 6-dimension data-quality/leakage audit.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8011
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from data import build_dataset
from audit import run_full_audit

app = FastAPI(title="Enterprise DS Audit API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TRAIN, TEST, METADATA = build_dataset(seed=42)


@app.get("/api/summary")
def summary():
    return {
        "train_rows": len(TRAIN),
        "test_rows": len(TEST),
        "columns": ["id", "f1", "f2", "f3", "leaky_feature", "target"],
        "metadata": METADATA,
        "sample_train_rows": TRAIN[:5],
        "sample_test_rows": TEST[:5],
    }


@app.get("/api/audit")
def audit():
    return run_full_audit(TRAIN, TEST, METADATA)
