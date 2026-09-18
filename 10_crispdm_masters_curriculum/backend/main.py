"""
FastAPI backend walking through the 7 phases of CRISP-DM on a synthetic
census-style panel, using cosine LSH as the "modeling" technique (finding
similar demographic profiles rather than predicting a single label).

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8010
"""

import time
from collections import Counter
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data import generate_records, EDUCATION_LEVELS, OCCUPATIONS
from features import build_feature_matrix
from lsh import CosineLSH, brute_force_top_k, cosine_similarity

app = FastAPI(title="CRISP-DM Census Analytics API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Phase 1 (Business Understanding) is just a static statement.
# Phases 2-3 run once at startup on the synthetic panel.
# Phase 4 builds the LSH index (also at startup, index is cheap to build).
# Phases 5-7 are queried live.
# ---------------------------------------------------------------------------
RECORDS = generate_records(n=300, seed=42)
FEATURE_MATRIX, FEATURE_NAMES = build_feature_matrix(RECORDS)
IDS = [r["id"] for r in RECORDS]

lsh = CosineLSH(dim=FEATURE_MATRIX.shape[1], num_tables=4, num_hyperplanes=6, seed=7)
lsh.build(FEATURE_MATRIX, IDS)

monitoring_stats = {"total_queries": 0, "total_latency_ms": 0.0, "last_query_id": None}


@app.get("/api/phase1")
def phase1_business_understanding():
    return {
        "phase": "1. Business Understanding",
        "goal": (
            "Given a panel of demographic profiles, find other profiles most similar to a "
            "given one (for use cases like cohort analysis or peer benchmarking), using an "
            "approximate-nearest-neighbor technique that scales better than brute-force "
            "comparison as the panel grows."
        ),
        "success_criteria": "LSH-based retrieval should recover most of the true top-k nearest neighbors (high recall@k) while only scanning a small fraction of the panel.",
    }


@app.get("/api/phase2")
def phase2_data_understanding():
    ages = [r["age"] for r in RECORDS]
    hours = [r["hours_per_week"] for r in RECORDS]
    return {
        "phase": "2. Data Understanding",
        "num_records": len(RECORDS),
        "columns": ["age", "education", "occupation", "hours_per_week", "income"],
        "age": {"min": min(ages), "max": max(ages), "mean": round(sum(ages) / len(ages), 1)},
        "hours_per_week": {"min": min(hours), "max": max(hours), "mean": round(sum(hours) / len(hours), 1)},
        "education_counts": dict(Counter(r["education"] for r in RECORDS)),
        "occupation_counts": dict(Counter(r["occupation"] for r in RECORDS)),
        "income_counts": dict(Counter(r["income"] for r in RECORDS)),
        "sample_records": RECORDS[:5],
    }


@app.get("/api/phase3")
def phase3_data_preparation():
    return {
        "phase": "3. Data Preparation",
        "steps": [
            "Min-max scale numeric columns (age, hours_per_week) to [0, 1]",
            f"One-hot encode education ({len(EDUCATION_LEVELS)} categories)",
            f"One-hot encode occupation ({len(OCCUPATIONS)} categories)",
            "Concatenate into a single feature vector per record",
        ],
        "vector_dim": FEATURE_MATRIX.shape[1],
        "feature_names": FEATURE_NAMES,
        "sample_vectors": [
            {"id": IDS[i], "vector": [round(v, 3) for v in FEATURE_MATRIX[i].tolist()]} for i in range(3)
        ],
    }


@app.get("/api/phase4")
def phase4_modeling():
    return {
        "phase": "4. Modeling",
        "technique": "Cosine similarity LSH (random hyperplane hashing)",
        "num_hash_tables": lsh.num_tables,
        "hyperplanes_per_table": lsh.num_hyperplanes,
        "vector_dim": lsh.dim,
        "bucket_stats": lsh.bucket_stats(),
    }


@app.get("/api/phase5")
def phase5_evaluation(k: int = 5, sample: int = 30):
    sample = min(sample, len(RECORDS))
    rng = np.random.default_rng(123)
    sample_indices = rng.choice(len(RECORDS), size=sample, replace=False)

    recalls = []
    details = []
    for idx in sample_indices:
        rid = IDS[idx]
        vector = FEATURE_MATRIX[idx]

        lsh_results, _ = lsh.query(vector, k=k, exclude_id=rid)
        exact_results = brute_force_top_k(FEATURE_MATRIX, IDS, vector, k=k, exclude_id=rid)

        lsh_ids = {r[0] for r in lsh_results}
        exact_ids = {r[0] for r in exact_results}
        recall = len(lsh_ids & exact_ids) / k
        recalls.append(recall)
        details.append({"query_id": int(rid), "recall_at_k": round(recall, 2)})

    return {
        "phase": "5. Evaluation",
        "k": k,
        "sample_size": sample,
        "average_recall_at_k": round(sum(recalls) / len(recalls), 3),
        "details": details,
    }


@app.get("/api/phase6/similar/{record_id}")
def phase6_deployment(record_id: int, k: int = 5):
    if record_id not in IDS:
        raise HTTPException(status_code=404, detail=f"No record with id {record_id}")

    idx = IDS.index(record_id)
    vector = FEATURE_MATRIX[idx]

    start = time.perf_counter()
    results, num_candidates = lsh.query(vector, k=k, exclude_id=record_id)
    elapsed_ms = (time.perf_counter() - start) * 1000

    monitoring_stats["total_queries"] += 1
    monitoring_stats["total_latency_ms"] += elapsed_ms
    monitoring_stats["last_query_id"] = record_id

    by_id = {r["id"]: r for r in RECORDS}
    return {
        "phase": "6. Deployment",
        "query_record": by_id[record_id],
        "num_candidates_scanned": num_candidates,
        "elapsed_ms": round(elapsed_ms, 3),
        "similar": [
            {"record": by_id[rid], "cosine_similarity": round(sim, 4)} for rid, sim in results
        ],
    }


@app.get("/api/phase7")
def phase7_monitoring():
    n = monitoring_stats["total_queries"]
    avg_latency = monitoring_stats["total_latency_ms"] / n if n else 0.0
    return {
        "phase": "7. Monitoring & Maintenance",
        "total_queries_served": n,
        "avg_latency_ms": round(avg_latency, 3),
        "last_query_id": monitoring_stats["last_query_id"],
    }


@app.get("/api/records")
def list_records():
    return {"records": RECORDS}
