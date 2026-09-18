# 10 — CRISP-DM Census Analytics (Cosine LSH)

Walks through all seven CRISP-DM phases on a synthetic census-style panel,
using **cosine LSH (locality-sensitive hashing)** as the modeling technique:
instead of predicting a single label, it finds the most similar demographic
profiles to a given one, without brute-force scanning the whole panel.

Kept deliberately simple: plain FastAPI backend (one endpoint per phase),
plain HTML/JS frontend (tabs, no framework, no build step).

## Stack

- **Backend:** FastAPI + NumPy, all data synthetic and generated on startup
  (no external dataset download)
- **Frontend:** one `index.html` + `app.js`, tab-based, no framework

## Running it

```bash
# Terminal 1 — backend (port 8010)
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8010

# Terminal 2 — frontend
cd frontend
python3 -m http.server 5183
# open http://localhost:5183
```

## The seven phases (and where each lives in the code)

1. **Business Understanding** — static goal statement (`/api/phase1`)
2. **Data Understanding** — summary stats over 300 synthetic records:
   age/hours distributions, category counts (`/api/phase2`, `data.py`)
3. **Data Preparation** — min-max scaling + one-hot encoding into a
   13-dimensional feature vector per record (`/api/phase3`, `features.py`)
4. **Modeling** — builds the cosine LSH index: 4 hash tables, 6 random
   hyperplanes each (`/api/phase4`, `lsh.py`)
5. **Evaluation** — samples 20–30 records, compares each one's LSH-retrieved
   top-k neighbors against an exact brute-force top-k, and reports
   recall@k (`/api/phase5`) — comes out around **0.95–1.0** on this dataset,
   meaning the approximate index almost always finds the true nearest
   neighbors
6. **Deployment** — the actual "product": pick any record in the UI and get
   its most similar profiles back, with the number of candidates scanned
   and latency shown (`/api/phase6/similar/{id}`)
7. **Monitoring & Maintenance** — running counters (query count, average
   latency) accumulated across phase 6 calls (`/api/phase7`)

## Why LSH here

With only 300 records, brute-force cosine search would be just as fast —
the point of this project is to show the *mechanism* (random hyperplane
hashing, candidate buckets, recall@k evaluation) on a scale small enough to
inspect by hand, not to demonstrate a speed win. The bucket stats on
`/api/phase4` and the recall numbers on `/api/phase5` are what would let you
tune `num_tables` / `num_hyperplanes` before pointing the same code at a
much larger panel where the speed-up would actually matter.
