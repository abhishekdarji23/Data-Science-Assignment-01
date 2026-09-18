# 11 — Enterprise DS Audit (6-Dimension Scorecard)

An automated audit that scores a train/test dataset across six independent
data-quality and leakage dimensions, and lists exactly what it found wrong
in each one. Plain FastAPI backend, plain HTML/JS frontend, no framework,
no build step.

## Stack

- **Backend:** FastAPI, no external dependencies beyond the framework itself
- **Frontend:** one `index.html` + `app.js`, a scorecard grid, no framework

## Running it

```bash
# Terminal 1 — backend (port 8011)
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8011

# Terminal 2 — frontend
cd frontend
python3 -m http.server 5184
# open http://localhost:5184
```

## The dataset has real problems, on purpose

`backend/data.py` builds a synthetic train/test dataset with six issues
deliberately seeded in, one per audit dimension, so the scorecard has
something real to catch instead of always reporting a perfect score:

| Dimension | Injected problem | What the check does |
|---|---|---|
| **Leakage** | 5 test rows are exact duplicates of train rows; a `leaky_feature` column is a near-perfect proxy for the target | Flags exact feature-duplicates across the split, and flags any feature correlated >0.9 with the target |
| **Missing Data** | ~8% of `f2` values are missing in train | Flags any column above a 5% missing-value threshold |
| **Distribution Drift** | `f2`'s test-set values are shifted away from its train-set values | Compares train vs test mean, normalized by train std-dev, per feature |
| **Label Quality** | target is imbalanced in part of train; one feature combination appears twice with opposite labels | Flags class imbalance and conflicting-label duplicates |
| **Split Integrity** | (this one is actually clean) | Checks the test-split ratio is 15-25% of the data and that ids are unique within each split |
| **Reproducibility** | the recorded dataset hash in `metadata` is stale relative to the actual data | Flags a missing seed record, and flags a recorded hash that no longer matches the current data |

Run it and the scorecard should come back around **70/100 overall**, with
leakage and reproducibility failing outright, missing data / drift / label
quality warning, and split integrity passing clean — five real findings out
of six dimensions, which is the point: an audit that always says "100/100"
isn't checking anything.

## Extending it

`backend/audit.py` has one function per dimension (`audit_leakage`,
`audit_missing_data`, etc.), each taking the raw train/test rows and
returning `{score, status, findings}`. To point this at a real dataset
instead of the synthetic one, swap out `data.build_dataset()` for a loader
that returns the same shape (`train`, `test`, `metadata`) — the audit
functions themselves don't know or care that the data was synthetic.
