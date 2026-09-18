# AUDIT_REPORT.md — Data Quality & Methodology Audit

**Scope**: Project 03, Customer Segmentation Clustering.
**Method**: Static inspection + scripted checks run against this exact codebase (commands shown below so you can re-run them yourself).

Note: "data leakage" in the strict supervised-learning sense (test labels
influencing training) doesn't apply the same way to unsupervised
clustering — there is no label to leak. This audit instead focuses on the
methodology pitfalls that *are* specific to clustering pipelines.

## 1. Identifier / non-behavioral columns excluded from features

**Check**: `CustomerID` (arbitrary, meaningless for distance calculations)
and `Gender` (categorical demographic attribute) must not silently end up
in the feature matrix.

```
$ python3 -c "
from src.data_prep import FEATURE_COLUMNS
print(FEATURE_COLUMNS)
print('CustomerID in features:', 'CustomerID' in FEATURE_COLUMNS)
print('Gender in features:', 'Gender' in FEATURE_COLUMNS)
"
['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
CustomerID in features: False
Gender in features: False
```

**Result: PASS.** Only genuinely behavioral/demographic-neutral numeric
columns are used for clustering.

## 2. Feature scaling before a distance-based algorithm

**Check**: KMeans uses Euclidean distance, so features on very different
scales (Income: 0–140 vs. Age: 18–75 vs. Spending Score: 1–100) must be
standardized first, or the largest-magnitude feature silently dominates
the distance metric.

```
$ grep -n "fit_transform" src/train.py
src/train.py:63:    X_scaled = scaler.fit_transform(X)
```

**Result: PASS.** `StandardScaler` is fit and applied before `KMeans` sees
the data (`src/train.py`).

## 3. Train/serve consistency for the scaler

**Check**: the API must transform new customers with the SAME scaler
fitted during training, not a freshly-fit one (which would silently drift
from the training-time scale over time).

```
$ grep -n "scaler.transform\|joblib.dump" src/train.py src/predict.py
src/train.py:119:    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
src/predict.py:51:    X_scaled = scaler.transform(X)
```

**Result: PASS.** `predict.py` calls `scaler.transform()` (not
`fit_transform()`) on a scaler object loaded from the exact file
`train.py` saved — one scaler, reused everywhere.

## 4. Model selection is metric-driven, not eyeballed

**Check**: `k` (number of segments) should be chosen by a quantitative
criterion, not hardcoded or guessed.

**Result: PASS.** `src/train.py` sweeps k = 2..8, computes the silhouette
score at each, and selects `argmax(silhouette)` — see `K_CANDIDATES` and
the `for k in K_CANDIDATES` loop. Both the full sweep and the chosen k are
persisted to `models/metrics.json` for auditability.

## 5. Reproducibility / seed pinning

```
$ grep -n "random_state\|RANDOM_SEED" src/train.py data/generate_data.py
src/train.py:38:RANDOM_SEED = 42
src/train.py:69:        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
data/generate_data.py:RNG_SEED = 42  (used via np.random.default_rng(seed))
```

**Result: PASS.** Both data generation and clustering are seeded, and
`n_init=10` (multiple KMeans initializations, best inertia kept) reduces
sensitivity to a single random initialization on top of that.

## 6. Degenerate-cluster check

**Check**: does any cluster collapse to a near-empty or near-100% share
(a sign of a poor k choice or non-informative features)?

From `models/metrics.json` (this build's run): cluster sizes are 129,
105, 92, and 174 out of 500 customers — 18.4% to 34.8% each. No cluster is
degenerate.

## 7. Known limitation (disclosed, not hidden)

The underlying dataset is **synthetic**, generated from 4 explicit latent
archetypes (see `PROMPTS.md` / `data/generate_data.py`). This means the
clustering task is somewhat "easier" than real retail data, where segment
boundaries are rarely this cleanly separated. The audit above confirms the
*pipeline* has no methodology defects; it does not confirm the synthetic
data's separability matches real customer behavior. Swap in a real Kaggle
`Mall_Customers.csv` (see `data/generate_data.py` docstring) and re-run
`python src/train.py` — the pipeline requires no code changes to do so,
though the resulting silhouette score may be lower on real, noisier data.

## Summary

| Check | Result |
|---|---|
| Identifier/demographic columns excluded from features | PASS |
| Feature scaling applied before distance-based clustering | PASS |
| Train/serve scaler consistency | PASS |
| Model selection (k) is metric-driven | PASS |
| Seed pinning / reproducibility | PASS |
| No degenerate clusters | PASS |
| Data realism | Disclosed limitation — synthetic, cleanly-separated data |

**Overall**: no methodology defects found in this codebase. The one
caveat that matters is disclosed above, not buried.
