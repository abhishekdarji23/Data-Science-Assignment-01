# WALKTHROUGH.md — Architectural Deep-Dive

> No video walkthrough is included with this build: recording and uploading
> a narrated video (e.g. to YouTube) isn't something this assistant can do —
> there's no video/audio generation or YouTube-upload capability available.
> This document is the substitute: a code-ordered tour you (or a screen
> recorder) can follow to produce one in a few minutes if you want it, plus
> a suggested shot list at the bottom.

## Reading order (this *is* the tour)

1. **`data/generate_data.py`** — where the data comes from. Read the
   module docstring, then compare the `normal` and `anomaly` DataFrames
   side by side: anomalies draw from genuinely different distributions
   for amount (higher), hour (`_nighttime_weighted_hours()`), account age
   (newer), transaction velocity (higher), and distance from home
   (farther) — not just noise added to the same distribution.

2. **`src/data_prep.py`** — small and deliberate: `FEATURE_COLUMNS`
   excludes `is_anomaly` (the label) and `transaction_id` (an identifier)
   — only genuine behavioral signals are model inputs.

3. **`src/train.py`** — the core of this project. Read the module
   docstring for why all 3 methods are unsupervised and why PR-AUC is the
   headline metric, then:
   - the 3 method blocks (Isolation Forest, LOF, z-score) — note the sign
     flip on each ML method's score (`-decision_function`,
     `-score_samples`) so "higher = more anomalous" is consistent across
     all 3, letting `evaluate()` treat them identically
   - `evaluate()` — the shared, fair comparison logic (flag the top-K by
     score, compare against ground truth)
   - the printed comparison table — this is where you see Isolation
     Forest and the z-score baseline do well, and LOF do poorly (see
     `AUDIT_REPORT.md` §3 for why)

4. **`src/predict.py`** — `score_transaction()`: loads the saved
   Isolation Forest + scaler (never refit at serving time), scores one
   new transaction, and rescales the raw score to a 0-100 "risk score"
   purely for UI readability (`_risk_score()` — the underlying ranking is
   unaffected by this rescaling).

5. **`app.py`** — FastAPI wiring: `/api/score` calls
   `predict.score_transaction()`; `/api/anomalies` serves the
   pre-computed review queue from `models/scored_transactions.csv`;
   `/api/metrics` is the full method-comparison model card.

6. **`static/app.js`** — `loadMetrics()` renders the method comparison
   table, sorted by PR-AUC with the winner starred; the score form posts
   to `/api/score` and color-codes the result (red if flagged, green if
   not); `loadAnomalies()` renders the review queue with ground truth
   shown alongside each flagged transaction.

## How a request flows end-to-end

```
user fills the "score a new transaction" form, submits
  → static/app.js POSTs the 5 feature values to /api/score
    → app.py: TransactionRequest validates bounds
      → predict.py: score_transaction() scales via the SAVED scaler,
        scores via the SAVED Isolation Forest
      ← raw score, 0-100 risk score, flagged boolean
    ← JSON response
  ← static/app.js renders a red or green risk card
```

## Suggested shot list (if you want to record your own walkthrough video)

1. `python data/generate_data.py` running, show the printed anomaly-rate and feature-mean comparison (normal vs. anomaly)
2. `python src/train.py` running, show the 3-method comparison table print out — point out Isolation Forest winning and LOF's weak PR-AUC
3. `uvicorn app:app --reload --port 8006` starting up
4. Browser: point at the method comparison table
5. Browser: submit an obviously-normal transaction (small amount, daytime, old account) — show the green low-risk result
6. Browser: submit an obviously-suspicious one (large amount, 3am, brand-new account, many transactions/hour, far from home) — show the red high-risk result
7. Browser: scroll to the review queue, point out the "true anomaly" badges lining up with high scores
8. Terminal: `python -m pytest tests/` passing
9. Code editor: 30 seconds each on `train.py`'s 3 method blocks and `evaluate()` — the reading order above
