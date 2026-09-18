# WALKTHROUGH.md — Architectural Deep-Dive

> No video walkthrough is included with this build: recording and uploading
> a narrated video (e.g. to YouTube) isn't something this assistant can do —
> there's no video/audio generation or YouTube-upload capability available.
> This document is the substitute: a code-ordered tour you (or a screen
> recorder) can follow to produce one in a few minutes if you want it, plus
> a suggested shot list at the bottom.

## Reading order (this *is* the tour)

1. **`data/generate_data.py`** — where the data comes from. Read the
   module docstring, then the churn-probability model: baseline + boosts
   for month-to-month contracts, short tenure, high charges, support-call
   volume, and lack of tech support, plus noise. Note `CONTRACT_CHURN_BOOST`
   specifically — it's the single strongest driver, and you'll see it
   surface again in the meta-learner weights later.

2. **`src/data_prep.py`** — read `CATEGORY_LEVELS` and the docstring's
   explanation of why categories are FIXED rather than inferred from
   whatever's in a given dataframe. This is a real production bug class
   ("single-row inference silently produces fewer dummy columns than
   training did") that's easy to introduce accidentally and worth seeing
   handled explicitly.

3. **`src/train.py`** — the core of this project. Read the module
   docstring for the overall stacking procedure, then walk the `for name,
   model in BASE_MODELS.items()` loop step by step:
   - `cross_val_predict(..., method="predict_proba")` — the OOF step;
     note this happens BEFORE `model.fit()` on the full train set, and
     uses a *fresh* unfit `model` object each time internally (that's
     what `cross_val_predict` does under the hood) — this is what
     prevents leakage
   - the refit (`model.fit(X_train_scaled, y_train)`) — now the same
     model IS fit on the full train set, for serving
   - after the loop: `meta_X_train = pd.DataFrame(oof_predictions)` —
     this is the "stack" — each row is one training customer, each
     column is one base model's OOF opinion of that customer
   - `meta_learner.fit(meta_X_train, y_train)` — the actual stacking step
   - the printed leaderboard — this is where you see the small, honest
     lift stacking provides over the best single model

4. **`src/predict.py`** — `predict_churn()`: builds features for one new
   customer, runs it through every saved base model (L1), assembles their
   probabilities into the same column order the meta-learner was trained
   on, then calls the meta-learner (L2) for the final probability —
   structurally identical to what `train.py` did, just for one row live
   instead of the whole training set in a loop.

5. **`app.py`** — FastAPI wiring: `Literal[...]` types on the categorical
   request fields (`CustomerRequest`) reject an invalid category before
   it ever reaches feature encoding; `/api/predict` calls
   `predict.predict_churn()`; `/api/leaderboard` serves the full model
   comparison.

6. **`static/app.js`** — `loadLeaderboard()` renders the bar chart +
   table, sorted by ROC-AUC with the winner starred; the score form
   posts to `/api/predict` and renders BOTH the final stacked prediction
   AND a table of every individual base model's opinion, so the
   ensembling is visible, not hidden behind one number.

## How a request flows end-to-end

```
user fills the "score a customer" form, submits
  → static/app.js POSTs the 8 field values to /api/predict
    → app.py: CustomerRequest validates (including Literal category checks)
      → predict.py: predict_churn()
        1. build_features() -- same fixed-category encoding as training
        2. every base model scores this one row (L1)
        3. meta_learner combines the 5 L1 opinions (L2)
      ← {churn_probability, risk_level, base_model_predictions, ...}
    ← JSON response
  ← static/app.js renders the stacked result + the 5-model breakdown table
```

## Suggested shot list (if you want to record your own walkthrough video)

1. `python data/generate_data.py` running, show the printed churn rate and contract-type churn gradient
2. `python src/train.py` running, show the leaderboard printout — point out the stacked ensemble's small lift over the best base model
3. `uvicorn app:app --reload --port 8007` starting up
4. Browser: point at the leaderboard bar chart, note the ensemble is starred
5. Browser: submit an obviously high-risk customer, show the red result and the 5-model breakdown table
6. Browser: submit an obviously low-risk customer, show the green result
7. Terminal: `python -m pytest tests/` passing
8. Code editor: 30 seconds each on the OOF loop in `train.py` and `predict_churn()` in `predict.py` — the reading order above
