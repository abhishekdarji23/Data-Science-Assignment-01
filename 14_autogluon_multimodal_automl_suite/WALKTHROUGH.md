# WALKTHROUGH.md — Architectural Deep-Dive

> No video walkthrough is included with this build: recording and uploading
> a narrated video (e.g. to YouTube) isn't something this assistant can do —
> there's no video/audio generation or YouTube-upload capability available.
> This document is the substitute: a code-ordered tour you (or a screen
> recorder) can follow to produce one in a few minutes if you want it, plus
> a suggested shot list at the bottom.

## Reading order (this *is* the tour)

1. **`data/generate_data.py`** — read the module docstring carefully;
   it's the key to the whole project. Look at how `tabular_price` and
   `quality_price` are computed SEPARATELY and then summed — and how
   `QUALITY_PHRASES` (used to build `description`) is keyed by
   `quality_tier`, never by `bedrooms`/`bathrooms`/etc. This separation
   is what guarantees neither modality alone can explain all the
   variance in `price`.

2. **`src/data_prep.py`** — `build_tabular_features()` with its
   fixed-category encoding (same pattern as project 07's data_prep.py,
   for the same reason: single-row live inference must match
   training-time columns exactly).

3. **`src/train.py`** — the core of this project. Read in this order:
   - the module docstring's explanation of why all 3 models use the
     SAME algorithm (Ridge)
   - the `TfidfVectorizer(...).fit_transform(texts.iloc[idx_train])` —
     fit only on train, transformed (not re-fit) on test
   - the 3 model blocks — notice `model_fusion` is trained on
     `hstack([csr_matrix(X_tab_train), X_text_train])` — literally just
     concatenating the two feature matrices side by side into one wider
     matrix; that concatenation IS "multimodal fusion" at the feature
     level
   - the printed leaderboard — this is where you see fusion's R2 far
     exceed either single modality's

4. **`src/predict.py`** — `predict_price()`: builds a single listing's
   tabular and text features the same way training did, then scores it
   through all 3 saved models — returning all 3 predictions, not just
   the best one, specifically so the comparison is visible per-request.

5. **`app.py`** — thin FastAPI wiring; `Literal[...]` on `room_type` and
   `neighborhood` in `ListingRequest`.

6. **`static/app.js`** — `loadLeaderboard()` renders the bar chart with
   fusion visually distinguished (green); the predict form renders all
   3 predictions as adjacent cards, with fusion's card highlighted, so a
   viewer can directly see the "extra" the fusion model captures.

## How a request flows end-to-end

```
user submits a listing (tabular fields + description)
  → static/app.js POSTs to /api/predict
    → app.py: ListingRequest validates (including Literal category checks)
      → predict.py: predict_price()
        1. build_tabular_features() + scaler.transform() -- same as training
        2. vectorizer.transform([description]) -- same TF-IDF vocabulary as training
        3. score through model_tabular, model_text, and model_fusion
           (the last one on the hstack'd combined matrix)
      ← {tabular_only_prediction, text_only_prediction, fusion_prediction}
    ← JSON response
  ← static/app.js renders 3 side-by-side price cards
```

## Suggested shot list (if you want to record your own walkthrough video)

1. `python data/generate_data.py` running, show the sample rows with description + price
2. `python src/train.py` running, show the 3-model leaderboard printout — point out fusion's much higher R2
3. `uvicorn app:app --reload --port 8014` starting up
4. Browser: point at the leaderboard bar chart
5. Browser: submit a listing with a luxury-sounding description, note the 3 predictions; change ONLY the description to a budget-sounding one, resubmit — show tabular-only stays exactly the same while fusion (and text-only) shift
6. Terminal: `python -m pytest tests/` passing, especially the "tabular-only is blind to text" test
7. Code editor: 30 seconds each on `generate_data.py`'s docstring and the `hstack(...)` line in `train.py` — the two places that make the whole story work
