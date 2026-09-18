# WALKTHROUGH.md — Architectural Deep-Dive

> No video walkthrough is included with this build: recording and uploading
> a narrated video (e.g. to YouTube) isn't something this assistant can do —
> there's no video/audio generation or YouTube-upload capability available.
> This document is the substitute: a code-ordered tour you (or a screen
> recorder) can follow to produce one in a few minutes if you want it, plus
> a suggested shot list at the bottom.

## Reading order (this *is* the tour)

1. **`data/generate_data.py`** — where the data comes from. Read the
   module docstring first, then `BAKED_IN_RULES`: a small, explicit list
   of (antecedent, consequent, strength) tuples — e.g. `diapers → beer`
   at strength 0.40 — used to boost co-purchase probability on top of
   each item's independent baseline rate (`BASELINE_P`). This is what
   makes the mining exercise non-trivial: a purely random basket
   generator would produce no genuine patterns to find.

2. **`src/data_prep.py`** — `build_basket_matrix()`: pivots long-format
   (transaction_id, item) rows into the one-hot basket matrix Apriori
   needs. Small function, but it's the shape-conversion step every
   market-basket pipeline needs.

3. **`src/train.py`** — the core of this project. Read the module
   docstring for the support/confidence/lift definitions, then:
   - `apriori(basket, min_support=...)` — frequent itemsets
   - `association_rules(..., metric="confidence", ...)` — turns itemsets
     into directional antecedent→consequent rules
   - the `rules[rules["lift"] >= MIN_LIFT]` filter and
     `sort_values("lift", ...)` — keeps only genuine associations, ranked
   - the printed "top 5 rules by lift" — this is where you can see the
     baked-in diapers→beer/wipes pattern surface with lift ≈ 5–8

4. **`src/predict.py`** — `recommend()`: given a cart (set of items),
   finds every rule whose antecedent is a *subset* of the cart, collects
   consequent items not already in the cart, and for each keeps the
   highest-lift rule that recommends it. This is a lookup over
   precomputed rules, not a live model — the mining happened offline in
   `train.py`.

5. **`app.py`** — FastAPI wiring: `/api/recommend` calls
   `predict.recommend()`; `/api/rules` and `/api/metrics` expose the
   mined rules and dataset stats for the dashboard; `/api/items` serves
   the catalog for the cart-picker UI.

6. **`static/app.js`** — `renderItemGrid()` draws clickable item chips;
   `updateCartAndRecommendations()` POSTs the current cart to
   `/api/recommend` on every click and renders the returned
   recommendation cards with their "because you have: X" explanation.

## How a request flows end-to-end

```
user clicks "diapers" chip
  → static/app.js adds "diapers" to the in-browser cart set
    → POSTs {items: ["diapers"]} to /api/recommend
      → app.py: CartRequest validates (min 1 item)
        → predict.py: recommend() finds rules where {"diapers"} ⊆ antecedents
          → matches: diapers→beer, diapers→wipes, diapers→[beer,wipes]
        ← ranked recommendations by lift, each with "because_of"
      ← JSON response
  ← static/app.js renders "beer" and "wipes" recommendation cards
```

## Suggested shot list (if you want to record your own walkthrough video)

1. `python data/generate_data.py` running, show the printed basket-size stats
2. `python src/train.py` running, show the printed top-5-by-lift rules — point out diapers→beer/wipes
3. `uvicorn app:app --reload --port 8004` starting up
4. Browser: click "diapers", show the beer/wipes recommendation cards appear with lift/confidence shown
5. Browser: click "bread", show butter recommended
6. Browser: scroll to the top-rules table and the model card
7. Terminal: `python -m pytest tests/` passing, especially the diapers→beer/wipes regression test
8. Code editor: 30 seconds each on `generate_data.py`'s `BAKED_IN_RULES`, `train.py`, `predict.py` — the reading order above
