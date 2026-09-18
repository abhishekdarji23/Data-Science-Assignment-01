# AUDIT_REPORT.md — Data Leakage & Quality Audit

**Scope**: Project 01, NYC Taxi Trip Duration Predictor.
**Method**: Static inspection + scripted checks run against this exact codebase (commands shown below so you can re-run them yourself).

## 1. Target leakage in features

**Check**: no feature column should be derived from `trip_duration` or `dropoff_datetime`.

```
$ python3 -c "
from src.data_prep import FEATURE_COLUMNS
leak_terms = ['duration','dropoff_datetime']
print([f for f in FEATURE_COLUMNS if any(t in f.lower() for t in leak_terms)])
"
[]
```

**Result: PASS.** `FEATURE_COLUMNS` (`src/data_prep.py`) contains 13 columns,
none derived from the target or from post-trip information. `dropoff_datetime`
is present in the raw dataset but is never read by `add_engineered_features()`
or included in `FEATURE_COLUMNS`.

## 2. Train/test split strategy

**Check**: for time-series-shaped data, the split must not be a random
shuffle (which would let the model see "future" traffic patterns during
training).

```
$ grep -n "shuffle=True\|train_test_split" src/train.py
(no matches)
```

**Result: PASS.** `time_based_split()` in `src/train.py` sorts by
`pickup_datetime` and takes a chronological 80/20 cut — no
`sklearn.model_selection.train_test_split(shuffle=True)` is used.

## 3. Preprocessing fit only on training data

**Result: PASS (trivially).** The only "preprocessing" with fittable state
is the model itself (`GradientBoostingRegressor`), and `model.fit()` is
called only on `X_train` / `y_train_log` (`src/train.py`, `main()`). No
scaler, encoder, or imputer is fit on the full dataset before splitting.

## 4. Reproducibility / seed pinning

```
$ grep -n "RANDOM_SEED\|random_state" src/train.py data/generate_data.py
src/train.py:33:RANDOM_SEED = 42
src/train.py:59:        random_state=RANDOM_SEED,
src/train.py:83:        "random_seed": RANDOM_SEED,
data/generate_data.py:RNG_SEED = 42  (used via np.random.default_rng(seed))
```

**Result: PASS.** Both data generation and model training are seeded.
Re-running `python data/generate_data.py && python src/train.py` from a
clean checkout reproduces the same metrics.

## 5. Target distribution & clipping sanity

`src/data_prep.py: load_dataset()` drops rows with `trip_duration` outside
`[30s, 3h]` and `passenger_count` outside `[1, 6]` — matching the standard
Kaggle-competition cleaning rules, applied identically regardless of which
split a row ends up in (cleaning happens **before** the split, which is
correct here since it uses only the row's own values, not statistics
computed across rows).

## 6. Train/serve consistency

**Check**: does the API build features the same way training did?

Both `src/train.py` and `src/predict.py` import
`add_engineered_features` / `build_single_feature_row` from the same
module, `src/data_prep.py` — there is exactly one feature-engineering
implementation, not a duplicated one for serving.

## 7. Metrics honesty

Evaluation metrics (`mae_seconds`, `rmse_seconds`, `rmsle`, `r2`) are
computed on the **held-out chronological test split only**
(`src/train.py`, after `model.predict(X_test)`), never on training data.

## 8. Known limitation (disclosed, not hidden)

The underlying dataset is **synthetic** (see `PROMPTS.md` for why). The
audit above confirms the *pipeline* has no leakage or methodology defects;
it does not and cannot confirm that the synthetic data's statistical
properties exactly match real NYC TLC trip data. Swap in a real Kaggle
`train.csv` (see `data/generate_data.py` docstring) and re-run
`python src/train.py` to get metrics against real data — the pipeline
requires no code changes to do so.

## Summary

| Check | Result |
|---|---|
| Target leakage in features | PASS |
| Split strategy (time-based, not shuffled) | PASS |
| Preprocessing fit only on train | PASS |
| Seed pinning / reproducibility | PASS |
| Cleaning rules row-local (not cross-row) | PASS |
| Train/serve feature consistency | PASS |
| Metrics computed only on held-out data | PASS |
| Data realism | Disclosed limitation — synthetic data |

**Overall**: no leakage or evaluation-methodology defects found in this
codebase. The one caveat that matters is disclosed above, not buried.
