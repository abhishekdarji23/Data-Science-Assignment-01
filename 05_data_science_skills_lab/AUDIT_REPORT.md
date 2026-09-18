# AUDIT_REPORT.md — Correctness & Ground-Truth Recovery Audit

**Scope**: Project 05, Data Science Skills Mastery Lab.
**Method**: Static inspection + scripted checks run against this exact codebase (commands shown below so you can re-run them yourself).

Because the synthetic dataset has deliberately known properties (exact
missing-value counts, exact injected duplicates, and a survival model
with known drivers), this audit's main technique is checking that each
skill's output matches the known ground truth — a stronger check than
"the code ran without an exception."

## 1. Missing-value skill matches injected ground truth

**Check**: `generate_data.py` injects missing values into `Age` (~20%
of rows) and `Embarked` (~0.5% of rows). `missing_values` should report
exactly this.

```
$ python3 -c "... get_skill('missing_values').fn(df) ..."
[['Age', 123, '20.2%'], ['Embarked', 3, '0.5%']]
```

**Result: PASS.** 123/608 ≈ 20.2% for Age, 3/608 ≈ 0.5% for Embarked —
matches the injection rates in `generate_data.py` (`0.20` and `0.005`).

## 2. Duplicate-detection skill matches injected ground truth

**Check**: `generate_data.py` injects exactly 8 duplicate rows
(`N_INJECTED_DUPLICATES = 8`). Since `duplicate_rows` uses
`keep=False` (flags *both* the original and the copy), it should report
16 rows.

```
$ python3 -c "... get_skill('duplicate_rows').fn(df) ..."
16 (expect 16 = 8 injected x 2 copies)
```

**Result: PASS.**

## 3. Feature importance recovers the known survival drivers

**Check**: the survival-probability model in `generate_data.py` gives
the largest boost to `Sex == "female"` (+0.45), a smaller one to
`Pclass == 1` (+0.22), and a smaller one still to `Age < 12` (+0.18).
The trained model's feature importance should roughly reflect that
ordering (Sex strongest).

```
$ python3 -c "... get_skill('feature_importance').fn(df) ..."
[('Sex_male', 1.189), ('Pclass_3', 0.447), ('Age', 0.277)]
```

**Result: PASS.** `Sex_male` (the encoded inverse of the strongest
survival driver) has by far the largest coefficient magnitude, consistent
with Sex being the dominant signal baked into the data generator.

## 4. The trained model genuinely beats the baseline

**Check**: since real signal exists in the data (by construction), a
logistic regression should outperform a majority-class baseline. If it
didn't, that would indicate a bug in feature preparation or model
fitting, not just a weak dataset.

```
$ python3 -c "... get_skill('baseline_vs_model').fn(df) ..."
{'Majority-class baseline': 0.541, 'Logistic Regression': 0.6803}
```

**Result: PASS.** +13.9 percentage points over baseline — a real,
non-trivial lift.

## 5. Every registered skill executes without error

**Check**: every skill in `SKILL_REGISTRY` must run successfully against
the current dataset — a broken skill would otherwise only surface when a
user happened to click it in the UI.

**Result: PASS.** Verified two ways: a standalone script iterating
`SKILL_REGISTRY` directly (16/16 succeeded), and
`tests/test_api.py: test_execute_every_registered_skill`, which loops
over the *live* `/api/skills` catalog through the real HTTP API — so this
check automatically covers any skill added later, not just the 16
present today.

## 6. Consistent feature engineering across modeling/evaluation skills

**Check**: `train_test_split`, `baseline_vs_model`, `cross_validation`,
`confusion_matrix`, and `feature_importance` all train on features built
by the same function, `_prepare_model_features()` in `src/skills.py` —
not five separately-written (and potentially inconsistent) feature-prep
blocks.

**Result: PASS** (by inspection — all five skill functions call
`_prepare_model_features(df)` as their first step).

## 7. Known limitation (disclosed, not hidden)

The underlying dataset is **synthetic**, and — as with project 04 — that
is deliberate here: it's what makes checks #1–#4 above possible at all.
This audit confirms the skills correctly compute what they claim to on
data with known properties; it does not confirm how each skill's specific
numeric output would look on the real Kaggle Titanic dataset (missingness
patterns, survival correlations, and outlier counts would differ, though
the pipeline requires no code changes to run against real data — see
`data/generate_data.py`'s docstring for the swap-in path).

## Summary

| Check | Result |
|---|---|
| Missing-value skill matches injected ground truth | PASS |
| Duplicate-detection skill matches injected ground truth | PASS |
| Feature importance recovers known survival drivers | PASS |
| Trained model beats baseline (real signal exists) | PASS |
| Every registered skill executes without error | PASS (16/16, twice over) |
| Consistent feature engineering across modeling skills | PASS |
| Data realism | Disclosed limitation — synthetic, deliberately so |

**Overall**: every checkable claim this lab makes about its own data was
independently verified against known ground truth, not just assumed from
the code running without an exception.
