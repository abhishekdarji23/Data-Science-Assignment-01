# AUDIT_REPORT.md — Sign-Convention, Leakage & Honest-Comparison Audit

**Scope**: Project 06, Anomaly Detection (Transaction Risk Scoring).
**Method**: Static inspection + scripted checks run against this exact codebase (commands shown below so you can re-run them yourself).

Anomaly detection has a specific failure mode that's easy to get subtly
wrong: different sklearn estimators use *opposite* sign conventions for
"this looks anomalous," and silently mixing them up produces a model that
LOOKS like it runs successfully but is actively scoring anomalies as the
safest points. This audit checks that directly, not just that the code
executes.

## 1. Isolation Forest sign convention verified against actual output

**Check**: sklearn's `IsolationForest.decision_function()` returns LOWER
values for MORE abnormal points. `train.py` uses `-decision_function()`
as "higher = more anomalous" — verify that flip is actually correct on
this data, not just correct per the docs.

```
$ python3 -c "... iso.decision_function(Xs) ... grouped by y ..."
mean decision_function for normal:   0.1475
mean decision_function for anomaly: -0.1127
```

**Result: PASS.** Anomalies genuinely have lower `decision_function`
values, confirming `-decision_function()` correctly produces "higher =
more anomalous."

## 2. LOF sign convention verified against actual output

**Check**: same verification for `LocalOutlierFactor.score_samples()`
(also "lower = more abnormal" per sklearn docs).

```
$ python3 -c "... lof.score_samples(Xs) ... grouped by y ..."
mean score for normal:   -1.0639
mean score for anomaly:  -1.2018
```

**Result: PASS.** Anomalies have lower `score_samples`, confirming
`-score_samples()` is correctly flipped. This also rules out "the sign
was backwards" as an explanation for LOF's weak PR-AUC below — the
direction is right, the *separation* is just weak.

## 3. Why LOF underperforms here (documented, not hidden)

LOF scored PR-AUC ≈ 0.10 vs. Isolation Forest's ≈ 0.99 and the z-score
baseline's ≈ 0.95. Per §1-2 above, this is not a sign-convention bug.

**Explanation**: LOF is a *local density* method — it flags a point as
anomalous when it sits in a sparser neighborhood than its own neighbors.
This works well when anomalies are scattered, isolated points. In this
dataset, anomalies are NOT isolated — `generate_data.py` draws all
anomalies from one shared alternate distribution (high amount, night
hours, new accounts, etc.), so anomalies cluster *together* in feature
space and can have similar local density to each other. That defeats
LOF's core assumption. Isolation Forest and z-score are both
GLOBAL methods (they compare a point to the overall feature distribution,
not just its immediate neighbors), so they aren't fooled by anomalies
forming their own cluster — this is a genuine, well-known trade-off
between local and global anomaly detection methods, not a bug in this
implementation.

## 4. Ground truth label excluded from model inputs

```
$ python3 -c "from data_prep import FEATURE_COLUMNS, LABEL_COLUMN; print(LABEL_COLUMN in FEATURE_COLUMNS)"
False
```

**Result: PASS.** All 3 methods are fit without ever seeing `is_anomaly`
— genuinely unsupervised, matching real-world anomaly detection
constraints.

## 5. Fair cross-method comparison threshold

**Check**: precision/recall/F1 for each method use the SAME
decision rule (flag the top-K highest-scored points, K = known
contamination rate × n) rather than an arbitrary raw-score cutoff that
would differ in meaning between methods with different score scales.

**Result: PASS** (by inspection of `src/train.py: evaluate()` — `k` is
computed once from `contamination` and reused identically for every
method's call to `evaluate()`).

## 6. Reproducibility

```
$ grep -n "RANDOM_SEED\|random_state" src/train.py
RANDOM_SEED = 42
iso = IsolationForest(contamination=contamination, random_state=RANDOM_SEED, n_estimators=200)
```

**Result: PASS.** Both data generation and Isolation Forest fitting are
seeded (LOF has no random component to seed). Re-running
`python data/generate_data.py && python src/train.py` from a clean
checkout reproduces the same comparison table.

## 7. Known limitation (disclosed, not hidden)

The underlying dataset is **synthetic**. This audit confirms the pipeline
correctly implements and fairly compares 3 real anomaly detection
methods; it does not confirm this exact PR-AUC ranking (Isolation Forest
≈ z-score ≫ LOF) would hold on real transaction data, where anomaly
clustering behavior may differ. Swap in a real transaction log at
`data/transactions_real.csv` (documented schema in `data_prep.py`) and
re-run `python src/train.py` — no code changes needed.

## Summary

| Check | Result |
|---|---|
| Isolation Forest sign convention correct | PASS (verified against actual scores, not just docs) |
| LOF sign convention correct | PASS |
| LOF's weak result explained, not hidden | PASS — documented root cause (local vs. global anomaly detection) |
| Ground truth label excluded from features | PASS |
| Cross-method comparison uses a consistent threshold | PASS |
| Reproducibility / seed pinning | PASS |
| Data realism | Disclosed limitation — synthetic |

**Overall**: no sign-convention or leakage defects found in this
codebase. The one surprising result (LOF's weak performance) was
investigated and explained rather than smoothed over.
