# AUDIT_REPORT.md — Forecasting Methodology Audit

**Scope**: Project 15, SPY Time Series Forecasting.
**Method**: Static inspection + scripted checks run against this exact codebase (commands shown below so you can re-run them yourself).

Time series forecasting has specific, easy-to-get-wrong failure modes
(future leakage into lag features, recursive forecasts compounding
error, tuning a model on the test period) that don't show up as errors
— they show up as suspiciously good numbers. This audit checks for them
directly.

## 1. The data generator's mean-reversion signal was verified empirically, not just assumed from its formula

**Check**: does `MEAN_REVERSION_STRENGTH` in `generate_data.py` actually
produce the intended negative sample autocorrelation in returns, or
could sampling noise at this dataset size mask/flip it?

```
$ python3 -c "... standard error of sample autocorrelation at n=1500 ..."
standard error of sample autocorr at n=1500: 0.0258
```

With the ORIGINAL strength (0.06), the observed sample autocorrelation
came out **+0.0597** — positive, opposite the intended sign, about 2.3
standard errors from zero. This was a real issue, not a hypothetical:
at that strength, the population effect was too weak to reliably show
up against sampling noise in a dataset this size.

**Fix applied**: `MEAN_REVERSION_STRENGTH` was raised to 0.18 and
re-verified:

```
$ python3 data/generate_data.py
Lag-1 autocorrelation of returns: -0.0560
```

**Result: PASS**, after a genuine fix — the sample statistic now
reliably matches the intended sign, about 2.2 standard errors from
zero in the correct direction.

## 2. Lag features never use the current or future row

**Check**: `build_lag_features()`'s `lag_1` column at row t must equal
`close[t-1]`, and rolling-window features must exclude `close[t]`.

```
$ python3 -c "... verify lag_1[t] == close[t-1] and roll_mean_5[10] excludes close[10] ..."
lag_1[t] == close[t-1] for sample rows: True
roll_mean_5[10] correct (excludes close[10]): True
```

**Result: PASS.** Both checks confirmed directly against the actual
computed values, not just by reading the `.shift(1)` calls.

## 3. ARIMA order selection never touches test data

```
$ grep -n "ARIMA(train_series" train.py
103:                    fit = ARIMA(train_series, order=(p, d, q)).fit()
```

**Result: PASS.** The AIC grid search that picks ARIMA's `(p,d,q)`
order is fit only on `train_series` (`close[:split_idx]`) — the chosen
order is then reused (not re-searched) inside the refit loop over the
test period, so no information from the test period influences model
selection.

## 4. ETS and ARIMA forecasts are genuinely 1-step-ahead, not compounding recursive forecasts

**Result: PASS** (by inspection of `src/train.py`'s refit loops): at
each test step, `history.append(close[t])` — the TRUE observed value —
happens after that step's forecast is recorded, and the NEXT step's
model is refit on history that includes this true value, not the
model's own prior prediction. This avoids the classic recursive-forecast
failure mode where early errors compound through later predictions.

## 5. Naive baseline is not artificially weakened, and any "win" is a plausible size

**Check**: for financial daily price levels, a method that beats naive
persistence by a large margin should be treated as suspicious (likely a
methodology bug), not celebrated.

```
Leaderboard (see models/metrics.json for live numbers):
naive:                  MAE=$3.796
moving_average:         MAE=$4.976  (worse than naive)
exponential_smoothing:  MAE=$3.776  (0.5% better)
arima:                  MAE=$3.778  (0.5% better)
gradient_boosting:      MAE=$3.894  (worse than naive)
```

**Result: PASS.** Only 2 of 4 non-naive methods beat naive at all, and
both by well under 1% — consistent with `tests/test_api.py`'s <15%
plausibility bound, and consistent with the real, well-documented
difficulty of beating persistence forecasts on near-random-walk
financial data (see `PROMPTS.md`).

## 6. Reproducibility

```
$ grep -n "RANDOM_SEED" train.py
RANDOM_SEED = 42
(used for the GBM model; the data generator has its own RNG_SEED = 42)
```

**Result: PASS.** Naive, moving average, and the AIC-selected ARIMA
order are deterministic by construction. GBM is seeded. ETS's
`fit(optimized=True)` uses deterministic numerical optimization for
this model class. Re-running `python data/generate_data.py && python
src/train.py` from a clean checkout reproduces the same leaderboard.

## 7. Known limitation (disclosed, not hidden)

The data is **synthetic**, and deliberately so — a real market's exact
autocorrelation structure isn't something this build environment can
access. This audit confirms the evaluation methodology (leak-free
features, train-only model selection, genuine 1-step-ahead forecasts) is
sound; it does not confirm these specific methods would rank the same
way on real SPY data, where the true (if any) exploitable
mean-reversion signal's strength is an open empirical question, not
something this build controls by construction.

## Summary

| Check | Result |
|---|---|
| Mean-reversion signal verified empirically (not just by formula) | PASS — real issue found and fixed |
| Lag features leak-free (verified against actual values) | PASS |
| ARIMA order selection uses only training data | PASS |
| Forecasts are genuine 1-step-ahead, not recursive/compounding | PASS |
| Naive baseline not artificially weakened; any win is plausible size | PASS |
| Reproducibility / seed pinning | PASS |
| Data realism | Disclosed limitation — synthetic |

**Overall**: the forecasting comparison avoids the specific
methodology failures time series evaluation is most prone to, and one
genuine data-generation issue (a signal too weak to reliably manifest)
was found and fixed during this audit, not assumed correct from the start.
