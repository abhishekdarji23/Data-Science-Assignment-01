# SKILLS.md — Skills Demonstrated / Applied in This Build

| # | Skill | Where applied |
|---|---|---|
| 1 | Genuine 1-step-ahead time series evaluation (no leakage, no compounding) | `src/train.py` — every method's forecast at test step t uses only true data through t-1; ETS/ARIMA are refit at each step on real history, never on their own prior forecasts |
| 2 | Choosing the right baseline, and not rigging it | `PROMPTS.md` / `AUDIT_REPORT.md` — naive persistence is a real, hard-to-beat baseline for financial data, not a strawman; `tests/test_api.py` explicitly checks this isn't artificially weakened |
| 3 | Model selection via information criterion, on training data only | `src/train.py` — ARIMA order chosen by AIC grid search restricted to `train_series`, never touching test data |
| 4 | Verifying a synthetic-data property empirically, not just by construction | `data/generate_data.py` + `AUDIT_REPORT.md` section 1 — the mean-reversion signal's sample autocorrelation was checked and the generator's strength parameter adjusted after finding it could flip sign due to sampling noise |
| 5 | Leak-free lag feature engineering | `src/data_prep.py: build_lag_features()` — every feature shifted to use only past values |
| 6 | Reporting an honest negative result instead of reframing it | `PROMPTS.md` "honest headline result" — moving average and gradient boosting both losing to naive is reported plainly, with the well-documented reason why |
| 7 | Practical feasibility checking before a potentially-slow operation | `IMPLEMENTATION_PLANS.md` section 7 — single-fit timing was measured before committing to a ~225-step refit loop for 2 models |
| 8 | Testing a statistical property with a plausibility bound, not just an exact value | `tests/test_api.py: test_naive_is_a_hard_baseline_not_strawman` — any method beating naive must do so by <15%, catching an implausibly large "win" as a bug signal rather than accepting it |
| 9 | Documentation-as-artifact | This file, plus `PROMPTS.md`, `IMPLEMENTATION_PLANS.md`, `WALKTHROUGH.md`, `AUDIT_REPORT.md`, `README.md` |

## Explicitly not attempted here (see `IMPLEMENTATION_PLANS.md` section 8)

A deep-learning forecasting model (LSTM/GRU/Transformer), real market
data, multi-step recursive forecasting, and walk-forward periodic
refitting of the GBM model were all plausible parts of a full "SOTA"
build but were intentionally left out — per your explicit instruction
to keep this "not fancy," with the main aim being a simple, working
frontend and backend.
