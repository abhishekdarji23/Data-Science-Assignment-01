# 📈 SPY Time Series Forecasting

A small, real, honest comparison of 5 forecasting methods — naive
persistence, moving average, exponential smoothing, ARIMA, and gradient
boosting — on a daily SPY-style (S&P 500 ETF) price series, evaluated
with genuine 1-step-ahead backtesting.

Reproduced (scoped down, "not fancy," simple frontend + backend) from
Project 15 of
[`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples).
**Note**: this project's exact original prompt could not be retrieved
(same GitHub access restriction as projects 13-14) — see
[`PROMPTS.md`](PROMPTS.md) for what's known and what's this build's own
interpretation, including why the real (deep-learning-based) "SOTA"
forecasting stack was deliberately not used.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt

# 1. Generate data (synthetic — see data/generate_data.py to swap in real market data)
python data/generate_data.py

# 2. Train and compare all 5 methods (~30s — refits ARIMA/ETS at each backtest step)
cd src && python train.py && cd ..

# 3. Run the app
uvicorn app:app --reload --port 8015
```

Open **http://127.0.0.1:8015** — see the leaderboard, the test-period
actual-vs-naive chart, and get a live next-day forecast from all 5 methods.

Run the tests any time with:
```bash
python -m pytest tests/
```

## What's in the box

```
15_spy_timeseries_sota_forecasting/
├── README.md                  ← you are here
├── PROMPTS.md                 ← what's known about the original ask, honestly caveated
├── IMPLEMENTATION_PLANS.md    ← architecture & technical spec
├── WALKTHROUGH.md             ← code-ordered deep-dive tour (text, see note below)
├── SKILLS.md                  ← practices applied, with file/line pointers
├── AUDIT_REPORT.md            ← forecasting-methodology audit, with a real bug found & fixed
├── requirements.txt
├── data/
│   └── generate_data.py       ← synthetic SPY-style price generator (near-random-walk + small mean reversion)
├── src/
│   ├── data_prep.py           ← leak-free lag/rolling feature engineering
│   ├── train.py                ← trains & compares 5 methods via honest 1-step-ahead backtesting
│   └── predict.py             ← live next-day forecast from all 5 methods
├── app.py                     ← FastAPI backend
├── static/                    ← plain HTML/CSS/JS frontend (leaderboard, chart, live forecast)
├── models/                    ← GBM model, metrics.json, chart data
└── tests/
    └── test_api.py            ← automated tests (5/5 passing)
```

## The honest headline result

| Method | MAE | Beats naive? |
|---|---|---|
| Naive persistence | $3.80 | — |
| Moving average (5d) | $4.98 | ❌ worse |
| Exponential smoothing | $3.78 | ✅ by 0.5% |
| ARIMA | $3.78 | ✅ by 0.5% |
| Gradient boosting | $3.89 | ❌ worse |

This is not a disappointing result to smooth over — it's the real,
well-documented finding in financial forecasting: daily price levels
are close enough to a random walk that naive persistence is extremely
hard to beat, and a "fancier" ML method doesn't automatically win. See
`PROMPTS.md` and `AUDIT_REPORT.md` for the full picture, including a
genuine bug (a too-weak signal that could flip sign in the data due to
sampling noise) found and fixed while verifying this.

## Honest limitations of this build

- **This project's exact original prompt is unknown** — see `PROMPTS.md`.
- **No deep-learning forecasting model.** Real "SOTA" time series work
  today usually means LSTM/Transformer architectures requiring PyTorch —
  a heavy dependency that conflicts with "keep it simple." This build
  compares 5 real classical/ML methods instead.
- **Data is synthetic**, because this build environment has no
  internet/market-data-API access. Drop real data at
  `data/spy_real.csv` (columns: `date, close`) and re-run — no code
  changes needed.
- **No video walkthrough is included.** `WALKTHROUGH.md` is a written,
  code-ordered substitute, with a shot list at the bottom.
- **This zip wasn't pushed to GitHub for you.** To publish it yourself:
  ```bash
  cd 15_spy_timeseries_sota_forecasting
  git init && git add . && git commit -m "SPY time series forecasting comparison"
  git branch -M main
  git remote add origin https://github.com/<you>/<repo>.git
  git push -u origin main
  ```
- Deliberately **simple**, per your instruction: no deep learning, no
  multi-step recursive forecasting, no database — see
  `IMPLEMENTATION_PLANS.md` §8 for the full list of what was left out and why.

## License

MIT — do whatever you like with it.
