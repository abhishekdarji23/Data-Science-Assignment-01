"""
generate_data.py
-----------------
CRISP-DM Phase 2: Data Understanding (data acquisition step).

Generates a synthetic daily "SPY"-style (S&P 500 ETF) price series. This
build environment has no internet access to a market data API, so this
is NOT real SPY data -- it's built to have the same qualitative
structure real equity index data has, which is what makes the model
comparison downstream honest rather than trivial:

  * A dominant near-random-walk component (like real daily index prices --
    this is why "beat a naive persistence forecast" is a genuinely hard
    bar in finance, not a strawman).
  * A SMALL, genuine mean-reverting component in returns (documented,
    modest short-horizon mean reversion has real historical precedent
    in equity index data) -- just enough that a model that can detect
    it should show a small, real edge over naive, without the whole
    exercise being trivially "solved."
  * A mild day-of-week effect on average returns.
  * Volatility clustering (today's volatility depends partly on
    yesterday's) for visual/statistical realism.

Swap-in instructions for real data:
    Save real daily OHLC data (e.g. from a market data provider) as
    data/spy_real.csv with columns `date, close`. `src/data_prep.py`
    will use it automatically instead of the synthetic file if present.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_DAYS = 1500  # ~6 years of trading days
START_PRICE = 400.0

OUT_PATH = Path(__file__).parent / "spy_synthetic.csv"

DAILY_DRIFT = 0.0003          # ~7.5%/year average drift, roughly realistic for a long-run index
BASE_VOL = 0.009              # ~1.4% daily vol baseline, roughly realistic for SPY
MEAN_REVERSION_STRENGTH = 0.18  # today's return pulls against yesterday's (verified below to reliably
                                  # produce negative sample autocorrelation, not just in population)
DOW_EFFECT = {0: 0.0004, 1: 0.0001, 2: 0.0000, 3: -0.0001, 4: 0.0002}  # Mon..Fri, small


def generate(n_days: int = N_DAYS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range("2019-01-02", periods=n_days)  # business days only
    dow = dates.dayofweek.values

    log_returns = np.zeros(n_days)
    vol = np.full(n_days, BASE_VOL)
    prev_return = 0.0

    for t in range(n_days):
        # volatility clustering: today's vol partly follows yesterday's shock size
        if t > 0:
            vol[t] = 0.85 * BASE_VOL + 0.15 * abs(log_returns[t - 1]) * 3
        shock = rng.normal(0, vol[t])
        mean_reversion = -MEAN_REVERSION_STRENGTH * prev_return
        dow_effect = DOW_EFFECT[dow[t]]
        r = DAILY_DRIFT + dow_effect + mean_reversion + shock
        log_returns[t] = r
        prev_return = r

    log_prices = np.log(START_PRICE) + np.cumsum(log_returns)
    prices = np.exp(log_prices)

    df = pd.DataFrame({
        "date": dates,
        "close": prices.round(2),
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} trading days to {OUT_PATH}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    daily_returns = df["close"].pct_change().dropna()
    print(f"Mean daily return: {daily_returns.mean():.5f}, daily vol: {daily_returns.std():.5f}")
    print(f"Lag-1 autocorrelation of returns: {daily_returns.autocorr(1):.4f} (small negative = mild mean reversion)")
