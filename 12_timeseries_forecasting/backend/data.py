"""
Generates a synthetic daily time series: a linear trend, a 7-day
(weekly) seasonal pattern, and random noise — deterministic via a seed,
so no external dataset is needed.
"""

import random
import math
from datetime import date, timedelta

DAY_OF_WEEK_EFFECT = [0, -2, -1, 1, 2, 6, 4]  # Mon..Sun, weekends busier


def generate_series(num_days=730, seed=11):
    rng = random.Random(seed)
    start = date(2024, 1, 1)

    series = []
    for i in range(num_days):
        d = start + timedelta(days=i)
        trend = 50 + i * 0.08
        seasonal = DAY_OF_WEEK_EFFECT[d.weekday()]
        # a slow annual wave on top of the weekly one, so it's not perfectly periodic
        annual = 4 * math.sin(2 * math.pi * i / 365)
        noise = rng.gauss(0, 3)
        value = round(max(0.0, trend + seasonal + annual + noise), 2)
        series.append({"date": d.isoformat(), "value": value})

    return series
