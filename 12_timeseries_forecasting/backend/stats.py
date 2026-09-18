"""
Autocorrelation (ACF) and partial autocorrelation (PACF), both implemented
from scratch — no statsmodels. ACF is the textbook lag-k sample
autocorrelation; PACF uses the Durbin-Levinson recursion, which builds each
order's partial autocorrelation from the previous order's AR coefficients
instead of fitting a separate regression per lag.
"""


def acf(values, max_lag=40):
    n = len(values)
    mean = sum(values) / n
    centered = [v - mean for v in values]
    denom = sum(c * c for c in centered)

    result = []
    for k in range(0, max_lag + 1):
        num = sum(centered[t] * centered[t - k] for t in range(k, n))
        result.append(num / denom if denom > 0 else 0.0)
    return result


def pacf(values, max_lag=40):
    r = acf(values, max_lag)  # r[0] == 1.0 by construction

    phi = {}  # phi[(k, j)] = j-th AR coefficient at order k
    pacf_values = [1.0]  # lag 0 is always 1

    phi[(1, 1)] = r[1]
    pacf_values.append(phi[(1, 1)])

    for k in range(2, max_lag + 1):
        numerator = r[k] - sum(phi[(k - 1, j)] * r[k - j] for j in range(1, k))
        denominator = 1 - sum(phi[(k - 1, j)] * r[j] for j in range(1, k))
        phi[(k, k)] = numerator / denominator if denominator != 0 else 0.0

        for j in range(1, k):
            phi[(k, j)] = phi[(k - 1, j)] - phi[(k, k)] * phi[(k - 1, k - j)]

        pacf_values.append(phi[(k, k)])

    return pacf_values
