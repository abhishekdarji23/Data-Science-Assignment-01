"""
test_api.py
-----------
Minimal smoke tests, plus tests for the properties that actually matter
for a time-series forecasting comparison: that naive is a genuinely hard
baseline (not artificially weakened), and that any method claiming to
beat it does so by a real, if modest, margin.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200


def test_leaderboard_has_all_5_methods():
    r = client.get("/api/leaderboard")
    assert r.status_code == 200
    body = r.json()
    assert set(body["leaderboard"].keys()) == {
        "naive", "moving_average", "exponential_smoothing", "arima", "gradient_boosting",
    }


def test_naive_is_a_hard_baseline_not_strawman():
    """A rigged/artificially-weak naive baseline would be a red flag for
    a forecasting comparison -- moving_average and gradient_boosting
    should NOT beat naive here (an honest, well-documented outcome for
    near-random-walk financial data), while ETS/ARIMA squeak out only a
    small edge, not a dramatic one."""
    r = client.get("/api/leaderboard")
    body = r.json()
    lb = body["leaderboard"]
    naive_mae = lb["naive"]["mae_usd"]

    # moving average is expected to be WORSE than naive (lags behind trend)
    assert lb["moving_average"]["mae_usd"] > naive_mae

    # any method that does beat naive should do so by a small, plausible
    # margin (<15%), not a suspiciously large one for daily price levels
    for name in ["exponential_smoothing", "arima"]:
        if lb[name]["mae_usd"] < naive_mae:
            improvement = (naive_mae - lb[name]["mae_usd"]) / naive_mae
            assert improvement < 0.15, f"{name} beat naive by an implausibly large margin: {improvement:.1%}"


def test_chart_data_has_test_period_length():
    leaderboard = client.get("/api/leaderboard").json()
    r = client.get("/api/chart")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == leaderboard["n_test"]
    for row in rows:
        assert "actual" in row and "naive_pred" in row


def test_forecast_returns_all_5_methods_and_plausible_values():
    r = client.post("/api/forecast")
    assert r.status_code == 200
    body = r.json()
    forecasts = body["forecasts"]
    assert set(forecasts.keys()) == {
        "naive", "moving_average", "exponential_smoothing", "arima", "gradient_boosting",
    }
    # naive forecast must exactly equal the last known close (that's its whole definition)
    assert forecasts["naive"] == body["last_close"]
    # every forecast should be within a plausible range of the last close (no wild extrapolation)
    for name, price in forecasts.items():
        assert abs(price - body["last_close"]) / body["last_close"] < 0.10
