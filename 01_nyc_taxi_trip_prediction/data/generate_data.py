"""
generate_data.py
-----------------
CRISP-DM Phase 2: Data Understanding (data acquisition step)

This build environment has no internet access to Kaggle, so this script
generates a SYNTHETIC dataset that mirrors the schema and statistical
shape of the real "New York City Taxi Trip Duration" Kaggle dataset
(https://www.kaggle.com/c/nyc-taxi-trip-duration).

Schema (matches the real Kaggle competition exactly):
    id, vendor_id, pickup_datetime, dropoff_datetime, passenger_count,
    pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
    store_and_fwd_flag, trip_duration

Swap-in instructions for real data:
    Download `train.csv` from the Kaggle competition above and drop it in
    this `data/` folder as `nyc_taxi_train.csv`. `src/data_prep.py` will
    use it automatically instead of the synthetic file if it is present.

Trip duration is generated from a physically-plausible model:
    duration = haversine_distance / speed(hour, borough_congestion) + noise
so that the downstream regression task is genuine (a model that ignores
distance/time features will score poorly), not leaking the label into
any input feature.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_ROWS = 60_000

# Rough NYC bounding box (Manhattan + surrounding boroughs)
LAT_MIN, LAT_MAX = 40.63, 40.86
LON_MIN, LON_MAX = -74.03, -73.77

OUT_PATH = Path(__file__).parent / "nyc_taxi_synthetic.csv"


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8  # earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def generate(n_rows: int = N_ROWS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Pickup locations biased toward Manhattan core (dense center, sparse edges)
    pickup_lat = rng.normal(40.75, 0.045, n_rows).clip(LAT_MIN, LAT_MAX)
    pickup_lon = rng.normal(-73.98, 0.045, n_rows).clip(LON_MIN, LON_MAX)

    # Dropoff = pickup + a random offset (short trips more common than long ones)
    trip_radius = rng.exponential(scale=0.02, size=n_rows)  # degrees, ~ skewed short trips
    angle = rng.uniform(0, 2 * np.pi, n_rows)
    dropoff_lat = (pickup_lat + trip_radius * np.sin(angle)).clip(LAT_MIN, LAT_MAX)
    dropoff_lon = (pickup_lon + trip_radius * np.cos(angle)).clip(LON_MIN, LON_MAX)

    # Pickup timestamps across one synthetic year, minute-level resolution
    start = pd.Timestamp("2023-01-01")
    minutes_offset = rng.integers(0, 365 * 24 * 60, n_rows)
    pickup_dt = start + pd.to_timedelta(minutes_offset, unit="m")

    hour = pickup_dt.hour.values
    dow = pickup_dt.dayofweek.values  # 0=Mon
    is_weekend = (dow >= 5).astype(int)
    is_rush_hour = (((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 19))).astype(int)

    passenger_count = rng.choice([1, 1, 1, 2, 2, 3, 4, 5, 6], size=n_rows)
    vendor_id = rng.choice([1, 2], size=n_rows)
    store_and_fwd_flag = rng.choice(["N", "Y"], size=n_rows, p=[0.995, 0.005])

    distance_mi = haversine_miles(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    # Speed model: base speed drops during rush hour / weekday congestion,
    # plus per-trip noise. This is the "physics" the model has to learn.
    base_speed = np.where(is_rush_hour == 1, 9.5, 14.5)          # mph
    base_speed = np.where(is_weekend == 1, base_speed * 1.15, base_speed)
    speed_noise = rng.normal(1.0, 0.18, n_rows).clip(0.35, 2.2)
    effective_speed_mph = (base_speed * speed_noise).clip(2.0, 45.0)

    travel_hours = distance_mi / effective_speed_mph
    # Fixed overhead: pickup/dropoff friction, traffic lights, etc.
    overhead_sec = rng.gamma(shape=3.0, scale=40.0, size=n_rows)  # ~120s mean
    trip_duration = (travel_hours * 3600 + overhead_sec).round().astype(int)
    trip_duration = trip_duration.clip(30, 3 * 3600)  # 30s to 3h, matches Kaggle cleaning rules

    dropoff_dt = pickup_dt + pd.to_timedelta(trip_duration, unit="s")

    df = pd.DataFrame({
        "id": [f"id{100000 + i}" for i in range(n_rows)],
        "vendor_id": vendor_id,
        "pickup_datetime": pickup_dt,
        "dropoff_datetime": dropoff_dt,
        "passenger_count": passenger_count,
        "pickup_longitude": pickup_lon,
        "pickup_latitude": pickup_lat,
        "dropoff_longitude": dropoff_lon,
        "dropoff_latitude": dropoff_lat,
        "store_and_fwd_flag": store_and_fwd_flag,
        "trip_duration": trip_duration,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} synthetic trips to {OUT_PATH}")
    print(df.describe(include="all").T[["count", "mean", "min", "max"]] if False else df.head())
