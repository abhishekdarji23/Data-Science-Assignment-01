"""
generate_data.py
-----------------
CRISP-DM Phase 2: Data Understanding (data acquisition step).

Generates the dataset that THIS project's audit platform audits -- a
compact NYC-taxi-style trip duration dataset, in the same spirit as
project 01 (data/generate_data.py there) but kept self-contained here
since this project's deliverable is the AUDIT, not the taxi model itself.

This build environment has no internet access to Kaggle, so the data is
synthetic. A handful of exact-duplicate rows are deliberately injected
(same technique used in other projects in this series) so the "no
duplicate rows" audit check has something concrete to find.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_ROWS = 4000
N_INJECTED_DUPLICATES = 12

OUT_PATH = Path(__file__).parent / "taxi_trips_synthetic.csv"


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def generate(n_rows: int = N_ROWS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pickup_lat = rng.normal(40.75, 0.045, n_rows).clip(40.63, 40.86)
    pickup_lon = rng.normal(-73.98, 0.045, n_rows).clip(-74.03, -73.77)
    trip_radius = rng.exponential(scale=0.02, size=n_rows)
    angle = rng.uniform(0, 2 * np.pi, n_rows)
    dropoff_lat = (pickup_lat + trip_radius * np.sin(angle)).clip(40.63, 40.86)
    dropoff_lon = (pickup_lon + trip_radius * np.cos(angle)).clip(-74.03, -73.77)

    start = pd.Timestamp("2023-01-01")
    minutes_offset = rng.integers(0, 365 * 24 * 60, n_rows)
    pickup_dt = start + pd.to_timedelta(minutes_offset, unit="m")
    hour = pickup_dt.hour.values
    is_rush_hour = (((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 19))).astype(int)

    passenger_count = rng.choice([1, 1, 1, 2, 2, 3, 4], size=n_rows)
    distance_mi = haversine_miles(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    base_speed = np.where(is_rush_hour == 1, 9.5, 14.5)
    speed_noise = rng.normal(1.0, 0.18, n_rows).clip(0.35, 2.2)
    effective_speed_mph = (base_speed * speed_noise).clip(2.0, 45.0)
    travel_hours = distance_mi / effective_speed_mph
    overhead_sec = rng.gamma(shape=3.0, scale=40.0, size=n_rows)
    trip_duration = (travel_hours * 3600 + overhead_sec).round().astype(int)
    trip_duration = trip_duration.clip(30, 3 * 3600)

    df = pd.DataFrame({
        "trip_id": [f"trip{100000 + i}" for i in range(n_rows)],
        "pickup_datetime": pickup_dt,
        "passenger_count": passenger_count,
        "pickup_latitude": pickup_lat,
        "pickup_longitude": pickup_lon,
        "dropoff_latitude": dropoff_lat,
        "dropoff_longitude": dropoff_lon,
        "distance_mi": distance_mi.round(3),
        "trip_duration": trip_duration,
    })

    dup_rows = df.sample(n=N_INJECTED_DUPLICATES, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)
    df["trip_id"] = [f"trip{100000 + i}" for i in range(len(df))]
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    check_cols = [c for c in df.columns if c != "trip_id"]
    print(f"Wrote {len(df):,} trips to {OUT_PATH}")
    print(f"Duplicate rows (excluding trip_id): {df.duplicated(subset=check_cols).sum()}")
