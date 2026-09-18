"""
data_prep.py
------------
CRISP-DM Phase 3: Data Preparation.

Shared feature-engineering code used by BOTH training (train.py) and
inference (predict.py / the API) so the model always sees features built
the exact same way it was trained on -- this is the single most common
source of train/serve skew and data leakage, so keeping one function as
the source of truth is deliberate.

Leakage note: every feature below is derivable at prediction time, i.e.
before the trip happens (pickup coordinates, dropoff coordinates as
requested by the rider, passenger count, and the pickup timestamp).
Nothing derived from `dropoff_datetime` or `trip_duration` itself is
used as an input feature.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REAL_CSV = DATA_DIR / "nyc_taxi_train.csv"       # if the user drops in real Kaggle data
SYNTHETIC_CSV = DATA_DIR / "nyc_taxi_synthetic.csv"

FEATURE_COLUMNS = [
    "passenger_count",
    "distance_mi",
    "manhattan_distance_mi",
    "bearing_deg",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_rush_hour",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
TARGET_COLUMN = "trip_duration"


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def manhattan_distance_miles(lat1, lon1, lat2, lon2):
    return haversine_miles(lat1, lon1, lat1, lon2) + haversine_miles(lat1, lon2, lat2, lon2)


def bearing_degrees(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds all model-input features in place and returns the frame."""
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    df["distance_mi"] = haversine_miles(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    df["manhattan_distance_mi"] = manhattan_distance_miles(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    df["bearing_deg"] = bearing_degrees(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 10)) | ((df["hour"] >= 16) & (df["hour"] <= 19))
    ).astype(int)
    return df


def load_dataset() -> pd.DataFrame:
    """Loads real Kaggle data if the user dropped it in data/, else synthetic."""
    if REAL_CSV.exists():
        df = pd.read_csv(REAL_CSV)
        source = "real Kaggle nyc_taxi_train.csv"
    elif SYNTHETIC_CSV.exists():
        df = pd.read_csv(SYNTHETIC_CSV)
        source = "synthetic (offline build environment)"
    else:
        raise FileNotFoundError(
            "No dataset found. Run `python data/generate_data.py` first, "
            "or place a Kaggle train.csv at data/nyc_taxi_train.csv."
        )
    print(f"[data_prep] loaded {len(df):,} rows from {source}")

    # Basic Kaggle-style cleaning: drop physically impossible rows
    df = df[(df["trip_duration"] >= 30) & (df["trip_duration"] <= 3 * 3600)]
    df = df[df["passenger_count"].between(1, 6)]
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame):
    """Returns (X, y) ready for sklearn, from a raw dataframe."""
    df = add_engineered_features(df)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else None
    return X, y


def build_single_feature_row(
    pickup_lat: float, pickup_lon: float,
    dropoff_lat: float, dropoff_lon: float,
    passenger_count: int, pickup_datetime: str,
) -> pd.DataFrame:
    """Builds a one-row feature frame for a single live prediction request."""
    raw = pd.DataFrame([{
        "pickup_latitude": pickup_lat,
        "pickup_longitude": pickup_lon,
        "dropoff_latitude": dropoff_lat,
        "dropoff_longitude": dropoff_lon,
        "passenger_count": passenger_count,
        "pickup_datetime": pickup_datetime,
    }])
    X, _ = build_features(raw)
    return X
