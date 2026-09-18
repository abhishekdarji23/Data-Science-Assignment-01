"""
generate_data.py
-----------------
CRISP-DM Phase 2: Data Understanding (data acquisition step).

Generates a synthetic Airbnb-style listings dataset with BOTH tabular
features (bedrooms, bathrooms, accommodates, room_type, neighborhood)
and a free-text description -- the classic "multimodal AutoML" demo
shape (this exact combination -- tabular + text listing description --
is one of real AutoGluon Multimodal's own marquee tutorial datasets).

This build environment has no internet access to Kaggle, so the data is
synthetic -- generated so that price depends on BOTH modalities in a
way neither can fully explain alone:

  * TABULAR component: bedrooms, bathrooms, accommodates, room_type, and
    neighborhood set a base price (standard, structured pricing factors).
  * TEXT component: a hidden "quality tier" (budget / standard / premium
    / luxury) adds or subtracts from that base price, and ALSO determines
    which descriptive phrases appear in the listing's text description
    (e.g. "recently renovated", "stunning skyline view", "a bit dated").
    Critically, the text does NOT restate the bedroom/bathroom/capacity
    numbers -- it only carries the quality signal -- so a text-only model
    structurally cannot recover the tabular component, and a tabular-only
    model cannot recover the quality component. Only a model that uses
    BOTH has enough information to explain the full price.

This is what makes "multimodal fusion beats either single modality"
checkable rather than merely asserted (see AUDIT_REPORT.md).
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_ROWS = 1500

OUT_PATH = Path(__file__).parent / "listings_synthetic.csv"

ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room"]
ROOM_TYPE_BASE = {"Entire home/apt": 120, "Private room": 55, "Shared room": 30}

NEIGHBORHOODS = ["Downtown", "Uptown", "Waterfront", "Suburb"]
NEIGHBORHOOD_ADJUST = {"Downtown": 35, "Uptown": 15, "Waterfront": 45, "Suburb": -10}

QUALITY_TIERS = ["budget", "standard", "premium", "luxury"]
QUALITY_PRICE_ADJUST = {"budget": -30, "standard": 0, "premium": 40, "luxury": 90}
QUALITY_WEIGHTS = [0.25, 0.40, 0.25, 0.10]

QUALITY_PHRASES = {
    "budget": ["a bit dated but functional", "basic amenities", "no frills stay",
               "shared bathroom down the hall", "thin walls, budget option"],
    "standard": ["comfortable and clean", "well maintained", "convenient location",
                 "standard kitchen and bath", "reliable wifi"],
    "premium": ["recently renovated", "modern finishes throughout", "high-end appliances",
                "spacious and bright", "walk to great restaurants"],
    "luxury": ["stunning skyline view", "designer furnished", "five-star concierge service",
               "private rooftop access", "marble bathroom and premium linens"],
}
FILLER_PHRASES = [
    "close to public transit", "quiet street", "great for a short stay",
    "easy check-in process", "responsive host", "parking available nearby",
]


def generate(n_rows: int = N_ROWS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    bedrooms = rng.integers(1, 6, n_rows)
    bathrooms = rng.integers(1, 4, n_rows)
    accommodates = (bedrooms * rng.integers(1, 3, n_rows) + 1).clip(1, 12)
    room_type = rng.choice(ROOM_TYPES, size=n_rows, p=[0.55, 0.35, 0.10])
    neighborhood = rng.choice(NEIGHBORHOODS, size=n_rows, p=[0.30, 0.25, 0.20, 0.25])
    quality_tier = rng.choice(QUALITY_TIERS, size=n_rows, p=QUALITY_WEIGHTS)

    # --- TABULAR price component ---
    tabular_price = (
        np.array([ROOM_TYPE_BASE[r] for r in room_type])
        + np.array([NEIGHBORHOOD_ADJUST[n] for n in neighborhood])
        + bedrooms * 18
        + bathrooms * 10
        + accommodates * 4
    )

    # --- TEXT (quality) price component ---
    quality_price = np.array([QUALITY_PRICE_ADJUST[q] for q in quality_tier])

    noise = rng.normal(0, 12, n_rows)
    price = (tabular_price + quality_price + noise).clip(20, None).round(2)

    descriptions = []
    for q in quality_tier:
        n_quality_phrases = rng.integers(2, 4)
        chosen_quality = rng.choice(QUALITY_PHRASES[q], size=n_quality_phrases, replace=False)
        n_filler = rng.integers(1, 3)
        chosen_filler = rng.choice(FILLER_PHRASES, size=n_filler, replace=False)
        all_phrases = list(chosen_quality) + list(chosen_filler)
        rng.shuffle(all_phrases)
        descriptions.append(". ".join(all_phrases).capitalize() + ".")

    df = pd.DataFrame({
        "listing_id": [f"listing{1000 + i}" for i in range(n_rows)],
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "accommodates": accommodates,
        "room_type": room_type,
        "neighborhood": neighborhood,
        "description": descriptions,
        "price": price,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} listings to {OUT_PATH}")
    print(f"Price range: ${df['price'].min():.0f} - ${df['price'].max():.0f}, mean ${df['price'].mean():.0f}")
    print(df[["bedrooms", "room_type", "neighborhood", "description", "price"]].head(3).to_string())
