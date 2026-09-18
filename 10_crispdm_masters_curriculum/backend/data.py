"""
Generates a small synthetic "census" panel: demographic records loosely
shaped like the classic income-bracket-prediction datasets (age, education,
occupation, hours worked, income bracket), but entirely synthetic and
seeded for reproducibility — no external data download needed.
"""

import random

EDUCATION_LEVELS = ["HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate"]
OCCUPATIONS = ["Admin", "Sales", "Service", "Craft", "Tech", "Management"]

EDUCATION_WEIGHT = {"HS-grad": 0, "Some-college": 1, "Bachelors": 2, "Masters": 3, "Doctorate": 4}
OCCUPATION_WEIGHT = {"Admin": 0, "Service": 0, "Sales": 1, "Craft": 1, "Tech": 2, "Management": 2}


def generate_records(n=300, seed=42):
    rng = random.Random(seed)
    records = []
    for i in range(n):
        age = rng.randint(19, 68)
        education = rng.choice(EDUCATION_LEVELS)
        occupation = rng.choice(OCCUPATIONS)
        hours_per_week = rng.randint(20, 60)

        # a deterministic-ish rule (plus noise) for income bracket, so the
        # data has real, learnable structure rather than being pure noise
        score = (
            EDUCATION_WEIGHT[education] * 1.5
            + OCCUPATION_WEIGHT[occupation] * 1.2
            + (hours_per_week - 40) * 0.05
            + (age - 40) * 0.02
            + rng.uniform(-1.5, 1.5)
        )
        income = ">50K" if score > 3 else "<=50K"

        records.append(
            {
                "id": i,
                "age": age,
                "education": education,
                "occupation": occupation,
                "hours_per_week": hours_per_week,
                "income": income,
            }
        )
    return records
