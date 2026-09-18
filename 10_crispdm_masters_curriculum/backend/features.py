"""
CRISP-DM Phase 3 (Data Preparation): turn raw records into numeric feature
vectors — one-hot encode the categorical fields, min-max scale the numeric
ones, and concatenate. This is what both the similarity search (phase 4/6)
and the evaluation (phase 5) run on.
"""

import numpy as np
from data import EDUCATION_LEVELS, OCCUPATIONS


def build_feature_matrix(records):
    ages = np.array([r["age"] for r in records], dtype=float)
    hours = np.array([r["hours_per_week"] for r in records], dtype=float)

    def scale(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    age_scaled = scale(ages)
    hours_scaled = scale(hours)

    edu_onehot = np.zeros((len(records), len(EDUCATION_LEVELS)))
    occ_onehot = np.zeros((len(records), len(OCCUPATIONS)))
    for i, r in enumerate(records):
        edu_onehot[i, EDUCATION_LEVELS.index(r["education"])] = 1.0
        occ_onehot[i, OCCUPATIONS.index(r["occupation"])] = 1.0

    matrix = np.column_stack([age_scaled, hours_scaled, edu_onehot, occ_onehot])

    feature_names = (
        ["age_scaled", "hours_scaled"]
        + [f"education={e}" for e in EDUCATION_LEVELS]
        + [f"occupation={o}" for o in OCCUPATIONS]
    )
    return matrix, feature_names
