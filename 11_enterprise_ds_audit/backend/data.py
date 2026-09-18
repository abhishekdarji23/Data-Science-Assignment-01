"""
Builds a synthetic train/test dataset that deliberately contains a handful
of realistic data-quality problems, so the audit endpoints below have
something real to catch (rather than auditing a clean dataset and always
reporting a perfect score).

Issues seeded in on purpose:
  - a few rows leaked from train into test (exact duplicates)
  - a "leaky_feature" column that's almost a copy of the target
  - missing values injected into one column
  - one feature's test-set distribution shifted away from train (drift)
  - imbalanced target classes
  - a couple of duplicate feature-rows with conflicting labels (noisy labels)
  - a metadata block claiming a seed was recorded, but the recorded dataset
    hash is stale (doesn't match the current data) — a documentation/
    reproducibility gap
"""

import random
import hashlib


def _make_row(rng, idx, drift=False):
    f1 = rng.uniform(0, 10)
    f2 = rng.uniform(-5, 5) + (3.0 if drift else 0.0)  # drifted feature in test
    f3 = rng.choice(["A", "B", "C"])
    target = 1 if (f1 + f2) > 5 else 0
    return {"id": idx, "f1": round(f1, 3), "f2": round(f2, 3), "f3": f3, "target": target}


def build_dataset(seed=42):
    rng = random.Random(seed)

    train = [_make_row(rng, i) for i in range(240)]
    test = [_make_row(rng, i + 1000, drift=True) for i in range(60)]

    # imbalance: bias target toward 0 in a chunk of train rows
    for r in train[:100]:
        if rng.random() < 0.6:
            r["target"] = 0

    # leaky_feature is derived from the FINAL target (after the imbalance
    # flip above), so it stays a near-perfect proxy for what the audit
    # should actually catch
    for r in train + test:
        r["leaky_feature"] = round(r["target"] + rng.uniform(-0.02, 0.02), 4)

    # inject missing values into f2 for ~8% of train rows
    for r in train:
        if rng.random() < 0.08:
            r["f2"] = None

    # leak 5 exact rows from train into test (same id range, identical content)
    leaked = [dict(r) for r in train[:5]]
    for r in leaked:
        r["id"] = r["id"] + 5000  # give them a "new" id but identical features/target
    test.extend(leaked)

    # noisy labels: duplicate a couple of feature-rows but flip the label
    noisy_pair = dict(train[10])
    noisy_pair["id"] = 9999
    noisy_pair["target"] = 1 - noisy_pair["target"]
    train.append(noisy_pair)

    metadata = {
        "seed_recorded": True,
        "recorded_seed": 42,
        # deliberately stale: this hash was computed on an earlier version of
        # the dataset and never updated after the leak/noise were introduced
        "recorded_dataset_hash": "stale-0000000000",
    }

    return train, test, metadata


def dataset_hash(train, test):
    payload = repr(train) + repr(test)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
