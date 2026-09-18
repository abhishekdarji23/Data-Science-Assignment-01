"""
Six independent audit dimensions, each scored 0-100 with a list of concrete
findings. This mirrors the kind of automated data-quality/leakage review a
team would run before trusting a dataset for modeling.
"""

from collections import Counter, defaultdict
from data import dataset_hash


def _status(score):
    if score >= 90:
        return "pass"
    if score >= 70:
        return "warning"
    return "fail"


def _feature_key(row):
    return (row["f1"], round(row["f2"], 3) if row["f2"] is not None else None, row["f3"])


def audit_leakage(train, test):
    findings = []
    score = 100

    train_keys = {_feature_key(r): r["target"] for r in train}
    leaked_rows = [r for r in test if _feature_key(r) in train_keys]
    if leaked_rows:
        score -= min(60, len(leaked_rows) * 10)
        findings.append(
            f"{len(leaked_rows)} test row(s) are exact feature-duplicates of a train row "
            f"(ids: {[r['id'] for r in leaked_rows]}) — likely train/test contamination."
        )

    # crude correlation check for target leakage via a feature
    numeric_features = ["f1", "f2", "leaky_feature"]
    for feat in numeric_features:
        pairs = [(r[feat], r["target"]) for r in train if r.get(feat) is not None]
        if len(pairs) < 5:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        corr = cov / ((var_x * var_y) ** 0.5) if var_x > 0 and var_y > 0 else 0
        if abs(corr) > 0.9:
            score -= 30
            findings.append(f"Feature '{feat}' correlates {corr:.3f} with target — suspected target leakage.")

    if not findings:
        findings.append("No exact train/test duplicates or highly-correlated leaky features found.")

    return {"score": max(0, score), "status": _status(max(0, score)), "findings": findings}


def audit_missing_data(train):
    columns = ["f1", "f2", "f3"]
    findings = []
    score = 100

    for col in columns:
        missing = sum(1 for r in train if r.get(col) is None)
        pct = missing / len(train) * 100
        if pct > 20:
            score -= 40
            findings.append(f"Column '{col}' is missing in {pct:.1f}% of rows — severe.")
        elif pct > 5:
            score -= 15
            findings.append(f"Column '{col}' is missing in {pct:.1f}% of rows — above the 5% threshold.")

    if not findings:
        findings.append("No column exceeds the 5% missing-data threshold.")

    return {"score": max(0, score), "status": _status(max(0, score)), "findings": findings}


def audit_drift(train, test):
    findings = []
    score = 100

    for feat in ["f1", "f2"]:
        train_vals = [r[feat] for r in train if r.get(feat) is not None]
        test_vals = [r[feat] for r in test if r.get(feat) is not None]
        mean_train = sum(train_vals) / len(train_vals)
        mean_test = sum(test_vals) / len(test_vals)
        std_train = (sum((v - mean_train) ** 2 for v in train_vals) / len(train_vals)) ** 0.5

        drift = abs(mean_test - mean_train) / std_train if std_train > 0 else 0
        if drift > 1.0:
            score -= 40
            findings.append(
                f"Feature '{feat}': test mean ({mean_test:.2f}) is {drift:.2f} train-std-devs away from "
                f"train mean ({mean_train:.2f}) — severe distribution drift."
            )
        elif drift > 0.5:
            score -= 15
            findings.append(f"Feature '{feat}' shows moderate drift between train and test ({drift:.2f} std devs).")

    if not findings:
        findings.append("No feature shows meaningful train/test distribution drift.")

    return {"score": max(0, score), "status": _status(max(0, score)), "findings": findings}


def audit_label_quality(train):
    findings = []
    score = 100

    counts = Counter(r["target"] for r in train)
    total = sum(counts.values())
    minority_ratio = min(counts.values()) / total
    if minority_ratio < 0.2:
        score -= 30
        findings.append(f"Severe class imbalance: {dict(counts)} ({minority_ratio:.1%} minority class).")
    elif minority_ratio < 0.35:
        score -= 10
        findings.append(f"Moderate class imbalance: {dict(counts)} ({minority_ratio:.1%} minority class).")

    groups = defaultdict(set)
    for r in train:
        groups[(r["f1"], r["f3"])].add(r["target"])
    conflicting = [k for k, v in groups.items() if len(v) > 1]
    if conflicting:
        score -= min(30, len(conflicting) * 15)
        findings.append(f"{len(conflicting)} feature combination(s) appear with conflicting labels — noisy labels.")

    if not findings:
        findings.append("Classes are reasonably balanced and no conflicting labels were found.")

    return {"score": max(0, score), "status": _status(max(0, score)), "findings": findings}


def audit_split_integrity(train, test):
    findings = []
    score = 100

    total = len(train) + len(test)
    test_ratio = len(test) / total
    if not (0.15 <= test_ratio <= 0.25):
        score -= 20
        findings.append(f"Test split is {test_ratio:.1%} of the data — outside the expected 15-25% range.")

    for name, rows in [("train", train), ("test", test)]:
        ids = [r["id"] for r in rows]
        dupes = [id_ for id_, count in Counter(ids).items() if count > 1]
        if dupes:
            score -= 20
            findings.append(f"{len(dupes)} duplicate id(s) within the {name} split.")

    if not findings:
        findings.append(f"Split ratio ({test_ratio:.1%} test) is within range and ids are unique within each split.")

    return {"score": max(0, score), "status": _status(max(0, score)), "findings": findings}


def audit_reproducibility(train, test, metadata):
    findings = []
    score = 100

    if not metadata.get("seed_recorded"):
        score -= 50
        findings.append("No random seed was recorded for this dataset build — the split can't be reproduced.")

    current_hash = dataset_hash(train, test)
    recorded_hash = metadata.get("recorded_dataset_hash")
    if recorded_hash and recorded_hash != current_hash:
        score -= 50
        findings.append(
            f"Recorded dataset hash ('{recorded_hash}') doesn't match the current data "
            f"('{current_hash}') — the dataset changed after the hash was recorded, or the "
            f"hash was never updated."
        )

    if not findings:
        findings.append("Seed is recorded and the dataset hash matches what was recorded.")

    return {"score": max(0, score), "status": _status(max(0, score)), "findings": findings}


def run_full_audit(train, test, metadata):
    dimensions = {
        "leakage": audit_leakage(train, test),
        "missing_data": audit_missing_data(train),
        "distribution_drift": audit_drift(train, test),
        "label_quality": audit_label_quality(train),
        "split_integrity": audit_split_integrity(train, test),
        "reproducibility": audit_reproducibility(train, test, metadata),
    }
    overall = sum(d["score"] for d in dimensions.values()) / len(dimensions)
    return {
        "overall_score": round(overall, 1),
        "overall_status": _status(overall),
        "dimensions": dimensions,
    }
