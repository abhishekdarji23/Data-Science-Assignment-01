"""
Cosine-similarity LSH via random hyperplanes ("SimHash"): each hash table
projects every vector onto a handful of random hyperplanes and records only
the sign of each projection as a bit. Vectors that are close in cosine
similarity tend to land on the same side of most random hyperplanes, so
they collide into the same bucket far more often than unrelated vectors —
that's what makes bucket lookup a good *candidate filter* for approximate
nearest-neighbor search, avoiding a full brute-force scan.
"""

import numpy as np


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class CosineLSH:
    def __init__(self, dim, num_tables=4, num_hyperplanes=6, seed=7):
        self.dim = dim
        self.num_tables = num_tables
        self.num_hyperplanes = num_hyperplanes
        rng = np.random.default_rng(seed)
        # one random hyperplane matrix per table: shape (num_hyperplanes, dim)
        self.hyperplanes = [rng.normal(size=(num_hyperplanes, dim)) for _ in range(num_tables)]
        self.tables = [dict() for _ in range(num_tables)]
        self.vectors = None
        self.ids = None

    def _bucket_key(self, vector, table_idx):
        signs = self.hyperplanes[table_idx] @ vector >= 0
        return "".join("1" if s else "0" for s in signs)

    def build(self, vectors, ids):
        self.vectors = vectors
        self.ids = list(ids)
        self.tables = [dict() for _ in range(self.num_tables)]
        for t in range(self.num_tables):
            for row_idx, vector in enumerate(vectors):
                key = self._bucket_key(vector, t)
                self.tables[t].setdefault(key, []).append(row_idx)

    def bucket_stats(self):
        return [
            {"table": t, "num_buckets": len(table), "largest_bucket": max((len(v) for v in table.values()), default=0)}
            for t, table in enumerate(self.tables)
        ]

    def query(self, vector, k=5, exclude_id=None):
        candidate_rows = set()
        for t in range(self.num_tables):
            key = self._bucket_key(vector, t)
            candidate_rows.update(self.tables[t].get(key, []))

        scored = []
        for row_idx in candidate_rows:
            rid = self.ids[row_idx]
            if rid == exclude_id:
                continue
            sim = cosine_similarity(vector, self.vectors[row_idx])
            scored.append((rid, sim))

        scored.sort(key=lambda x: -x[1])
        return scored[:k], len(candidate_rows)


def brute_force_top_k(vectors, ids, query_vector, k=5, exclude_id=None):
    scored = []
    for row_idx, vec in enumerate(vectors):
        rid = ids[row_idx]
        if rid == exclude_id:
            continue
        scored.append((rid, cosine_similarity(query_vector, vec)))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]
