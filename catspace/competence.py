"""
competence.py — a COMPETENCE MAP over the reachability embedding space
(2026-07-13, Kaveh's Method 2): "define some performance metric of our engine
in different areas of the embedding space, and have it spend more time searching
in parts of the space where it's weaker."

Method 1 (FBSearchPolicy.reliability) measures the engine's unreliability at a
position by ACTUALLY searching (shallow vs deep) -- exact but expensive. Method 2
PREDICTS that unreliability from the position's F-embedding alone, cheaply, by
remembering where the engine has been unreliable before: a non-parametric kNN
field over embedding space. So Method 2 can gate search WITHOUT first paying for
the deep search, and it generalizes ("this region has been sharp for me").

Built offline (experiments/build_competence_map.py) from a corpus of positions,
each stamped with its F-embedding and its Method-1 reliability. Query is cosine
kNN: the predicted unreliability of a position is the mean measured reliability
of its k nearest embedded neighbors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class CompetenceMap:
    def __init__(self, embeddings: np.ndarray, reliabilities: np.ndarray, k: int = 16):
        assert len(embeddings) == len(reliabilities) and len(embeddings) >= k
        e = np.asarray(embeddings, dtype=np.float32)
        self.E = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)   # unit rows
        self.r = np.asarray(reliabilities, dtype=np.float32)
        self.k = k

    def query(self, f: np.ndarray) -> np.ndarray:
        """(d,) or (M, d) F-embeddings -> (M,) predicted unreliability = mean
        measured reliability over the k nearest embedded neighbours (cosine)."""
        f = np.asarray(f, dtype=np.float32)
        single = f.ndim == 1
        if single:
            f = f[None, :]
        fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
        sims = fn @ self.E.T                                  # (M, N) cosine
        idx = np.argpartition(-sims, self.k - 1, axis=1)[:, :self.k]
        out = self.r[idx].mean(axis=1)
        return out[0] if single else out

    def save(self, path) -> None:
        np.savez(Path(path), embeddings=self.E, reliabilities=self.r, k=self.k)

    @staticmethod
    def load(path) -> "CompetenceMap":
        d = np.load(Path(path))
        return CompetenceMap(d["embeddings"], d["reliabilities"], int(d["k"]))
