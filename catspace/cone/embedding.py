"""
cone/embedding.py — QuasimetricEmbedding protocol: the pluggable seam for
"how the quasimetric space is built".

FB-factorized methods (tabular SVD, neural InfoNCE) express reach as
F(s)@zG; a distance-form quasimetric method would express it as -d_q(s,g).
Either way the planner/arena/viz code only ever calls `reach(idx, goal)` --
this is what lets embedding methods be swapped and A/B-compared (abtest.py)
without touching the planner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class QuasimetricEmbedding(Protocol):
    d: int

    def reach(self, idx: np.ndarray | None, goal: "GoalSpec") -> np.ndarray:
        """Asymmetric closeness score from states at `idx` (None = all states)
        toward `goal`. Higher = closer/more reachable."""
        ...


@dataclass
class GoalSpec:
    name: str
    region: np.ndarray                  # union-state indices comprising the goal
    z: np.ndarray | None = None          # FB cache: zG = B[region].sum(0); optional for non-FB methods


def make_goal(name: str, region: np.ndarray, emb: QuasimetricEmbedding) -> GoalSpec:
    region = np.asarray(region)
    z = emb.B_of(region).sum(axis=0) if hasattr(emb, "B_of") else None
    return GoalSpec(name=name, region=region, z=z)


def reach(emb: QuasimetricEmbedding, goal: GoalSpec, idx: np.ndarray | None = None) -> np.ndarray:
    return emb.reach(idx, goal)


# name -> embedding type/factory; experiment CLIs select by `--embedding <name>`.
EMBEDDING_METHODS: dict[str, Callable[..., QuasimetricEmbedding]] = {}


def register_embedding(name: str):
    def deco(cls):
        EMBEDDING_METHODS[name] = cls
        return cls
    return deco
