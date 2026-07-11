"""planner/policy.py — the Policy protocol and its implementations."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from latentchess.chain import TransitionChain
from latentchess.scoring import TerminalScores, dtm_filled, fill_terminal_state_scores
from latentchess.planner.readout import ReplyAgg, greedy_policy


class Policy(Protocol):
    def move_id(self, chain: TransitionChain, s: int, rng: np.random.Generator) -> int:
        """Return a GLOBAL move id (an index into chain.move_kind/move_names)."""
        ...


class TablePolicy:
    """Wraps a precomputed per-live-state LOCAL move-index array -- the `pol`
    convention used throughout the original trainers."""

    def __init__(self, local_moves: np.ndarray):
        self.local_moves = local_moves

    def move_id(self, chain: TransitionChain, s: int, rng: np.random.Generator) -> int:
        return int(chain.move_ptr[s]) + int(self.local_moves[s])


class RandomPolicy:
    def move_id(self, chain: TransitionChain, s: int, rng: np.random.Generator) -> int:
        a, b = int(chain.move_ptr[s]), int(chain.move_ptr[s + 1])
        return a + int(rng.integers(0, b - a))


class EpsGreedy:
    """`base` policy w.p. 1-eps, else a uniform-random move -- the eps_w
    curriculum used throughout the PI trainers."""

    def __init__(self, base: Policy, eps: float):
        self.base = base
        self.eps = eps
        self._random = RandomPolicy()

    def move_id(self, chain: TransitionChain, s: int, rng: np.random.Generator) -> int:
        if self.eps > 0.0 and rng.random() <= self.eps:
            return self._random.move_id(chain, s, rng)
        return self.base.move_id(chain, s, rng)


class DTMOraclePolicy:
    """The exact DTM-minimizing ceiling policy: white minimizes black's best
    (dtm-maximizing) reply, immediate mate always preferred. Reuses the
    MIN-aggregation readout on negated DTM -- MIN(-dtm) = -MAX(dtm), i.e.
    exactly "minimize the worst-case distance to mate"."""

    def __init__(self, chain: TransitionChain, dtm: np.ndarray):
        neg_dtm_full = -dtm_filled(dtm, chain.n)
        neg_dtm_full = fill_terminal_state_scores(neg_dtm_full, chain, TerminalScores.big())
        self._table = greedy_policy(neg_dtm_full, chain, ReplyAgg.MIN, TerminalScores.big())

    def move_id(self, chain: TransitionChain, s: int, rng: np.random.Generator) -> int:
        return int(chain.move_ptr[s]) + int(self._table[s])
