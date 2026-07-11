"""
opponents.py — the Opponent protocol and its implementations.

Previously there was no shared protocol: the dominant form was a precomputed
`B_opt` reply-index array reimplemented ~4x (loop and vectorized-reduceat
variants), plus two broken standalone classes (minimax_opp.py imported a
nonexistent `DRAW` constant; stockfish_opp.py had a buggy FEN encoder). Both
are dropped -- `optimal_reply_table` below is THE vectorized B_opt, and a real
UCI Stockfish opponent returns at the full-board milestone against actual
python-chess boards (a toy 5x5 chain has no legal FEN mapping worth the
complexity stockfish_opp.py never got right).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from latentchess.chain import TransitionChain
from latentchess.scoring import dtm_filled


class Opponent(Protocol):
    def reply_index(self, chain: TransitionChain, mid: int, rng: np.random.Generator) -> int:
        """Return a LOCAL index into chain.outs_of(mid) selecting black's reply."""
        ...


def optimal_reply_table(chain: TransitionChain, dtm: np.ndarray) -> np.ndarray:
    """Per-move optimal black-reply LOCAL index, vectorized (THE B_opt): black
    maximizes dtm_filled over the move's outcome states -- captures/draws are
    scored at the sentinel (black's best, "escape into a draw"), matching the
    sign-flip law: optimal black MAXIMIZES white's distance-to-mate."""
    dtm_full = dtm_filled(dtm, chain.n)
    vals_flat = dtm_full[chain.out_flat]
    seg_max = np.maximum.reduceat(vals_flat, chain.op0)
    is_max = vals_flat == np.repeat(seg_max, chain.out_counts)
    cand = np.where(is_max, np.arange(len(vals_flat)), len(vals_flat))
    first = np.minimum.reduceat(cand, chain.op0)
    return (first - chain.op0).astype(np.int32)


class RandomOpponent:
    def reply_index(self, chain: TransitionChain, mid: int, rng: np.random.Generator) -> int:
        return int(rng.integers(0, chain.out_counts[mid]))


class TableOpponent:
    """Wraps any precomputed per-move reply-index table (the DTM oracle via
    optimal_reply_table, or a learned/data-derived black policy later)."""

    def __init__(self, reply_table: np.ndarray):
        self.reply_table = reply_table

    def reply_index(self, chain: TransitionChain, mid: int, rng: np.random.Generator) -> int:
        return int(self.reply_table[mid])


class EpsOptimalDTM:
    """Optimal w.p. 1-eps, else uniform-random -- the eps_b opponent curriculum
    used throughout the PI trainers (eps annealed 1.0 -> 0.0 across rounds)."""

    def __init__(self, reply_table: np.ndarray, eps: float):
        self.reply_table = reply_table
        self.eps = eps

    def reply_index(self, chain: TransitionChain, mid: int, rng: np.random.Generator) -> int:
        if self.eps > 0.0 and rng.random() <= self.eps:
            return int(rng.integers(0, chain.out_counts[mid]))
        return int(self.reply_table[mid])
