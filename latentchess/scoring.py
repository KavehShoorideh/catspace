"""
scoring.py — the single source of truth for terminal-outcome scoring.

README bug ledger lesson 5: two readouts consuming the SAME learned field
differed 24% vs 99.8% mate-rate purely because one scored a rook-capture
outcome as neutral (0.0) and the other as catastrophic (0.1%-quantile). This
module is that one tested place: every readout takes a TerminalScores and
every magic constant (1e6/1e9/1e15/1e18, 0.0-draw, ...) dies at its call site.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latentchess.chain import TransitionChain, KIND_MATE, KIND_STALEMATE, KIND_WHITE_TERMINAL


@dataclass(frozen=True)
class TerminalScores:
    mate: float
    draw: float
    bwin: float

    @classmethod
    def big(cls, scale: float = 1e18) -> "TerminalScores":
        return cls(mate=scale, draw=-scale, bwin=-scale)

    @classmethod
    def from_reach_quantiles(cls, reach_live: np.ndarray, hi: float = 0.999, lo: float = 0.001) -> "TerminalScores":
        """The 24%-vs-99.8% fix: score MATE at the top reach quantile, DRAW/BWIN
        at the bottom, instead of an arbitrary constant like 0.0."""
        mate = float(np.quantile(reach_live, hi))
        draw = float(np.quantile(reach_live, lo))
        return cls(mate=mate, draw=draw, bwin=draw)

    def for_kind(self, kind: int) -> float:
        if kind == KIND_MATE: return self.mate
        if kind in (KIND_STALEMATE, KIND_WHITE_TERMINAL): return self.draw
        raise ValueError(f"for_kind only defined for terminal kinds, got {kind}")


def dtm_filled(dtm: np.ndarray, n: int, inf_value: float = 1e6) -> np.ndarray:
    """Extend a live-state DTM array to the full chain's state space, replacing
    inf (drawn/lost) with a large finite sentinel -- the `dtm_full` idiom that
    was copy-pasted across exp_krkn2.py/exp_search.py."""
    full = np.full(n, inf_value)
    full[: len(dtm)] = np.where(np.isfinite(dtm), dtm, inf_value)
    return full


def override_move_values(V: np.ndarray, chain: TransitionChain, ts: TerminalScores) -> np.ndarray:
    """Overwrite per-move values at terminal-kind moves with the pinned
    convention, in place-safe fashion (returns a new array)."""
    V = V.copy()
    mk = chain.move_kind
    V[mk == KIND_MATE] = ts.mate
    V[mk == KIND_STALEMATE] = ts.draw
    V[mk == KIND_WHITE_TERMINAL] = ts.bwin
    return V


def fill_terminal_state_scores(scores: np.ndarray, chain: TransitionChain, ts: TerminalScores) -> np.ndarray:
    """Overwrite the absorbing-state entries of a length-n state-score vector
    with the pinned terminal convention."""
    scores = scores.copy()
    scores[chain.terminals.mate] = ts.mate
    scores[chain.terminals.draw] = ts.draw
    if chain.terminals.bwin is not None:
        scores[chain.terminals.bwin] = ts.bwin
    return scores
