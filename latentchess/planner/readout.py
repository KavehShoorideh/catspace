"""
planner/readout.py — turning per-state scores into per-move values and policies.

Formalizes two seams that were previously separate, ad hoc functions scattered
across ~8 files:
  - MEAN vs MIN reply aggregation (README lesson 3: switching from MEAN, the
    cone's own training dynamics, to MIN, the actual minimax opponent, was
    worth +20 points of conversion by itself)
  - k-ply minimax backup on the learned field (exp_search.py's
    `minimax_backup`, generalized from a hardcoded MIN to any ReplyAgg)

Callers must terminal-fill `state_scores` first (see
scoring.fill_terminal_state_scores) -- backup() only ever overwrites live-state
entries, leaving the absorbing MATE/DRAW/[BWIN] entries at their pinned values
across every ply.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from latentchess.chain import TransitionChain
from latentchess.scoring import TerminalScores, override_move_values


class ReplyAgg(Enum):
    MEAN = "mean"
    MIN = "min"


def move_values(state_scores: np.ndarray, chain: TransitionChain, agg: ReplyAgg,
                 ts: TerminalScores) -> np.ndarray:
    """Per-move value: aggregate `state_scores` over each move's outcome set,
    then override terminal-kind moves (mate/stalemate/white-terminal) with the
    pinned TerminalScores convention -- the single tested place for this."""
    vals_flat = state_scores[chain.out_flat]
    if agg is ReplyAgg.MEAN:
        sums = np.add.reduceat(vals_flat, chain.op0)
        V = sums / chain.out_counts
    elif agg is ReplyAgg.MIN:
        V = np.minimum.reduceat(vals_flat, chain.op0)
    else:
        raise ValueError(f"unknown ReplyAgg: {agg}")
    return override_move_values(V, chain, ts)


def policy_from_values(V: np.ndarray, chain: TransitionChain) -> np.ndarray:
    """Per-live-state argmax move index (LOCAL index within the state's move
    range), first-argmax tie-break -- matches the reduceat idiom used
    throughout the original trainers so ported policies agree bit-for-bit."""
    smax = np.maximum.reduceat(V, chain.mp0)
    is_max = V == np.repeat(smax, chain.move_counts)
    cand = np.where(is_max, chain.pos_idx, chain.n_moves)
    first = np.minimum.reduceat(cand, chain.mp0)
    return (first - chain.mp0).astype(np.int32)


def greedy_policy(state_scores: np.ndarray, chain: TransitionChain, agg: ReplyAgg,
                   ts: TerminalScores) -> np.ndarray:
    V = move_values(state_scores, chain, agg, ts)
    return policy_from_values(V, chain)


def backup(state_scores: np.ndarray, chain: TransitionChain, agg: ReplyAgg,
           ts: TerminalScores, k: int) -> np.ndarray:
    """k-ply minimax backup on the learned field: k=0 is identity; each backup
    applies one white/black ply pair (agg over black replies, MAX over white's
    own moves). k=1 with agg=MIN reproduces exp_search.minimax_backup exactly."""
    scores = state_scores.copy()
    for _ in range(k):
        V = move_values(scores, chain, agg, ts)
        smax = np.maximum.reduceat(V, chain.mp0)
        scores = scores.copy()
        scores[: chain.n_live] = smax
    return scores
