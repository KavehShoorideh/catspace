"""
nn/hard_negatives.py — exact, directional unreachable negatives from chess's
material monotonicity (Kaveh 2026-07-16: "positions that aren't reachable as
contrast to help speed up learning").

Total piece count is NON-INCREASING along any game: a capture removes one, a
promotion is net-zero (pawn->piece), a quiet move is zero. Therefore

    count(g) > count(s)  =>  d(s -> g) = infinity   (EXACT, free, directional).

We build a hard negative g for anchor s by ADDING one piece on an empty square.
The negative is a HARD one (g differs from s by a single piece, so it *looks*
one capture away) and DIRECTIONAL (d(s->g)=inf while d(g->s) may be 1 ply --
capture the added piece). The two-encoder F/B split carries the direction:
pushing d(F(s), B(g)) large does NOT touch d(F(g), B(s)).

This operates on PACKED bitboards (N,12) uint64 -- pure numpy, no board
reconstruction, so it is cheap enough to run every training step on CPU
alongside batch prep.

WHY THIS IS IMMUNE TO EN-PASSANT AND DIAGONAL CAPTURES (Kaveh 2026-07-16):
the count-monotonicity invariant makes NO claim about where pawns can go --
it only counts pieces, and every pawn dynamic (single/double push, diagonal
capture, en-passant, promotion) is either count-neutral or count-decreasing.
So we never reason about pawn reachability and never synthesize a pawn move.
A FUTURE pawn-structure negative (e.g. "a pawn cannot be on a lower rank")
would be UNSOUND without handling en-passant (removes an enemy pawn from an
adjacent file) and diagonal captures (a pawn changes file), so such negatives
are deliberately NOT built here; the horizon negatives below use only OBSERVED
positions, which bake in every pawn rule correctly for free.
"""
from __future__ import annotations

import numpy as np

# non-pawn, non-king planes are the safe pieces to add: a pawn on rank 1/8 is
# an invalid square and a second king is illegal; adding N/B/R/Q keeps g a
# plausible board while still strictly increasing the count (added-back captured
# material is exactly what a "hard" reverse-of-capture negative looks like).
_ADDABLE_PLANES = (1, 2, 3, 4, 7, 8, 9, 10)   # W/B knight,bishop,rook,queen
_FULL64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def unreachable_goals(packed: np.ndarray, seed: int = 0) -> np.ndarray:
    """(N,12) uint64 anchors -> (N,12) uint64 negatives, each with exactly one
    extra piece on a previously-empty square. count(neg) = count(anchor)+1, so
    every neg is provably unreachable from its anchor."""
    packed = np.atleast_2d(packed).astype(np.uint64)
    n = packed.shape[0]
    rng = np.random.default_rng(seed)
    out = packed.copy()
    occ = np.zeros(n, dtype=np.uint64)
    for p in range(12):
        occ |= packed[:, p]
    for i in range(n):
        empty_mask = int(_FULL64) & ~int(occ[i]) & int(_FULL64)
        empties = [s for s in range(64) if (empty_mask >> s) & 1]
        if not empties:
            continue                                   # full board (never in chess)
        sq = empties[int(rng.integers(len(empties)))]
        plane = _ADDABLE_PLANES[int(rng.integers(len(_ADDABLE_PLANES)))]
        out[i, plane] = np.uint64(int(out[i, plane]) | (1 << sq))
    return out


def piece_count(packed: np.ndarray) -> np.ndarray:
    """(N,12) uint64 -> (N,) total piece count (popcount over all planes)."""
    packed = np.atleast_2d(packed).astype(np.uint64)
    tot = np.zeros(packed.shape[0], dtype=np.int64)
    for p in range(12):
        tot += np.array([int(x).bit_count() for x in packed[:, p]], dtype=np.int64)
    return tot


def repel_loss(d_neg, margin):
    """Hinge that pushes each unreachable/out-of-horizon distance ABOVE margin:
    mean(relu(margin - d_neg)). Shared by both negative sources -- count
    negatives (margin = a large 'infinity' target) and horizon negatives
    (margin = k / scale). d_neg is a torch tensor of d(F(s), B(neg)); margin a
    scalar or per-row tensor. Complements the two-ply stitch's attraction with
    targeted repulsion, so structure no longer has to emerge only from where
    stitches fail to hold (the uniform-repulsion regime that leaves the field
    slow to separate)."""
    import torch
    return torch.relu(margin - d_neg).mean()
