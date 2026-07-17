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
    # vectorized random-empty-square pick: give each square a random priority,
    # veto occupied squares, argmax -> one uniform empty square per row
    sq_bits = (occ[:, None] >> np.arange(64, dtype=np.uint64)[None, :]) & np.uint64(1)
    pri = rng.random((n, 64))
    pri[sq_bits.astype(bool)] = -1.0
    sq = pri.argmax(axis=1)                            # (n,) empty square per row
    plane = np.array(_ADDABLE_PLANES)[rng.integers(len(_ADDABLE_PLANES), size=n)]
    out[np.arange(n), plane] |= (np.uint64(1) << sq.astype(np.uint64))
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


def irreversible_sibling_pairs(boards, rng, cap: int = 48):
    """PROVABLY mutually-unreachable sibling pairs, admitted by CERTIFICATE
    INCOMPARABILITY (Kaveh 2026-07-17, fungibility fix): naive rules like
    "different capture square" are UNSOUND -- capturing knight-e6 vs knight-c6
    leaves identical material multisets and the survivors transpose (same for
    rooks; pawns via doubled-pawn substitution). Positions are SETS; piece
    identity is not conserved.

    Criterion (per the define-identifications rule): monotone certificates
    C(s) = (white pawn count, black pawn count, white total, black total,
    white pawn budget, black pawn budget) -- each provably NONINCREASING under
    every legal move, promotion-safe (type counts are NOT: promotion mints
    pieces). A move pair (m1, m2) is admitted iff the child certificates are
    INCOMPARABLE: each child strictly below the other on >=1 coordinate. Then
    neither can ever descend to the other => mutually unreachable, with no
    identity/fungibility assumptions. Surviving pairs are mostly pawn-push vs
    capture (mover budget drops vs victim count drops -- orthogonal coords).

    boards: list[chess.Board]. Returns stacked (packed_a, meta_a, packed_b,
    meta_b) for <=cap pairs, or None."""
    import chess as _c
    from catspace.data.encode import encode_meta, encode_packed

    def cert(b):
        wp = len(b.pieces(_c.PAWN, _c.WHITE)); bp = len(b.pieces(_c.PAWN, _c.BLACK))
        wt = _popcount(b.occupied_co[_c.WHITE]); bt = _popcount(b.occupied_co[_c.BLACK])
        wb = sum(7 - _c.square_rank(sq) for sq in b.pieces(_c.PAWN, _c.WHITE))
        bb = sum(_c.square_rank(sq) for sq in b.pieces(_c.PAWN, _c.BLACK))
        return (wp, bp, wt, bt, wb, bb)

    def _popcount(x):
        return bin(x).count("1")

    def incomparable(c1, c2):
        a_less = any(x < y for x, y in zip(c1, c2))
        b_less = any(y < x for x, y in zip(c1, c2))
        return a_less and b_less

    pa, ma, pb, mb = [], [], [], []
    order = rng.permutation(len(boards))
    for i in order:
        if len(pa) >= cap:
            break
        b = boards[i]
        # irreversible candidates only (captures + pawn moves); cap the scan
        cands = []
        for m in b.legal_moves:
            if b.is_capture(m) or b.piece_type_at(m.from_square) == _c.PAWN:
                b2 = b.copy(stack=False); b2.push(m)
                cands.append((b2, cert(b2)))
                if len(cands) >= 6:
                    break
        found = None
        for j in range(len(cands)):
            for k in range(j + 1, len(cands)):
                if incomparable(cands[j][1], cands[k][1]):
                    found = (cands[j][0], cands[k][0]); break
            if found:
                break
        if not found:
            continue
        for bb2, ps, ms in ((found[0], pa, ma), (found[1], pb, mb)):
            ps.append(encode_packed(bb2)); ms.append(encode_meta(bb2))
    if not pa:
        return None
    return (np.stack(pa), np.stack(ma), np.stack(pb), np.stack(mb))
