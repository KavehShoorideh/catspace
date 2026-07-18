"""
nn/unreachable.py — exact, directional legal-unreachability oracle
(Kaveh 2026-07-18; replaces the sibling hinge as the metric's spread signal).

`provably_unreachable(packed_s, meta_s, packed_g, meta_g)` returns, per pair,
whether position g is PROVABLY not reachable from position s by any sequence
of legal moves. Semantics: flag = theorem (safe to repel the s->g distance to
the unreachability floor); no flag = unknown (never repelled). Every test is
a NECESSARY condition for reachability, so a failure proves unreachability:

  1. COUNT MONOTONICITY (promotion-safe): per-color pawn count and per-color
     total piece count never increase along play (promotion converts a pawn,
     captures remove). If g exceeds s in any of the four, s cannot reach g.
  2. CASTLING RIGHTS only disappear. A right present in g but absent in s is
     impossible to regain.
  3. PAWN FORWARD CONES with injective matching: a white pawn on (file f,
     rank r) can only ever stand on squares with rank' >= r and
     |file' - f| <= rank' - r (each advance gains >= 1 rank; each file shift
     is a capture costing exactly one rank). Pawns are never created, so
     every pawn of g must be matched to a DISTINCT pawn of s inside whose
     cone it lies (unmatched s-pawns were captured or promoted -- allowed).
     Infeasibility of this bipartite matching (checked exactly; <=8 per
     side) proves unreachability. We deliberately RELAX the requirement
     that capture-steps have victims available: the relaxed system admits
     strictly more evolutions, so its infeasibility still proves the real
     game cannot do it.

Used by the trainer on the pairs already being pushed (zero extra embedding
cost): flagged pairs get a one-sided hinge floor on d(F(s)->B(g)) -- the
directional, certified replacement for random-pair "softness".
"""
from __future__ import annotations

import numpy as np

# packed plane indices (data/encode.py): 0..5 white PNBRQK, 6..11 black
_WP, _BP = 0, 6
_WHITE_SLICE, _BLACK_SLICE = slice(0, 6), slice(6, 12)
# meta indices (data/encode.py): 1..4 = castling rights K,Q,k,q flags
_RIGHTS = slice(1, 5)


def _pawn_list(bb: np.uint64) -> list[tuple[int, int]]:
    """bitboard -> [(file, rank)] (bit s = square s, rank = s//8, file = s%8)."""
    out = []
    v = int(bb)
    while v:
        s = (v & -v).bit_length() - 1
        out.append((s % 8, s // 8))
        v &= v - 1
    return out


def _cone_ok(src: tuple[int, int], dst: tuple[int, int], white: bool) -> bool:
    """Is dst inside src's forward cone for this color?"""
    df, dr = dst[0] - src[0], dst[1] - src[1]
    if not white:
        dr = -dr
    return dr >= 0 and abs(df) <= dr


def _match_feasible(srcs: list, dsts: list, white: bool) -> bool:
    """Injective matching of every dst pawn to a distinct src pawn within its
    cone (Kuhn's augmenting paths; <=8x8)."""
    if len(dsts) > len(srcs):
        return False
    adj = [[i for i, s in enumerate(srcs) if _cone_ok(s, d, white)] for d in dsts]
    match_of_src = [-1] * len(srcs)

    def augment(j, seen):
        for i in adj[j]:
            if i in seen:
                continue
            seen.add(i)
            if match_of_src[i] == -1 or augment(match_of_src[i], seen):
                match_of_src[i] = j
                return True
        return False

    return all(augment(j, set()) for j in range(len(dsts)))


def provably_unreachable(packed_s: np.ndarray, meta_s: np.ndarray,
                         packed_g: np.ndarray, meta_g: np.ndarray) -> np.ndarray:
    """(n,12) x (n,8) x (n,12) x (n,8) -> (n,) bool: True where s -/-> g is
    PROVEN. Vectorized count/rights rejection first; pawn-cone matching only
    on the survivors."""
    n = len(packed_s)
    cnt_s = np.stack([np.bitwise_count(packed_s[:, i]) for i in range(12)], 1)
    cnt_g = np.stack([np.bitwise_count(packed_g[:, i]) for i in range(12)], 1)
    flag = (cnt_g[:, _WP] > cnt_s[:, _WP]) | (cnt_g[:, _BP] > cnt_s[:, _BP]) \
         | (cnt_g[:, _WHITE_SLICE].sum(1) > cnt_s[:, _WHITE_SLICE].sum(1)) \
         | (cnt_g[:, _BLACK_SLICE].sum(1) > cnt_s[:, _BLACK_SLICE].sum(1))
    # castling: a right in g missing in s
    flag |= ((meta_g[:, _RIGHTS].astype(np.int16)
              - meta_s[:, _RIGHTS].astype(np.int16)) > 0).any(1)
    # pawn cones on the not-yet-flagged
    for i in np.flatnonzero(~flag):
        wg = _pawn_list(packed_g[i, _WP]); ws = _pawn_list(packed_s[i, _WP])
        if not _match_feasible(ws, wg, white=True):
            flag[i] = True
            continue
        bg = _pawn_list(packed_g[i, _BP]); bs = _pawn_list(packed_s[i, _BP])
        if not _match_feasible(bs, bg, white=False):
            flag[i] = True
    return flag
