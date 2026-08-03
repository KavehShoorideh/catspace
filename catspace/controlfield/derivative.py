"""catspace/controlfield/derivative.py -- Phase 2 of the control-field spec:
directional derivative D_m(g) per legal move, and the ascent cone K(s).

Critical detail (spec 3.1, verified by test_derivative.py::test_orientation_bug):
after pushing a move the side to move flips, so the raw field flips sign/orientation.
D_m must be computed by re-expressing C' in the ORIGINAL mover's frame before
subtracting -- control.orient() takes mover_is_white explicitly rather than reading
board.turn internally, specifically so this can't be gotten wrong by accident here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from catspace.controlfield.control import (
    weighted_attacker_field, orient, critical_square_mask, compute_fields,
)


def move_derivatives(board: chess.Board, weights=None, exploitable=False):
    """-> (moves: list[chess.Move], D: (n_legal, 64) float32) -- D_m(g) for every
    legal move, all re-expressed in the ORIGINAL mover's frame (spec 3.1).

    If exploitable=True, also returns E: (n_legal, 64) bool, mover-oriented --
    E[i, g] = True iff the OPPONENT has at least one LEGAL reply landing on/
    capturing square g after move i. Fixes a real bug found via gate 2 (only
    12.5% hit rate vs the >=60% bar, reports/phase-2.md): `damage(m)` originally
    penalized a move for "losing control" of ANY square with a negative D_m,
    even when the opponent has no legal way to act on it next move -- e.g. a
    checking move that "loses" control of a far-away square is not actually
    risky there, because the opponent's only legal replies are check responses
    that can't reach it. E lets damage() restrict itself to squares the
    opponent can actually, legally, immediately punish -- one extra ply of
    legal-move enumeration (already implicit in needing the post-move
    position), not a search, consistent with the spec's "no search" non-goal."""
    mover_is_white = board.turn == chess.WHITE
    c_before = orient(weighted_attacker_field(board, weights), mover_is_white)
    moves = list(board.legal_moves)
    D = np.zeros((len(moves), 64), np.float32)
    E = np.zeros((len(moves), 64), bool) if exploitable else None
    for i, mv in enumerate(moves):
        b2 = board.copy(stack=False)
        b2.push(mv)
        c_after_raw = weighted_attacker_field(b2, weights)
        # orient with the ORIGINAL mover's color, NOT b2.turn (which just flipped) --
        # this is exactly the bug the spec warns about.
        c_after = orient(c_after_raw, mover_is_white)
        D[i] = c_after - c_before
        if exploitable:
            dest_abs = {reply.to_square for reply in b2.legal_moves}
            if mover_is_white:
                dest_oriented = dest_abs
            else:
                dest_oriented = {chess.square(chess.square_file(s), 7 - chess.square_rank(s))
                                  for s in dest_abs}
            for g in dest_oriented:
                E[i, g] = True
    if exploitable:
        return moves, D, E
    return moves, D


@dataclass
class ConeConfig:
    tau: float = 0.0
    target_mode: str = "king_zone"     # king_zone | weak_squares | explicit
    k_weak: int = 6
    explicit_squares: frozenset = field(default_factory=frozenset)


def target_squares(board: chess.Board, C_mover_pov: np.ndarray, cfg: ConeConfig):
    """-> set of square indices, in the MOVER's oriented frame (matching D_m's frame)."""
    mover_is_white = board.turn == chess.WHITE
    if cfg.target_mode == "explicit":
        squares = set(cfg.explicit_squares)
        # caller-supplied squares are in absolute board terms; reorient like the field.
        if mover_is_white:
            return squares
        return {chess.square(chess.square_file(s), 7 - chess.square_rank(s)) for s in squares}
    if cfg.target_mode == "king_zone":
        enemy_king = board.king(not board.turn)
        if enemy_king is None:
            return set()
        kf, kr = chess.square_file(enemy_king), chess.square_rank(enemy_king)
        zone_abs = set()
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                f, r = kf + df, kr + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    zone_abs.add(chess.square(f, r))
        if mover_is_white:
            return zone_abs
        return {chess.square(chess.square_file(s), 7 - chess.square_rank(s)) for s in zone_abs}
    if cfg.target_mode == "weak_squares":
        # squares already weakest for the opponent = highest C(g) in the mover's
        # own oriented frame (mover already controls them well / opponent doesn't).
        order = np.argsort(-C_mover_pov)
        return set(int(s) for s in order[: cfg.k_weak])
    raise ValueError(f"unknown target_mode {cfg.target_mode!r}")


def ascent_cone(board: chess.Board, weights=None, mask_cfg=None, cone_cfg: ConeConfig | None = None):
    """-> dict with moves, D, gain, damage, in_cone (bool mask), and the derived
    scalars (spec 3.2/3.3): cone_size, best_gain, is_squeezed."""
    cone_cfg = cone_cfg or ConeConfig()
    mover_is_white = board.turn == chess.WHITE
    C = orient(weighted_attacker_field(board, weights), mover_is_white)
    M_white = critical_square_mask(board, mask_cfg)
    M = M_white if mover_is_white else np.array(
        [M_white[chess.square(chess.square_file(s), 7 - chess.square_rank(s))] for s in range(64)])
    moves, D, E = move_derivatives(board, weights, exploitable=True)
    T = target_squares(board, C, cone_cfg)
    t_idx = np.array(sorted(T), dtype=int) if T else np.array([], dtype=int)

    n = len(moves)
    gain = np.zeros(n, np.float32)
    damage = np.zeros(n, np.float32)
    for i in range(n):
        gain[i] = D[i, t_idx].sum() if len(t_idx) else 0.0
        # only count damage on squares the opponent can actually legally reach
        # next move (E) -- see move_derivatives' docstring for why (gate 2 fix).
        damage[i] = (M * np.minimum(D[i], 0.0) * E[i]).sum()
    in_cone = (gain > 0) & (damage >= -cone_cfg.tau)

    cone_size = float(in_cone.sum()) / n if n else 0.0
    best_gain = float(gain[in_cone].max()) if in_cone.any() else 0.0
    is_squeezed = not in_cone.any()
    return dict(moves=moves, D=D, gain=gain, damage=damage, in_cone=in_cone,
                cone_size=cone_size, best_gain=best_gain, is_squeezed=is_squeezed,
                target_squares=t_idx)
