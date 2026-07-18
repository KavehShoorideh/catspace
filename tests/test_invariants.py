"""Invariant / property tests -- the guards for the bug CLASSES this project has
actually been bitten by, that ordinary "does it run" tests miss:

  * value-scale symmetry   -- the DRAW_V=-0.999 minimax-asymmetry bug (2026-07-17)
  * quasimetric DIRECTION   -- the IQE reversed-interval bug (2026-07-16), which
                               passed all axiom tests because a transpose is
                               still a valid quasimetric; only the SEMANTICS
                               (which way is "already reached") caught it.

These encode what must hold, not a specific number, so they don't drift.
"""
import numpy as np
import torch

from catspace.nn.mcts import DRAW_V, MATE_V, MATED_V


# ---- value-scale symmetry (MCTS minimax) --------------------------------

def test_draw_is_neutral_under_side_flip():
    # the search scores the side to move as `q if white else -q`, so a value
    # must mean the same to both players. a draw is neutral => it must negate
    # to itself. DRAW_V=-0.999 read as +0.999 for Black (~ a Black win): bug.
    assert DRAW_V == -DRAW_V          # i.e. DRAW_V == 0
    assert DRAW_V == 0.0


def test_win_and_loss_are_antisymmetric():
    # White's win must be exactly Black's loss under the sign flip.
    assert MATE_V == -MATED_V


def test_draw_strictly_between_loss_and_win():
    # steering to a draw when LOSING must be an improvement (draw > loss), and
    # a draw must never look as good as a win. The old draw==loss broke this.
    assert MATED_V < DRAW_V < MATE_V


# ---- quasimetric direction / semantics (IQE) ----------------------------

def _iqe():
    from catspace.nn.iqe import IQE
    return IQE(d=16, components=4)


def test_iqe_self_distance_is_zero():
    q = _iqe()
    u = torch.randn(8, 16)
    assert torch.allclose(q(u, u), torch.zeros(8), atol=1e-5)


def test_iqe_direction_semantics():
    # THE bug guard: d(u->v) must be ~0 when u already dominates v coordinate-
    # wise ("already reached"), and LARGE when v exceeds u ("must climb"). The
    # reversed-interval bug had these swapped -- scoring reach backward in time.
    q = _iqe()
    big = torch.full((1, 16), 3.0)
    small = torch.zeros(1, 16)
    d_big_to_small = float(q(big, small))     # already past the target
    d_small_to_big = float(q(small, big))     # must climb up to it
    assert d_big_to_small < 1e-5
    assert d_small_to_big > 1.0
    assert d_small_to_big > d_big_to_small    # asymmetric, correct direction


def test_iqe_monotone_in_gap():
    # pushing v further above u must not DECREASE d(u->v): more to climb = farther.
    q = _iqe()
    u = torch.zeros(1, 16)
    d_prev = -1.0
    for gap in (0.5, 1.0, 2.0, 4.0):
        d = float(q(u, torch.full((1, 16), gap)))
        assert d >= d_prev - 1e-6
        d_prev = d


def test_iqe_triangle_inequality():
    # d(a->c) <= d(a->b) + d(b->c) for random triples (quasimetric by construction)
    q = _iqe()
    torch.manual_seed(0)
    a, b, c = (torch.randn(64, 16) for _ in range(3))
    d_ac = q(a, c)
    d_abc = q(a, b) + q(b, c)
    assert (d_ac <= d_abc + 1e-4).all()


# ---- monotone certificates (MATH_AUDIT A5 guard) ------------------------

def test_monotone_coords_nonincreasing_incl_promotion():
    import chess
    from catspace.data.encode import encode_meta, encode_packed
    from catspace.nn.monotone_coords import monotone_coords
    seqs = [
        (chess.Board(), ["e4", "d5", "exd5", "Qxd5"]),                 # captures
        (chess.Board("8/P6k/8/8/8/8/8/7K w - - 0 1"), ["a8=Q"]),      # promotion
        (chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"), ["exd6"]), # en passant
        (chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1"), ["O-O"]),     # castling
    ]
    for b, moves in seqs:
        prev = monotone_coords(encode_packed(b)[None], encode_meta(b)[None])[0]
        for mv in moves:
            b.push_san(mv)
            cur = monotone_coords(encode_packed(b)[None], encode_meta(b)[None])[0]
            assert (cur <= prev + 1e-6).all(), (mv, prev, cur)
            prev = cur
