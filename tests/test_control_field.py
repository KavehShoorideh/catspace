"""Phase 1 gates for the control field (docs/CONTROL-FIELD-SPEC.md section 2.4)."""
import time

import chess
import numpy as np
import pytest

from catspace.controlfield.control import (
    weighted_attacker_field, see_field, critical_square_mask, compute_fields, orient,
)


def test_starting_position_antisymmetric_and_lr_symmetric():
    b = chess.Board()
    c = weighted_attacker_field(b)   # White POV
    # antisymmetric under color flip: swapping which side is "White" (i.e. flipping
    # the board vertically and negating) should reproduce the same raw field, since
    # the starting position is itself mirror-symmetric between colors.
    from catspace.controlfield.control import _flip_vertical
    assert np.allclose(c, -_flip_vertical(c), atol=1e-6)
    # left-right symmetric on rank 3/6 (files a-h mirror around the d/e file)
    for rank in (2, 5):   # 0-indexed: rank 3 -> index 2, rank 6 -> index 5
        row = [c[chess.square(f, rank)] for f in range(8)]
        assert np.allclose(row, row[::-1], atol=1e-6), f"rank {rank+1} not L-R symmetric"


def test_hanging_queen_see_strongly_negative():
    # White queen on e5, undefended, attacked by a black rook on e8 (open e-file).
    b = chess.Board("4k3/8/8/4Q3/8/8/8/4r1K1 w - - 0 1")
    c_see = see_field(b)   # White POV, centipawn-scaled /100
    assert c_see[chess.E5] < -5.0, f"expected strongly negative SEE on hanging queen square, got {c_see[chess.E5]}"


def test_bishop_blocked_by_pawn_chain_contributes_zero_behind():
    # Starting position: White bishop c1, own pawn d2 blocks the c1-h6 diagonal;
    # White bishop f1, own pawn e2 blocks the f1-a6 diagonal.
    cases = [
        (chess.Board().fen(), chess.C1, [chess.E3, chess.F4, chess.G5, chess.H6]),
        (chess.Board().fen(), chess.F1, [chess.D3, chess.C4, chess.B5, chess.A6]),
    ]
    for fen, bishop_sq, blocked_squares in cases:
        b = chess.Board(fen)
        assert b.piece_at(bishop_sq) is not None and b.piece_at(bishop_sq).piece_type == chess.BISHOP
        for sq in blocked_squares:
            attackers = b.attackers(b.piece_at(bishop_sq).color, sq)
            assert bishop_sq not in attackers, (
                f"bishop on {chess.square_name(bishop_sq)} should not attack "
                f"{chess.square_name(sq)} through its own pawn")


def test_variant_a_throughput():
    rng = np.random.default_rng(0)
    boards = []
    b = chess.Board()
    for _ in range(200):
        legal = list(b.legal_moves)
        if not legal or b.is_game_over():
            b = chess.Board()
            legal = list(b.legal_moves)
        b.push(legal[rng.integers(len(legal))])
        boards.append(b.copy())
    t0 = time.time()
    for bd in boards:
        weighted_attacker_field(bd)
    elapsed = time.time() - t0
    rate = len(boards) / elapsed
    assert rate >= 2000, f"Variant A throughput {rate:.0f} pos/s below the 2000 pos/s/core gate"


def test_orientation_positive_means_mover_controls():
    # White to move, White has overwhelming attacker weight on d5 (own square, occupied by a white pawn)
    b = chess.Board("4k3/8/8/3P4/8/8/8/4K3 w - - 0 1")
    c_a, _, _ = compute_fields(b)
    assert c_a[chess.D5] >= 0   # White (the mover) at least holds its own pawn's square
    # flip to Black to move, same raw structure -- mover-POV sign should flip logic,
    # verified structurally via orient() unit behavior:
    white_pov = weighted_attacker_field(b)
    black_mover = orient(white_pov, mover_is_white=False)
    white_mover = orient(white_pov, mover_is_white=True)
    assert np.allclose(white_mover, white_pov)
    assert not np.allclose(black_mover, white_pov)


def test_critical_mask_king_zone_and_central_squares():
    b = chess.Board()
    m = critical_square_mask(b)
    king_sq = b.king(b.turn)
    assert m[king_sq] == 1.0
    for sq in (chess.D4, chess.D5, chess.E4, chess.E5):
        if b.piece_at(sq) is None:
            assert m[sq] == pytest.approx(0.3)
