"""
Tests for monotonicity hard negatives (nn/hard_negatives.py). CPU-only, so
they run without touching a training GPU.
"""
import numpy as np
import pytest

import chess

from catspace.data.encode import encode_packed
from catspace.nn.hard_negatives import piece_count, unreachable_goals

torch = pytest.importorskip("torch")
from catspace.nn.hard_negatives import repel_loss


def test_negatives_strictly_increase_count():
    # count(neg) = count(anchor)+1 => provably unreachable by monotonicity
    fens = ["2b1k3/3p4/8/8/8/8/8/R3K2R w - -",
            "8/8/8/4k3/8/K7/R7/R7 w - -",
            chess.Board().fen()]
    packed = np.stack([encode_packed(chess.Board(f)) for f in fens])
    neg = unreachable_goals(packed, seed=3)
    assert np.array_equal(piece_count(neg), piece_count(packed) + 1)


def test_negative_is_a_valid_superset_of_the_anchor():
    # every anchor bit survives; exactly one new bit appears on an empty square
    packed = np.stack([encode_packed(chess.Board("8/8/8/4k3/8/K7/R7/R7 w - -"))])
    neg = unreachable_goals(packed, seed=0)
    occ_a = np.bitwise_or.reduce(packed[0].astype(np.uint64))
    occ_n = np.bitwise_or.reduce(neg[0].astype(np.uint64))
    # anchor occupancy is a subset of negative occupancy
    assert int(occ_a) & int(occ_n) == int(occ_a)
    # exactly one added square
    assert bin(int(occ_n) & ~int(occ_a)).count("1") == 1


def test_added_piece_is_never_a_pawn_or_king():
    # planes 0/5/6/11 are W/B pawn/king; a monotonicity negative must not add
    # a pawn (rank-1/8 legality) or a second king
    packed = np.stack([encode_packed(chess.Board(chess.Board().fen()))])
    for seed in range(20):
        neg = unreachable_goals(packed, seed=seed)
        for plane in (0, 5, 6, 11):
            assert int(neg[0, plane]) == int(packed[0, plane])


def test_repel_loss_is_zero_when_already_far_and_positive_when_close():
    far = torch.tensor([2.0, 3.0, 2.5])
    close = torch.tensor([0.1, 0.2, 0.0])
    assert float(repel_loss(far, margin=1.5)) == 0.0
    assert float(repel_loss(close, margin=1.5)) > 0.0
