"""The mirror lemma (lean/ideas/Ideas/Mirror.lean, proved: legalMove_pos, applyMove_pos, forcedWithin_mirror) checked
against python-chess and the tablebase: without castling rights, the left-right reflection maps legal moves to legal moves
and keeps every result -- with pawns too."""
import random

import chess
import chess.syzygy
import pytest

from catspace.io import paths


def _random(rng, pieces):
    while True:
        b = chess.Board(None)
        for s, (k, c) in zip(rng.sample(range(64), len(pieces)), pieces):
            if k == chess.PAWN and chess.square_rank(s) in (0, 7):
                break
            b.set_piece_at(s, chess.Piece(k, c))
        else:
            b.turn = rng.choice((chess.WHITE, chess.BLACK))
            if b.is_valid():
                return b


def test_reflected_moves_are_the_reflected_position_s_moves():
    rng = random.Random(0)
    pieces = [(6, 1), (6, 0), (1, 1), (1, 1), (1, 0), (4, 1), (3, 0)]
    for _ in range(300):
        b = _random(rng, pieces)
        m = b.transform(chess.flip_horizontal)
        flip = lambda mv: chess.Move(mv.from_square ^ 7, mv.to_square ^ 7, mv.promotion)   # a <-> h
        assert {flip(x) for x in b.legal_moves} == set(m.legal_moves)
        for x in list(b.legal_moves)[:5]:                                   # playing commutes with the reflection
            a = b.copy(stack=False); a.push(x)
            c = m.copy(stack=False); c.push(flip(x))
            assert a.transform(chess.flip_horizontal).board_fen() == c.board_fen()


@pytest.mark.skipif(not paths.syzygy_dir().exists(), reason="no tablebases")
def test_reflected_results_agree():
    rng = random.Random(1)
    with chess.syzygy.open_tablebase(str(paths.syzygy_dir())) as tb:
        for pieces in ([(6, 1), (6, 0), (1, 1)], [(6, 1), (6, 0), (1, 1), (1, 0)], [(6, 1), (6, 0), (4, 1), (1, 1), (4, 0)]):
            for _ in range(200):
                b = _random(rng, pieces)
                assert tb.probe_wdl(b) == tb.probe_wdl(b.transform(chess.flip_horizontal))
