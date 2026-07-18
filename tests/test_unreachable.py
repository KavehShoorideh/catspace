"""Exactness tests for the directional unreachability oracle
(nn/unreachable.py). Semantics under test: flag => theorem (never flag a
reachable direction); no-flag = unknown (allowed)."""
import chess
import numpy as np

from catspace.data.encode import encode_meta, encode_packed
from catspace.nn.unreachable import provably_unreachable


def _pair(a: chess.Board, b: chess.Board):
    return (encode_packed(a)[None], encode_meta(a)[None],
            encode_packed(b)[None], encode_meta(b)[None])


def flag(a, b) -> bool:
    return bool(provably_unreachable(*_pair(a, b))[0])


def test_pawn_advance_is_one_directional():
    a = chess.Board()
    b = chess.Board(); b.push_san("e4")
    assert not flag(a, b)          # forward: reachable by witness -> must not flag
    assert flag(b, a)              # backward: e4-pawn cannot return to e2


def test_capture_directional_by_counts():
    a = chess.Board("4k3/8/3p4/4P3/8/8/8/4K3 w - - 0 1")
    b = a.copy(); b.push_san("exd6")
    assert not flag(a, b)
    assert flag(b, a)              # captured pawn cannot return


def test_pawn_file_cone():
    # white pawn on a4 can never reach h-file within 4 ranks (|df|=7 > dr<=3):
    a = chess.Board("4k3/8/8/8/P7/8/8/4K3 w - - 0 1")
    g = chess.Board("4k3/8/8/8/7P/8/8/4K3 w - - 0 1")   # pawn h4 instead
    assert flag(a, g) and flag(g, a)   # two-directional impossibility

def test_cone_capture_shift_not_overflagged():
    # e2 pawn CAN reach d3 (one capture-step): |df|=1 <= dr=1 -> must not flag
    a = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    g = chess.Board("4k3/8/8/8/8/3P4/8/4K3 w - - 0 1")
    assert not flag(a, g)
    assert flag(g, a)              # but d3 back to e2 is impossible


def test_promotion_flags_reverse_only():
    a = chess.Board("8/P6k/8/8/8/8/8/7K w - - 0 1")
    b = a.copy(); b.push_san("a8=Q")
    assert not flag(a, b)          # promotion forward: fine (pawn count drops)
    assert flag(b, a)              # pawn cannot be re-created


def test_castling_rights_directional():
    a = chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    b = a.copy(); b.push_san("Rg1")   # rook move forfeits the right
    assert not flag(a, b)
    assert flag(b, a)              # rights never come back


def test_no_false_positive_on_real_game_forward_pairs():
    # soundness on real play: every (s_t -> s_{t+k}) of an actual game is
    # reachable by witness; the oracle must never flag the forward direction.
    b = chess.Board()
    moves = ["e4","e5","Nf3","Nc6","Bb5","a6","Bxc6","dxc6","O-O","f6","d4",
             "exd4","Nxd4","c5","Ne2","Qxd1","Rxd1","Bd7","Nbc3","O-O-O"]
    boards = [b.copy()]
    for m in moves:
        b.push_san(m); boards.append(b.copy())
    for i in range(len(boards)):
        for j in range(i + 1, len(boards)):
            assert not flag(boards[i], boards[j]), (i, j)
