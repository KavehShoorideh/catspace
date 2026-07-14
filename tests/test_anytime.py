"""Model-free tests for the anytime path-to-mate search (catspace/nn/anytime.py):
synthetic reach_fn, real chess rules."""
import chess
import numpy as np
import pytest

from catspace.nn.anytime import AnytimePathSearch


def flat_reach(boards):
    return np.zeros(len(boards))


def king_box_reach(boards):
    """Informative synthetic field: fewer safe squares around the black king
    = closer to mate. (The anytime search is direction-guided BY DESIGN --
    with a flat field it is blind, and that's the correct behavior.)"""
    out = []
    for b in boards:
        k = b.king(chess.BLACK)
        adj = list(b.attacks(k))                       # king adjacency squares
        safe = [s for s in adj if not b.is_attacked_by(chess.WHITE, s)
                and b.color_at(s) != chess.BLACK]
        out.append(-float(len(safe)))
    return np.array(out)


def test_takes_mate_in_one_even_if_field_hates_it():
    # the incumbent bound must beat the heuristic: reach ranks OTHER moves
    # high, but Ra8# is found during expansion and nothing shorter exists
    b = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    s = AnytimePathSearch(flat_reach, max_nodes=64)
    m = s.search(b)
    b.push(m)
    assert b.is_checkmate()
    assert s.incumbent_plies == 1


def test_finds_mate_in_two_line():
    # rook ladder: Ra2-a7 (cut the 7th), then Rb1-b8# -- the king-box field
    # points at the cut, the search certifies the line
    b = chess.Board("7k/8/8/8/8/8/R7/1R4K1 w - - 0 1")
    s = AnytimePathSearch(king_box_reach, max_nodes=800)
    m = s.search(b)
    assert s.incumbent is not None, "should certify some mate line"
    assert s.incumbent_plies <= 5
    assert m in list(b.legal_moves)


def test_budget_respected():
    b = chess.Board("7k/8/8/8/8/8/R7/1R4K1 w - - 0 1")
    s = AnytimePathSearch(flat_reach, max_nodes=100)
    s.search(b)
    # may finish one in-flight expansion; never a whole extra level
    assert s.evals_used <= 100 + 60


def test_deterministic():
    b = chess.Board("6k1/5pp1/7p/8/8/6Q1/5PPP/6K1 w - - 0 1")
    a = AnytimePathSearch(flat_reach, max_nodes=300).search(b)
    c = AnytimePathSearch(flat_reach, max_nodes=300).search(b)
    assert a == c


def test_no_mate_returns_reasonable_move():
    b = chess.Board()                    # startpos: no mate in reach
    s = AnytimePathSearch(flat_reach, max_nodes=120)
    m = s.search(b)
    assert m in list(b.legal_moves)
    assert s.incumbent is None


def test_incumbent_improves_with_budget():
    # more budget can only shorten (or keep) the certified line
    b = chess.Board("7k/8/8/8/8/8/R7/1R4K1 w - - 0 1")
    small = AnytimePathSearch(king_box_reach, max_nodes=150)
    small.search(b)
    big = AnytimePathSearch(king_box_reach, max_nodes=1500)
    big.search(b)
    if small.incumbent is not None:
        assert big.incumbent is not None
        assert big.incumbent_plies <= small.incumbent_plies


def test_no_legal_moves_raises():
    b = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")    # stalemate
    with pytest.raises(ValueError):
        AnytimePathSearch(flat_reach, max_nodes=10).search(b)
