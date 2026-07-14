"""Model-free tests for the PUCT MCTS core (catspace/nn/mcts.py): synthetic
reach_fn, real chess rules. The FB-checkpoint wrapper is covered by the
playout_ab smoke, not here."""
import chess
import numpy as np
import pytest

from catspace.nn.mcts import DRAW_V, MATE_V, PLY_DISCOUNT, MCTS


def flat_reach(boards):
    return np.zeros(len(boards))


def make(reach=flat_reach, nodes=64, **kw):
    return MCTS(reach, max_nodes=nodes, **kw)


def test_white_takes_mate_in_one():
    # back-rank: Ra1-a8 is mate
    b = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    m = make().best_move(b)
    b.push(m)
    assert b.is_checkmate()


def test_black_takes_mate_in_one():
    # mirrored back-rank for Black
    b = chess.Board("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1")
    m = make().best_move(b)
    b.push(m)
    assert b.is_checkmate()


def test_avoids_stalemating_when_no_mate():
    # Qc7 stalemates Black (Ka8, no moves, not in check); no mate-in-1 exists
    b = chess.Board("k7/8/8/1K6/8/8/2Q5/8 w - - 0 1")
    stalemate = chess.Move.from_uci("c2c7")
    b2 = b.copy(stack=False)
    b2.push(stalemate)
    assert b2.is_stalemate()          # the trap is real
    assert make(nodes=128).best_move(b) != stalemate


def test_budget_respected_and_counted():
    b = chess.Board()                 # startpos, branching 20
    t = make(nodes=100)
    t.best_move(b)
    # may overshoot by at most one expansion's branching, never a full level
    assert 100 <= t.evals_used <= 100 + 40
    small = make(nodes=25)
    small.best_move(b)
    assert small.evals_used < t.evals_used


def test_deterministic():
    b = chess.Board("6k1/5pp1/7p/8/8/6Q1/5PPP/6K1 w - - 0 1")
    assert make(nodes=200).best_move(b) == make(nodes=200).best_move(b)


def test_visits_concentrate_on_high_reach_move():
    # reach oracle that loves positions where White's queen is on h5
    def reach(boards):
        return np.array([2.0 if bd.piece_at(chess.H5) is not None
                         and bd.piece_at(chess.H5).piece_type == chess.QUEEN
                         else 0.0 for bd in boards])
    b = chess.Board("6k1/5pp1/7p/8/8/8/5PPP/3Q2K1 w - - 0 1")
    t = make(reach, nodes=300)
    root = t.run(b)
    best = max(root.children, key=lambda c: c.N)
    assert best.move == chess.Move.from_uci("d1h5")


def test_terminal_values_and_discount():
    t = make()
    root = t.run(chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"))
    mate_child = next(c for c in root.children
                      if c.move == chess.Move.from_uci("a1a8"))
    assert mate_child.terminal_v == pytest.approx(MATE_V - PLY_DISCOUNT)
    draws = [c for c in root.children if c.terminal_v == DRAW_V]
    assert all(c.terminal_v < 0 for c in draws)


def test_single_legal_move():
    # in check from the (rook-protected) Qh2: Kf1 is the only legal move
    b = chess.Board("6kr/8/8/8/8/8/5PPq/6K1 w - - 0 1")
    legal = list(b.legal_moves)
    assert len(legal) == 1
    assert make(nodes=8).best_move(b) == legal[0]


def test_no_legal_moves_raises():
    b = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # stalemate, Black to move
    assert b.is_stalemate()
    with pytest.raises(ValueError):
        make().best_move(b)
