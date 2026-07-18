"""tests/test_probe.py — the probe() primitive (planner/probe.py, phase A).

All positions are sanity-checked with python-chess inside the tests (board
legality + the claimed mate really is mate) — FEN-by-hand burned us before.
The reach oracle is synthetic (flat or shaped) so tests are CPU-instant and
independent of any checkpoint."""
from __future__ import annotations

import chess
import numpy as np
import pytest

from catspace.nn.mcts import DRAW_V, MATE_V, MATED_V, MCTS
from catspace.planner.probe import ProbeResult, deepen, probe


def flat_reach(boards):
    return np.zeros(len(boards))


def make_mcts(budget=64, **kw):
    return MCTS(flat_reach, max_nodes=budget, cache={}, **kw)


# ---------------------------------------------------------------- mate-in-1
MATE_IN_1 = "6k1/8/6K1/Q7/8/8/8/8 w - - 0 1"      # Qa5-d8# (back rank, Kg6 guards)


def test_mate_in_1_position_is_real():
    b = chess.Board(MATE_IN_1)
    assert b.is_valid()
    b2 = b.copy()
    b2.push_san("Qd8")
    assert b2.is_checkmate()


def test_probe_certifies_mate_in_1():
    r = probe(make_mcts(), chess.Board(MATE_IN_1))
    assert r.best_move is not None
    b2 = chess.Board(MATE_IN_1)
    b2.push(r.best_move)
    assert b2.is_checkmate()
    # certified lower bound pins to the discounted mate value; hi stays MATE_V
    assert r.lo == pytest.approx(MATE_V, abs=1e-3)
    assert r.lo <= r.hi <= MATE_V
    assert "mate_w" in r.hits


def test_probe_black_mate_in_1_bounds_hi():
    fen = "6k1/8/8/8/8/7r/6r1/K7 b - - 0 1"       # Rh1# (ladder)
    b = chess.Board(fen)
    assert b.is_valid()
    b2 = b.copy()
    b2.push_san("Rh1")
    assert b2.is_checkmate()
    r = probe(make_mcts(), b)
    assert r.hi == pytest.approx(MATED_V, abs=1e-3)   # Black may simply mate
    assert r.lo == MATED_V


# ------------------------------------------------------------- proven draws
def test_all_children_terminal_pins_interval():
    """White in check, exactly ONE legal move (Kxa2), which bares the kings:
    every child is a game-truth draw => the interval pins at DRAW_V."""
    fen = "8/8/8/8/8/8/r7/K1k5 w - - 0 1"
    b = chess.Board(fen)
    assert b.is_valid()
    moves = list(b.legal_moves)
    assert len(moves) == 1 and b.is_capture(moves[0])
    b2 = b.copy()
    b2.push(moves[0])
    assert b2.is_insufficient_material()
    r = probe(make_mcts(budget=16), b)
    assert r.lo == r.hi == pytest.approx(DRAW_V, abs=1e-9)
    assert r.decided


def test_checkmated_root_reports_point_interval():
    fen = "6k1/6Q1/5K2/8/8/8/8/8 b - - 0 1"        # Black is mated (Qg7#)
    b = chess.Board(fen)
    assert b.is_valid() and b.is_checkmate()
    r = probe(make_mcts(budget=4), b)
    assert r.best_move is None
    assert r.lo == r.hi == pytest.approx(MATE_V, abs=1e-6)
    assert r.decided


# --------------------------------------------------- certification hygiene
def test_confident_recognizer_never_certifies():
    """A recognizer that is CONFIDENTLY WRONG must not tighten [lo, hi] —
    raw network confidence may propose, never close (the 0.60->0.20 rule)."""
    def wrong_certainty(boards):
        return (np.full(len(boards), 0.9),        # "White is basically winning"
                np.full(len(boards), 0.99))       # ... with 99% confidence

    fen = "8/8/8/3k4/8/3K4/3R4/8 w - - 0 1"       # KRvK, no mate-in-1
    b = chess.Board(fen)
    assert b.is_valid()
    m = MCTS(flat_reach, max_nodes=64, cache={},
             certainty_fn=wrong_certainty, certainty_stop=0.95)
    r = probe(m, b)
    assert r.lo == MATED_V and r.hi == MATE_V     # vacuous: nothing proven
    assert r.hits.get("resolved", 0) > 0          # the soft-terminal DID fire


# ----------------------------------------------------------- budget honesty
def test_budget_respected_and_deepen_reuses():
    fen = "2k5/8/8/8/8/8/R6P/2K3R1 w - - 0 1"     # roomy middlegame-ish
    b = chess.Board(fen)
    assert b.is_valid()
    m = make_mcts()
    r1 = probe(m, b, budget=40)
    max_branch = max(len(list(b.legal_moves)) * 2, 40)
    assert r1.evals_spent <= 40 + max_branch      # overshoot ≤ one expansion batch
    n1 = r1.tree.N
    r2 = deepen(m, b, r1, extra_budget=40)
    assert r2.tree is r1.tree                     # same tree object continued
    assert r2.tree.N > n1                         # visits accumulated
    assert r2.evals_spent <= 40 + max_branch      # deepen's OWN budget, not cumulative
    assert m.max_nodes == 64                      # budget override restored


def test_hits_are_visit_weighted_leaves():
    r = probe(make_mcts(), chess.Board(MATE_IN_1))
    assert sum(r.hits.values()) > 0
    assert all(v > 0 for v in r.hits.values())


def test_visit_top2_orders():
    r = probe(make_mcts(), chess.Board("2k5/8/8/8/8/8/R6P/2K3R1 w - - 0 1"))
    assert r.visit_top2[0] >= r.visit_top2[1] >= 0


# ----------------------------------------------------- decision-stability stop
def test_mate_stop_saves_budget():
    """decision_stop: a game-truth mate at the root ends the search after the
    root expansion — the certified stop. Without the flag the full budget is
    spent on an already-decided move."""
    b = chess.Board(MATE_IN_1)
    m_off = make_mcts(budget=200)
    r_off = probe(m_off, b)
    m_on = MCTS(flat_reach, max_nodes=200, cache={}, decision_stop=True)
    r_on = probe(m_on, b)
    assert r_on.best_move == r_off.best_move          # same (mating) move
    n_moves = len(list(b.legal_moves))
    assert r_on.evals_spent <= n_moves                # root expansion only
    assert r_off.evals_spent > 3 * r_on.evals_spent   # flagless burns the budget


def test_cert_planted_value_does_not_trigger_mate_stop():
    """A recognizer-planted terminal_v > 0.5 must NOT fire the certified
    mate-stop (network confidence is not game truth)."""
    def confident(boards):
        return (np.full(len(boards), 0.9), np.full(len(boards), 0.99))

    fen = "8/8/8/3k4/8/3K4/3R4/8 w - - 0 1"           # KRvK, no mate-in-1
    b = chess.Board(fen)
    m = MCTS(flat_reach, max_nodes=64, cache={},
             certainty_fn=confident, certainty_stop=0.95, decision_stop=True)
    r = probe(m, b)
    # every child cert-resolves to terminal => sims are eval-free; the search
    # must NOT have quit after the recognizer pass alone with a "mate" readout
    assert r.lo == MATED_V and r.hi == MATE_V


def test_stability_stop_ends_early_on_lopsided_field():
    """A reach field that overwhelmingly prefers one child concentrates
    visits; once the gap exceeds the remaining budget the search stops."""
    def lopsided(boards):
        return np.array([100.0 if b.piece_at(chess.D8) else 0.0 for b in boards])

    b = chess.Board(MATE_IN_1)                        # queen can reach d8
    m_on = MCTS(lopsided, max_nodes=400, cache={}, decision_stop=True)
    m_on.run(b)
    used_on = m_on.evals_used
    m_off = MCTS(lopsided, max_nodes=400, cache={})
    m_off.run(b)
    assert used_on < m_off.evals_used                 # stopped before the budget
