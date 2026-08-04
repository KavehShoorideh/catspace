"""tests/test_cascade.py — DecisionCascade (planner/cascade.py, phase B).

Synthetic reach oracles; every claimed position property is asserted with
python-chess in the test itself."""
from __future__ import annotations

import chess
import numpy as np

from catspace.research.components.search.approaches.puct_mcts.src.mcts import MCTS
from catspace.research.components.planner.approaches.subgoal_cascade.src.cascade import Candidate, Decision, DecisionCascade

MATE_IN_1 = "6k1/8/6K1/Q7/8/8/8/8 w - - 0 1"          # Qd8# (verified in test_probe)
ROOMY = "2k5/8/8/8/8/8/R6P/2K3R1 w - - 0 1"
DRAW_PIN = "8/8/8/8/8/8/r7/K1k5 w - - 0 1"            # only Kxa2 -> bare kings
MATED = "6k1/6Q1/5K2/8/8/8/8/8 b - - 0 1"             # Black checkmated


def flat_reach(boards):
    return np.zeros(len(boards))


def cand(name, reach=flat_reach, budget=64):
    return Candidate(name, MCTS(reach, max_nodes=budget, cache={}))


def test_mate_in_1_certifies_at_coarse_tier():
    b = chess.Board(MATE_IN_1)
    casc = DecisionCascade([cand("W"), cand("D")], coarse_budget=32)
    d = casc.decide(b)
    assert d.certified and d.action == "move" and d.rounds == 0
    b2 = b.copy()
    b2.push(d.move)
    assert b2.is_checkmate()
    # energy: both coarse probes together stay near two root expansions
    assert d.evals_spent <= 2 * (len(list(b.legal_moves)) + 32)


def test_uncertifiable_falls_back_to_point_estimate_within_budget():
    b = chess.Board(ROOMY)
    casc = DecisionCascade([cand("A"), cand("B")], coarse_budget=24,
                           deepen_budget=24, max_rounds=2)
    d = casc.decide(b)
    assert not d.certified and d.action == "move" and d.move is not None
    assert d.rounds == 2
    branch = len(list(b.legal_moves)) * 2
    assert d.evals_spent <= 2 * 24 + 2 * 2 * 24 + 6 * branch   # budgets + overshoot


def test_checkmated_root_resigns():
    b = chess.Board(MATED)
    assert b.is_checkmate()
    d = DecisionCascade([cand("W")], coarse_budget=8).decide(b)
    assert d.action == "resign" and d.move is None and d.certified


def test_certified_draw_ceiling_offers_draw():
    b = chess.Board(DRAW_PIN)
    d = DecisionCascade([cand("W"), cand("D")], coarse_budget=8).decide(b)
    assert d.action == "offer_draw"
    assert d.move is not None and b.is_capture(d.move)   # still plays Kxa2
    assert d.certified


def _pr(value, lo, hi):
    from catspace.research.components.planner.approaches.subgoal_cascade.src.probe import ProbeResult
    return ProbeResult(value=value, best_move=None, lo=lo, hi=hi)


def test_leader_selection_point_estimate_and_pov():
    """Vacuous intervals => point-estimate leader, mover-POV correct."""
    casc = DecisionCascade([cand("A"), cand("B")])
    res = {"A": _pr(0.4, -1.0, 1.0), "B": _pr(-0.2, -1.0, 1.0)}
    leader, certified = casc._leader(res, white=True)
    assert leader == "A" and not certified
    leader, certified = casc._leader(res, white=False)   # Black prefers -0.2
    assert leader == "B" and not certified


def test_leader_selection_certified_dominance():
    """A's certified lower bound clears B's upper bound => certified, and the
    point estimates are irrelevant (B even LOOKS better uncertified)."""
    casc = DecisionCascade([cand("A"), cand("B")], eps=0.05)
    res = {"A": _pr(0.1, 0.3, 1.0), "B": _pr(0.9, -1.0, 0.2)}
    leader, certified = casc._leader(res, white=True)
    assert leader == "A" and certified


def test_single_candidate_not_vacuously_certified():
    """Review 2026-07-18 MED regression: with no rivals, certified must mean
    the interval is decided, not that nobody contested it."""
    b = chess.Board(ROOMY)
    d = DecisionCascade([cand("only")], coarse_budget=16).decide(b)
    assert not d.certified                            # interval is vacuous here
    d2 = DecisionCascade([cand("only")], coarse_budget=8).decide(chess.Board(MATED))
    assert d2.certified                               # pinned interval stays certified
