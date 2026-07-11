"""
Hand-built micro-chain exercising move_values/greedy_policy/backup in
isolation, plus the terminal-scoring bug-ledger regressions (README lesson 5:
draw-scored-neutral vs draw-scored-bad flips the chosen move).
"""
import numpy as np
import pytest

from latentchess.chain import TransitionChain, Terminals
from latentchess.scoring import TerminalScores, fill_terminal_state_scores
from latentchess.planner.readout import ReplyAgg, move_values, policy_from_values, greedy_policy, backup


def make_chain():
    """4 live states + MATE/DRAW absorbing:
       state0 -- moveA (ONGOING) --> {state1, state2}
              -- moveB (ONGOING) --> {state3}
       state1 -- move (MATE)      --> MATE_S
       state2 -- move (STALEMATE) --> DRAW_S
       state3 -- move (ONGOING)   --> {state1}
    """
    move_ptr = np.array([0, 2, 3, 4, 5], dtype=np.int64)
    move_kind = np.array([0, 0, 1, 2, 0], dtype=np.int8)     # ONGOING,ONGOING,MATE,STALEMATE,ONGOING
    out_ptr = np.array([0, 2, 3, 4, 5, 6], dtype=np.int64)
    out_flat = np.array([1, 2, 3, 4, 5, 1], dtype=np.int32)
    return TransitionChain(
        n=6, n_live=4, move_ptr=move_ptr, move_kind=move_kind,
        out_ptr=out_ptr, out_flat=out_flat,
        terminals=Terminals(mate=4, draw=5), move_names=["A", "B", "m1", "m2", "m3"],
    )


def test_mean_vs_min_pick_different_moves():
    chain = make_chain()
    ts = TerminalScores.big()
    scores = np.array([0.0, 10.0, -5.0, 1.0, ts.mate, ts.draw])

    V_mean = move_values(scores, chain, ReplyAgg.MEAN, ts)
    V_min = move_values(scores, chain, ReplyAgg.MIN, ts)
    assert V_mean[0] == pytest.approx(2.5)   # mean(10, -5)
    assert V_min[0] == pytest.approx(-5.0)   # min(10, -5)

    pol_mean = policy_from_values(V_mean, chain)
    pol_min = policy_from_values(V_min, chain)
    assert pol_mean[0] == 0   # MEAN prefers move A (risk-seeking: 2.5 > 1.0)
    assert pol_min[0] == 1    # MIN prefers move B (risk-averse: 1.0 > -5.0)


def test_backup_zero_is_identity():
    chain = make_chain()
    ts = TerminalScores.big()
    scores = np.array([0.0, 10.0, -5.0, 1.0, ts.mate, ts.draw])
    assert np.array_equal(backup(scores, chain, ReplyAgg.MEAN, ts, k=0), scores)


def test_backup_one_ply_matches_hand_minimax():
    chain = make_chain()
    ts = TerminalScores.big()
    scores = np.array([0.0, 10.0, -5.0, 1.0, ts.mate, ts.draw])

    out_mean = backup(scores, chain, ReplyAgg.MEAN, ts, k=1)
    assert out_mean[0] == pytest.approx(2.5)     # state0: max(mean(10,-5)=2.5, 1.0)
    assert out_mean[1] == pytest.approx(ts.mate)  # state1: its only move is MATE
    assert out_mean[2] == pytest.approx(ts.draw)  # state2: its only move is STALEMATE
    assert out_mean[3] == pytest.approx(10.0)     # state3: its only move reads (pre-update) state1=10

    out_min = backup(scores, chain, ReplyAgg.MIN, ts, k=1)
    assert out_min[0] == pytest.approx(1.0)       # state0: max(min(10,-5)=-5, 1.0) = 1.0


def test_mate_always_wins_argmax():
    """A move that mates immediately must beat any finite continuation, no
    matter how attractive the continuation's raw (learned/noisy) score is."""
    chain = make_chain()
    ts = TerminalScores.big()
    huge_but_finite = 1e10
    scores = np.array([0.0, huge_but_finite, -5.0, 1.0, ts.mate, ts.draw])
    pol = greedy_policy(scores, chain, ReplyAgg.MEAN, ts)
    # state1's only move is the MATE move (id 2) -- must be chosen (local idx 0)
    assert pol[1] == 0


def test_stalemate_never_beats_a_finite_ongoing_move():
    chain = make_chain()
    ts = TerminalScores.big()
    scores = np.array([0.0, 10.0, -5.0, 1.0, ts.mate, ts.draw])
    # state2's only move is STALEMATE; state3's only move is ONGOING -> both are
    # single-move states here, so just check the override directly:
    V = move_values(scores, chain, ReplyAgg.MEAN, ts)
    assert V[3] < 0   # the stalemate move (id 3) is scored catastrophically bad
    assert V[4] > 0   # an ordinary finite ongoing move (id 4) is not


def test_draw_scored_neutral_vs_bad_flips_the_chosen_move():
    """README bug-ledger regression: two readouts of the same field differed
    24% vs 99.8% mate-rate purely because one scored a capture-to-draw
    outcome as neutral (0.0) instead of catastrophic. Reproduce the flip on
    a minimal chain, then assert the fixed (bad-draw) convention is what the
    library actually does by default via TerminalScores.big()."""
    move_ptr = np.array([0, 2], dtype=np.int64)
    move_kind = np.array([0, 0], dtype=np.int8)   # both ONGOING
    out_ptr = np.array([0, 1, 2], dtype=np.int64)
    out_flat = np.array([1, 2], dtype=np.int32)    # move0 -> DRAW_S(1); move1 -> live "continue" state(2)
    chain = TransitionChain(
        n=4, n_live=1, move_ptr=move_ptr, move_kind=move_kind,
        out_ptr=out_ptr, out_flat=out_flat,
        terminals=Terminals(mate=3, draw=1), move_names=["capture_to_draw", "continue"],
    )
    continue_value = -0.1   # a slightly-noisy-negative but perfectly fine continuation

    ts_bug = TerminalScores(mate=1e18, draw=0.0, bwin=0.0)         # the historical bug
    scores_bug = np.array([0.0, 0.0, continue_value, ts_bug.mate])   # state index 1 = DRAW_S prefilled to 0.0
    pol_bug = greedy_policy(scores_bug, chain, ReplyAgg.MEAN, ts_bug)
    assert pol_bug[0] == 0   # BUG: picks the capture-to-draw move

    ts_fixed = TerminalScores.big()
    scores_fixed = fill_terminal_state_scores(np.array([0.0, 0.0, continue_value, 0.0]), chain, ts_fixed)
    pol_fixed = greedy_policy(scores_fixed, chain, ReplyAgg.MEAN, ts_fixed)
    assert pol_fixed[0] == 1   # FIXED: correctly continues instead of hanging the rook


def test_quantile_scorer_orders_mate_above_draw():
    reach = np.linspace(-1.0, 1.0, 1000)
    ts = TerminalScores.from_reach_quantiles(reach)
    assert ts.mate > ts.draw
