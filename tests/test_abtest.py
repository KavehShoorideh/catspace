"""
A/B e-test harness tests: EValueTest null-control (false-stop rate bounded
by alpha) and effect-detection, confidence-sequence coverage, matched-seed
determinism, and a smoke comparison on KRk (oracle strictly beats random).
"""
import numpy as np
import pytest

from latentchess.abtest import EValueTest, MethodSpec, compare, confidence_sequence, paired_eval
from latentchess.chain import exact_P
from latentchess.cone.tabular import TabularFB
from latentchess.cone.embedding import make_goal, reach
from latentchess.domains import krk
from latentchess.opponents import EpsOptimalDTM, RandomOpponent, optimal_reply_table
from latentchess.planner.policy import DTMOraclePolicy, RandomPolicy, TablePolicy
from latentchess.planner.readout import ReplyAgg, greedy_policy
from latentchess.scoring import TerminalScores, fill_terminal_state_scores


def test_evalue_null_control():
    rng = np.random.default_rng(0)
    n_reps = 200
    false_stops = 0
    for rep in range(n_reps):
        test = EValueTest()
        diffs = rng.choice([-1, 1], size=300)
        stopped = False
        for d in diffs:
            test.update(float(d))
            if test.reject_at(0.05):
                stopped = True
                break
        false_stops += stopped
    assert false_stops / n_reps <= 0.07


def test_evalue_detects_effect():
    rng = np.random.default_rng(1)
    n_reps = 50
    rejected_within_150 = 0
    rejected_within_400 = 0
    for rep in range(n_reps):
        test = EValueTest()
        diffs = np.where(rng.random(400) < 0.75, 1, -1)
        stop_at = None
        for i, d in enumerate(diffs):
            test.update(float(d))
            if test.reject_at(0.05):
                stop_at = i + 1
                break
        if stop_at is not None:
            rejected_within_400 += 1
            if stop_at <= 150:
                rejected_within_150 += 1
    assert rejected_within_400 >= n_reps - 1
    assert rejected_within_150 >= n_reps // 2


def test_evalue_ties_ignored():
    test = EValueTest()
    for _ in range(50):
        test.update(0.0)
    assert test.e == 1.0
    assert test.n == 0


def test_paired_matched_seeds_zero_diff():
    chain = krk.build_chain()
    W, B = krk.enumerate_states()
    dtm, _ = krk.compute_dtm(W, B)
    table = np.zeros(chain.n_live, dtype=np.int32)
    spec = MethodSpec("same", lambda: TablePolicy(table))
    starts = np.arange(0, 20)
    outcomes = paired_eval(chain, dtm, spec, spec, lambda: RandomOpponent(), starts, cap=40)
    for o in outcomes:
        assert o.a["win"] == o.b["win"]
        assert o.a["moves"] == o.b["moves"]


def test_confidence_sequence_covers():
    rng = np.random.default_rng(2)
    n_sims = 100
    covered = 0
    true_mean = 0.2
    for _ in range(n_sims):
        bern = rng.random(400) < 0.6
        diffs = np.where(bern, 1.0, -1.0)
        lo, hi = confidence_sequence(diffs, alpha=0.05)
        if lo <= true_mean <= hi:
            covered += 1
    assert covered >= 90

    diffs = rng.choice([-1.0, 1.0], size=400)
    eps_small = confidence_sequence(diffs[:50], alpha=0.05)
    eps_large = confidence_sequence(diffs, alpha=0.05)
    width_small = eps_small[1] - eps_small[0]
    width_large = eps_large[1] - eps_large[0]
    assert width_large < width_small


def test_compare_smoke():
    chain = krk.build_chain()
    W, B = krk.enumerate_states()
    dtm, _ = krk.compute_dtm(W, B)
    oracle_spec = MethodSpec("oracle", lambda: DTMOraclePolicy(chain, dtm))
    random_spec = MethodSpec("random", lambda: RandomPolicy())
    methods = {"oracle": oracle_spec, "random": random_spec}

    black_builder = lambda: RandomOpponent()
    rng = np.random.default_rng(0)
    won = np.isfinite(dtm)
    starts = rng.choice(np.where(won)[0], size=30, replace=False)

    rows = compare(chain, dtm, methods, black_builder, starts, alpha=0.05, batch=10, base_seed=0)
    assert len(rows) == 1
    row = rows[0]
    assert row.rejected
    oracle_wins = row.a_wins if row.method_a == "oracle" else row.b_wins
    random_wins = row.b_wins if row.method_a == "oracle" else row.a_wins
    assert oracle_wins > random_wins
