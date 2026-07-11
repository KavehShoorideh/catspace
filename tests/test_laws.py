"""
The README's headline invariant: plies-to-mate >= DTM against optimal
defense (DTM is defined as the minimum, by construction of retrograde value
iteration -- no policy can beat it). A past sign-flip bug let "optimal" black
cooperate, producing mates FASTER than DTM, which is mathematically
impossible; this is the regression test for that class of bug.
"""
import numpy as np
import pytest

from latentchess.domains import krk
from latentchess.opponents import optimal_reply_table, EpsOptimalDTM
from latentchess.planner.policy import DTMOraclePolicy, EpsGreedy
from latentchess.game import play_game
from latentchess.arena import evaluate, tempo_ratio


@pytest.fixture(scope="module")
def krk_setup():
    chain = krk.build_chain()
    W, B = krk.enumerate_states()
    dtm_w, _ = krk.compute_dtm(W, B)
    return chain, dtm_w


def test_oracle_vs_optimal_black_hits_dtm_exactly(krk_setup):
    chain, dtm_w = krk_setup
    table = optimal_reply_table(chain, dtm_w)
    black = EpsOptimalDTM(table, eps=0.0)
    white = DTMOraclePolicy(chain, dtm_w)
    rng = np.random.default_rng(7)
    starts = rng.integers(0, chain.n_live, size=200)
    for s0 in starts:
        rec = play_game(chain, white, black, int(s0), cap=60, rng=rng)
        assert rec.result == "mate"
        white_moves = len(rec.states)
        assert white_moves == int(np.ceil(dtm_w[s0] / 2))


def test_no_policy_beats_dtm_vs_optimal_black(krk_setup):
    """A noisy (occasionally-random) policy can never mate faster than the
    DTM oracle -- only match it or take longer."""
    chain, dtm_w = krk_setup
    table = optimal_reply_table(chain, dtm_w)
    black = EpsOptimalDTM(table, eps=0.0)
    noisy_white = EpsGreedy(DTMOraclePolicy(chain, dtm_w), eps=0.3)
    rng = np.random.default_rng(11)
    starts = rng.integers(0, chain.n_live, size=200)
    for s0 in starts:
        rec = play_game(chain, noisy_white, black, int(s0), cap=80, rng=rng)
        if rec.result != "mate":
            continue
        white_moves = len(rec.states)
        assert white_moves >= int(np.ceil(dtm_w[s0] / 2))


def test_oracle_converts_100_percent_of_won_states(krk_setup):
    chain, dtm_w = krk_setup
    table = optimal_reply_table(chain, dtm_w)
    black = EpsOptimalDTM(table, eps=0.0)
    white = DTMOraclePolicy(chain, dtm_w)
    rng = np.random.default_rng(3)
    starts = rng.integers(0, chain.n_live, size=300)
    result = evaluate(chain, dtm_w, white, black, starts, cap=60)
    assert result.conversion == 1.0
    assert result.exact_dtm_rate == 1.0
    assert result.tempo == pytest.approx(1.0)


def test_tempo_ratio_is_at_least_one_vs_optimal(krk_setup):
    chain, dtm_w = krk_setup
    table = optimal_reply_table(chain, dtm_w)
    black = EpsOptimalDTM(table, eps=0.0)
    noisy_white = EpsGreedy(DTMOraclePolicy(chain, dtm_w), eps=0.5)
    rng = np.random.default_rng(13)
    starts = rng.integers(0, chain.n_live, size=200)
    result = evaluate(chain, dtm_w, noisy_white, black, starts, cap=100)
    assert result.tempo >= 1.0 - 1e-9
