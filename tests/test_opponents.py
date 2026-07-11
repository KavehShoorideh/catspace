import numpy as np
import pytest

from latentchess.domains import krk
from latentchess.opponents import optimal_reply_table, RandomOpponent, EpsOptimalDTM, TableOpponent
from latentchess.chain import KIND_ONGOING


@pytest.fixture(scope="module")
def krk_setup():
    chain = krk.build_chain()
    W, B = krk.enumerate_states()
    dtm_w, _ = krk.compute_dtm(W, B)
    return chain, dtm_w


def test_optimal_black_maximizes_dtm(krk_setup):
    """The sign-flip regression: optimal black must MAXIMIZE white's distance
    to mate (delay), never minimize it. For every ongoing move, the chosen
    reply's dtm_filled value must equal the max over all replies."""
    chain, dtm_w = krk_setup
    from latentchess.scoring import dtm_filled
    table = optimal_reply_table(chain, dtm_w)
    dtm_full = dtm_filled(dtm_w, chain.n)
    rng = np.random.default_rng(0)
    sample_mids = rng.integers(0, chain.n_moves, size=500)
    for mid in sample_mids:
        if chain.move_kind[mid] != KIND_ONGOING:
            continue
        outs = chain.outs_of(mid)
        chosen = table[mid]
        vals = dtm_full[outs]
        assert vals[chosen] == vals.max()


def test_capture_preferred_when_available(krk_setup):
    """A reply that captures the rook (-> DRAW_S, dtm_filled sentinel = 1e6,
    the largest possible value) must always be selected when present among
    the reply options, since it strictly maximizes delay."""
    chain, dtm_w = krk_setup
    table = optimal_reply_table(chain, dtm_w)
    for mid in range(chain.n_moves):
        if chain.move_kind[mid] != KIND_ONGOING:
            continue
        outs = chain.outs_of(mid)
        capture_positions = np.where(outs == chain.terminals.draw)[0]
        if len(capture_positions):
            assert table[mid] in capture_positions


def test_eps_optimal_zero_is_deterministic(krk_setup):
    chain, dtm_w = krk_setup
    table = optimal_reply_table(chain, dtm_w)
    opp = EpsOptimalDTM(table, eps=0.0)
    rng = np.random.default_rng(1)
    mid = 0
    picks = {opp.reply_index(chain, mid, rng) for _ in range(20)}
    assert picks == {int(table[mid])}


def test_random_opponent_in_range(krk_setup):
    chain, _ = krk_setup
    opp = RandomOpponent()
    rng = np.random.default_rng(2)
    for mid in range(10):
        idx = opp.reply_index(chain, mid, rng)
        assert 0 <= idx < chain.out_counts[mid]
