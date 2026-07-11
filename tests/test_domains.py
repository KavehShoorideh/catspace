import numpy as np
import pytest

from catspace.domains import krk


def test_krk_state_counts():
    W, B = krk.enumerate_states()
    assert len(W) == 7040
    assert len(B) == 10488


def test_krk_dtm_extremes():
    W, B = krk.enumerate_states()
    dtm_w, dtm_b = krk.compute_dtm(W, B)
    finite = np.isfinite(dtm_w)
    assert finite.sum() == 7040          # every KRk W state is forcibly won
    assert dtm_w[finite].max() == 19.0


def test_krk_legality_invariant():
    """The side not to move can never be in check."""
    W, B = krk.enumerate_states()
    for (wk, wr, bk) in W:
        assert not krk.black_in_check(wk, wr, bk)


def test_krk_build_chain_matches_enumeration():
    chain = krk.build_chain()
    W, _ = krk.enumerate_states()
    assert chain.n_live == len(W)
    assert chain.n == len(W) + 2


@pytest.mark.slow
def test_krkn_state_counts():
    from catspace.domains import krkn
    chain = krkn.build_chain(verbose=False)
    n2 = chain.strata["KRkn"].stop
    assert n2 == 158232
    assert chain.n_live == 158232 + 7040


@pytest.mark.slow
def test_krkn_dtm_max():
    from catspace.domains import krkn
    chain = krkn.build_chain(verbose=False)
    dtm = krkn.compute_dtm(chain)
    n2 = chain.strata["KRkn"].stop
    fin = np.isfinite(dtm[:n2])
    assert fin.mean() == pytest.approx(0.605, abs=0.01)
    assert np.nanmax(np.where(fin, dtm[:n2], np.nan)) == 43
