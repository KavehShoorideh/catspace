import numpy as np
import pytest
import scipy.stats as st

from latentchess.domains import krk
from latentchess.chain import exact_P
from latentchess.cone.tabular import sm_matvec, randomized_svd_sm, fb_from_svd, rank_error, TabularFB
from latentchess.cone.embedding import GoalSpec, reach


def test_sm_matvec_matches_dense_on_small_chain():
    rng = np.random.default_rng(0)
    n = 20
    A = rng.random((n, n))
    P = A / A.sum(1, keepdims=True)
    gamma = 0.9
    X = rng.standard_normal((n, 3))

    import scipy.sparse as sp
    Psp = sp.csr_matrix(P)
    got = sm_matvec(Psp, X, gamma, T=500)

    # dense closed form: (1-g) * (I - gP)^-1 @ X
    I = np.eye(n)
    expected = (1 - gamma) * np.linalg.solve(I - gamma * P, X)
    assert np.allclose(got, expected, atol=1e-4)


def test_rank_error_monotone_on_exact_krk():
    chain = krk.build_chain()
    P = exact_P(chain)
    gamma = 0.92
    errs = []
    for d in (8, 32, 128):
        U, S, V = randomized_svd_sm(P, gamma, d, seed=0)
        F, Bm = fb_from_svd(U, S, V)
        errs.append(rank_error(P, gamma, F, Bm, n_probe=8, seed=1))
    assert errs[0] >= errs[1] >= errs[2] - 1e-9


def test_reach_matches_exact_reach_at_rank_64():
    """Sanity-checks the TabularFB/GoalSpec/reach wiring against the exact
    reach vector. NOTE: rank-64 reach-spearman is documented at ~0.44 in
    RESULTS-v3 (rank-limited FB captures local move-ranking well but not
    global reach values -- the mis-registered G-M1 gate (a)); this is not a
    numerical bug, so the threshold reflects that finding rather than ~1.0."""
    chain = krk.build_chain()
    P = exact_P(chain)
    gamma = 0.92
    W, B = krk.enumerate_states()
    dtm_w, _ = krk.compute_dtm(W, B)
    region = np.array(list(np.where(dtm_w <= 3)[0]) + [chain.terminals.mate])

    e = np.zeros((chain.n, 1)); e[region] = 1.0
    reach_true = sm_matvec(P, e, gamma).ravel()

    emb = TabularFB.fit(P, gamma, d=64, seed=0)
    goal = GoalSpec(name="near_mate", region=region, z=emb.B[region].sum(0))
    r_hat = reach(emb, goal)

    rho = st.spearmanr(r_hat[: chain.n_live], reach_true[: chain.n_live]).statistic
    assert rho == pytest.approx(0.4351, abs=0.01)
