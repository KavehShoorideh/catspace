import numpy as np
import pytest

from catspace.domains import krk
from catspace.chain import exact_P, empirical_P


@pytest.fixture(scope="module")
def chain():
    return krk.build_chain()


def test_csr_pointers_monotone(chain):
    assert np.all(np.diff(chain.move_ptr) >= 0)
    assert np.all(np.diff(chain.out_ptr) >= 0)


def test_every_live_state_has_a_move(chain):
    assert np.all(chain.move_counts > 0)


def test_out_flat_in_range(chain):
    assert chain.out_flat.min() >= 0
    assert chain.out_flat.max() < chain.n


def test_exact_P_row_stochastic(chain):
    P = exact_P(chain)
    rowsum = np.asarray(P.sum(axis=1)).ravel()
    assert np.allclose(rowsum, 1.0)
    # absorbing rows are identity
    for a in chain.terminals.indices:
        assert P[a, a] == pytest.approx(1.0)


def test_empirical_P_row_stochastic_and_visited_mask(chain):
    rng = np.random.default_rng(0)
    rows, cols = [], []
    for s in rng.integers(0, chain.n_live, size=500):
        mid = int(chain.move_ptr[s])
        outs = chain.outs_of(mid)
        rows.append(int(s)); cols.append(int(outs[0]))
    P, visited = empirical_P(rows, cols, chain.n, chain.terminals)
    rowsum = np.asarray(P.sum(axis=1)).ravel()
    assert np.allclose(rowsum, 1.0)
    assert visited[rows].all()
    for a in chain.terminals.indices:
        assert P[a, a] == pytest.approx(1.0)
