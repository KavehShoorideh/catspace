"""Plan 78.40 probe: the fact-space forward operator (observables, pairs, ridge, roll, R2, AUC, events)."""
import json

import numpy as np

from catspace.incubator.analyst.discovery import fact_forward as FF
from catspace.incubator.analyst.discovery import game_store as GS

GAME = "chess"
FENS = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "7k/5K2/8/8/8/8/8/R7 w - - 0 1"]


def _facts(fen):
    spec = GS.spec_texts(GAME, "relational")[GAME]
    return FF.read_facts(GAME, spec, json.dumps(GS.vocabulary(GAME, "relational")), fen, [])


def test_observables_dimension_stable_and_named():
    corpus = [_facts(f) for f in FENS]
    sch = FF.build_schema(corpus, max_dim=40)
    xs = [sch.encode(f) for f in corpus]
    assert all(x.shape == (sch.dim,) for x in xs) and sch.dim <= 40
    assert len(sch.names) == sch.dim and all(sch.names) and sch.names[-1] == "const" and xs[0][-1] == 1
    assert xs[0][sch.index[("cell", None, None)]] == 32 and xs[1][sch.index[("cell", None, None)]] == 3
    assert xs[0][sch.material] == 0 and xs[1][sch.material] == 1      # second position: the mover has a rook more
    assert FF.material_of(corpus[1]) == 1


def test_pairs_and_split_by_game():
    seqs = {g: [np.array([g, t]) for t in range(5)] for g in range(6)}
    X, Y, G = FF.build_pairs(seqs, 2)
    assert len(X) == 6 * 3 and (Y[:, 1] - X[:, 1] == 2).all() and (X[:, 0] == G).all()
    tr, he = FF.split_games(list(range(6)), 0.34, 3)
    assert set(tr) | set(he) == set(range(6)) and not set(tr) & set(he) and he
    assert set(G[np.isin(G, tr)]).isdisjoint(set(G[np.isin(G, he)]))


def test_ridge_recovers_a_known_map():
    rng = np.random.RandomState(0)
    X = np.hstack([rng.randn(200, 3), np.ones((200, 1))])
    W = rng.randn(4, 4)
    Y = X @ W
    assert np.allclose(FF.ridge_fit(X, Y, 1e-8), W, atol=1e-5)
    w2, lam = FF.ridge_pick(X, Y, np.repeat(np.arange(20), 10))
    assert np.allclose(X @ w2, Y, atol=1e-2)


def test_rolled_equals_direct_on_a_linear_system():
    rng = np.random.RandomState(1)
    M = np.eye(3) * 0.9 + 0.05 * rng.randn(3, 3)
    xs = [rng.randn(3)]
    for _ in range(40):
        xs.append(M @ xs[-1])
    seq = {0: [np.append(x, 1.0) for x in xs]}
    X2, Y2, _ = FF.build_pairs(seq, 2)
    X4, Y4, _ = FF.build_pairs(seq, 4)
    A2, A4 = FF.ridge_fit(X2, Y2, 1e-10), FF.ridge_fit(X4, Y4, 1e-10)
    assert np.allclose(X4 @ A2 @ A2, X4 @ A4, atol=1e-5)
    assert np.allclose(X4 @ A2 @ A2, Y4, atol=1e-5)


def test_copy_r2_by_hand():
    Y = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    X = np.array([[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    r = FF.r2_columns(Y, X)
    assert abs(r[0] - (1 - 2.0 / 8.0)) < 1e-12 and np.isnan(r[1])


def test_auc_and_spearman():
    assert FF.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert FF.auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert FF.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert np.isnan(FF.auc([1, 2], [1, 1]))
    assert abs(FF.spearman([1, 2, 3, 4], [10, 20, 25, 40]) - 1) < 1e-12
    assert abs(FF.spearman([1, 2, 3], [3, 2, 1]) + 1) < 1e-12 and np.isnan(FF.spearman([1, 1, 1], [1, 2, 3]))


def test_event_labels_on_a_made_up_sequence():
    n = 12
    material = [0] * n
    material[8] = -1                                      # lower 8 plies on from t=0
    checks = [0] * n
    checks[5] = 1
    hu = [0] * n
    ho = [0] * n
    ho[3] = 2                                             # odd offset: the mover-at-0's count is read from `them`
    assert FF.event_labels(material, checks, hu, ho, 0) == (True, True, True)
    assert FF.event_labels(material, [0] * n, hu, [0] * n, 0) == (True, False, False)
    assert FF.event_labels(material, checks, hu, ho, 5) is None     # future too short
    hu2 = [0] * n
    hu2[2] = 1                                            # even offset: read from `us`
    assert FF.event_labels(material, checks, hu2, [0] * n, 0)[2] is True


def test_moved_kind_from_facts_alone():
    before = [("cell", "e", "2", "us", "pawn"), ("cell", "a", "1", "us", "rook")]
    after = [("cell", "e", "4", "them", "pawn"), ("cell", "a", "1", "them", "rook")]
    assert list(FF.moved_kind(before, after, ["pawn", "rook"])) == [1.0, 0.0]
