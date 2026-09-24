"""The search over bodies, the watchers, the scheduler, the MCTS, the guide's features and the shapiq decomposition."""
from pathlib import Path

import chess
import pytest

from catspace.incubator.analyst.chess_domain import domain
from catspace.incubator.analyst.lemma_apply import Applier, load_library, duties


@pytest.fixture(scope="module")
def app():
    return Applier(domain(), load_library(Path("data/derived/lemmas/library_v2_size3.json"), Path("data/derived/lemmas/proved_v2_size3.json")))


def test_body_search_grows_and_settles(app):
    from catspace.incubator.analyst.body_search import body_search
    b = chess.Board("4k3/7K/8/8/7b/8/8/2r4R w - - 0 1")
    r = body_search(app, b, chess.H1, chess.Move.from_uci("h1g1"), max_bodies=4)
    assert r["trace"][0]["bodies"] == [chess.H1]
    assert all(len(l["bodies"]) == i + 1 for i, l in enumerate(r["trace"]))     # one body per level
    assert r["final"] == "loses" and r["settled"] >= 1
    r2 = body_search(app, b, chess.H1, chess.Move.from_uci("h1c1"), max_bodies=4)
    assert r2["final"] == "keeps"
    assert not any(l["flip"] for l in r2["trace"])                              # the verdict never flipped


def test_related_ranks_the_attacker_before_the_far_piece(app):
    from catspace.incubator.analyst.body_search import related
    b = chess.Board("4k3/8/8/8/7b/8/8/2r4R w - - 0 1")
    ranked = related(app, b, [chess.H1], [chess.C1, chess.H4, chess.E8])
    assert ranked[0][1] == chess.C1                                               # the rook that attacks h1 outranks the bishop and the far king


def test_duties_name_the_hanging_piece(app):
    b = chess.Board("4k3/7K/8/8/7b/8/8/2r4R w - - 0 1")
    d = duties(app, b)
    assert chess.H7 in d and d[chess.H7]["duties"][0]["holds"] == ["h1"] and d[chess.H7]["duties"][0]["stake"] == 5.0


def test_watchers_pose_the_loss_and_its_answers():
    from catspace.incubator.analyst.watchers import pose
    b = chess.Board("4k3/7K/8/8/7b/8/8/2r4R w - - 0 1")
    qs = pose(b)
    loss = [q for q in qs if q.watcher == "loss"]
    assert loss and "h1" in loss[0].text and abs(loss[0].stake - 5.0) < 0.5
    assert "h1c1" in loss[0].answers and "h1h2" in loss[0].answers and "h7g7" not in loss[0].answers
    win = [q for q in qs if q.watcher == "win"]
    assert win and "h1c1" in win[0].answers
    b2 = chess.Board("4k3/8/8/8/8/8/8/5r1K w - - 0 1")                  # in check (rook f1): every legal move answers
    qs2 = pose(b2)
    assert any(q.text == "get out of check" and q.answers == {m.uci() for m in b2.legal_moves} for q in qs2)


def test_scheduler_store_footprint_and_budget():
    from catspace.incubator.analyst.scheduler import Scheduler
    s = Scheduler()
    calls = []
    assert s.want("k", "a", {1, 2}, 1.0, lambda: calls.append("a") or "A") is False
    assert s.want("k", "b", {5}, 3.0, lambda: calls.append("b") or "B") is False
    assert s.want("k", "a", {1, 2}, 9.0, lambda: None) is False and s.stats["dup"] == 1   # pending: importance raised, not re-queued
    s.on_position(1, changed={2})                                                    # drops 'a' (footprint crosses), keeps 'b'
    assert "a" not in s.pending and "b" in s.pending
    assert s.run(budget=10) == 1 and s.store["b"] == "B" and calls == ["b"]
    assert s.want("k", "b", {5}, 1.0, lambda: None) is True                          # now a hit
    s2 = Scheduler(); s2.want("k", "x", set(), 1.0, lambda: "X"); s2.want("k", "y", set(), 2.0, lambda: "Y")
    assert s2.run(budget=1) == 1 and "y" in s2.store and "x" not in s2.store          # importance order, budget respected


def test_changed_squares_include_attack_shifts():
    from catspace.incubator.analyst.scheduler import changed_squares
    b = chess.Board("4k3/7K/8/8/7b/8/8/2r4R w - - 0 1")
    ch = changed_squares(b, chess.Move.from_uci("h1h2"))
    assert chess.H1 in ch and chess.H2 in ch and chess.G1 in ch                       # g1 was attacked by the rook and no longer is


def test_mcts_finds_the_capture_and_visits_sum():
    from catspace.incubator.analyst.mcts import MCTS
    from catspace.incubator.analyst.player import Player
    pl = Player(None, None, use_lemmas=False)
    m = MCTS(pl, sims=120)
    b = chess.Board("4k3/7K/8/8/7b/8/8/2r4R w - - 0 1")
    r = m.run(b)
    assert r["move"] == chess.Move.from_uci("h1c1")
    assert sum(n for _, n in r["visits"]) <= 120


def test_guide_features_are_generic_and_fixed_width():
    from catspace.incubator.analyst.body_guide import features, relation_weights
    dom = domain()
    rels_bin = [r for r in dom.relations if not r.on_action and r.arity == 2]
    rels_un = [r for r in dom.relations if not r.on_action and r.arity == 1]
    b = chess.Board("4k3/7K/8/8/7b/8/8/2r4R w - - 0 1")
    w = relation_weights(dom, b, rels_bin)
    f1 = features(dom, b, chess.Move.from_uci("h1g1"), chess.C1, rels_bin, rels_un, dom.types, w)
    f2 = features(dom, b, chess.Move.from_uci("h1g1"), chess.H4, rels_bin, rels_un, dom.types, w)
    assert f1.shape == f2.shape and f1.shape[0] == 2 * len(dom.types) + 4 * len(rels_bin) + len(rels_un) + 2
    assert f1[-2] > f2[-2]                                                            # the heuristic score: the attacking rook relates more


def test_shap_decomposition_has_the_mover_alone_as_the_empty_set():
    from catspace.incubator.analyst.shap_bodies import decompose, outcome_on_subset
    import random
    dom = domain()
    b = chess.Board("4k3/7K/8/8/7b/8/8/2r4R w - - 0 1")
    mv = chess.Move.from_uci("h1g1")
    r = decompose(dom, b, mv, max_players=4, relative=False, rng=random.Random(0))
    mob = r["moebius"]
    empty = [v for s, v in zip(mob.interaction_lookup.keys(), mob.values) if len(s) == 0][0]
    assert empty == pytest.approx(outcome_on_subset(dom, b, mv, set(), False))          # the mover alone
    assert r["full"] == pytest.approx(-5.0)                                            # with everything: Rg1 loses the rook


def test_imagined_gains_never_come_from_a_threat_the_removal_created():
    """An imagined blocker removal must not lower the actor's own null-move baseline: otherwise the 'gain' is only the
    escape from a threat the imagination itself created (2026-09-24, the d4/d5 report)."""
    from pathlib import Path
    from catspace.incubator.analyst.chains import imagined_ideas, _baseline_dropped, load_library, OUT
    from catspace.incubator.analyst.lemma_apply import Applier
    from catspace.incubator.analyst.chess_domain import domain
    lib, prov = OUT / "library_games_gain.json", OUT / "proved_games_gain.json"
    if not lib.exists():
        import pytest; pytest.skip("no gain library")
    app = Applier(domain(oracle="frame"), load_library(lib, prov))
    b = chess.Board("r1bq1rk1/ppp2ppp/2n2n2/3pp3/1b1P4/2NBPN2/PPP2PPP/R2QK2R w KQ - 0 8")
    for i in imagined_ideas(app, b, max_out=20):
        st = app.dom.state_of(b, i["ids"]); b2 = b.copy(stack=False)
        for x in i["blockers"]:
            b2.remove_piece_at(x)
        assert not _baseline_dropped(app.dom, st, app.dom.state_of(b2, i["ids"]), i["ids"][i["lemma"].slot]), i["move"]
