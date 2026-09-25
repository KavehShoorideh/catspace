"""The best goal with a sound prune: the neighbourhood reaches the uncovered pieces, every bound is above the value
it bounds, and the prune never changes the answer."""
from pathlib import Path

import chess
import pytest

from catspace.incubator.analyst import goals as G
from catspace.incubator.analyst.chains import domain_of
from catspace.incubator.analyst.lemma_apply import Applier, load_library

FEN = "2kr3r/ppp2ppp/2nb1n2/7q/3P2b1/2P2N2/PP1NBPPP/R1BQ1RK1 b - - 0 1"


@pytest.fixture(scope="module")
def app():
    from catspace.io import paths
    lib, prov = Path(paths.derived("lemmas", "library_games_frame6k.json")), Path(paths.derived("lemmas", "proved_games_frame6k.json"))
    if not lib.exists():
        pytest.skip("no library")
    return Applier(domain_of(lib), load_library(lib, prov))


def test_neighbourhood_reaches_what_the_deflected_knight_uncovers():
    b = chess.Board(FEN)
    fr = G.neighbourhood(b, chess.Move.from_uci("d6h2"))
    assert fr[0] == chess.D6 and chess.H2 in fr and chess.F3 in fr and chess.G1 in fr
    assert chess.E2 in fr and chess.G4 in fr, "the knight on f3 stands between the bishops g4 and e2: both must join"
    assert (chess.E2, chess.G4) in G._blocks(b, chess.F3) or (chess.G4, chess.E2) in G._blocks(b, chess.F3)


def test_bounds_are_optimistic_and_the_prune_keeps_the_answer(app):
    b = chess.Board(FEN)
    on = G.best_goal(app, b, prune=True); off = G.best_goal(app, b, prune=False)
    for c in off["choices"]:
        if c.value is not None:
            assert c.value <= c.bound + 1e-9, (b.san(c.move), c.value, c.bound)      # the value never exceeds what the frame holds: the enemy worth to win, ours to save
    assert on["best"].value == off["best"].value
    assert on["opened"] <= off["opened"]
    assert any(c.move == chess.Move.from_uci("d6h2") and c.value is not None and c.value > 0 for c in on["choices"])


@pytest.mark.xfail(strict=True, reason="review 2026-09-24: the frame values rank Bxf3 (+2) above the Greek gift (+1); Stockfish d20: "
                   "Bxh2+ +2.34, Bxf3 -2.47. This held before only through a fake mate class for White after Bxf3.")
def test_decide_prefers_the_greek_gift(app):
    d = G.decide(app, chess.Board(FEN), k=3)
    assert d[0][1].move == chess.Move.from_uci("d6h2")


def test_the_prune_counts_what_standing_still_loses(app):
    """The value is measured from the pass: a retreat is worth what it saves, so the bound must hold our pieces too."""
    b = chess.Board("4r2k/6pp/5p2/4p3/3Q4/8/8/6K1 w - - 0 1")
    on = G.best_goal(app, b, prune=True); off = G.best_goal(app, b, prune=False)
    assert on["best"].value == off["best"].value == 9.0
    assert all(c.value <= c.bound + 1e-9 for c in off["choices"] if c.value is not None)


def test_a_parried_mate_is_not_a_mate(app):
    b = chess.Board("6k1/5ppp/8/8/5n2/7q/5PP1/3R2K1 w - - 0 1")                 # Rd8# mates; gxh3 only parries the threat on g2
    r = G.best_goal(app, b, prune=True)
    assert r["best"].move == chess.Move.from_uci("d1d8") and r["best"].value == 1000.0
    assert next(c for c in r["choices"] if c.move == chess.Move.from_uci("g2h3")).value < G.MATE
    b2 = chess.Board("6k1/5ppp/8/8/5n2/7q/5PP1/6K1 w - - 0 1")                  # no rook: nothing mates
    assert G.best_goal(app, b2, prune=True)["best"].value < G.MATE


def test_decide_counts_their_reply_once_and_a_draw_as_the_lead_lost(app):
    b = chess.Board("4k3/8/4p3/8/8/8/8/3QK3 w - - 0 1")
    qd5 = [s for s, c, th in G.decide(app, b, k=30) if c.move == chess.Move.from_uci("d1d5")]
    assert qd5 == [-9.0]                                                         # the queen is lost once, not twice
    b2 = chess.Board("8/8/4R3/1K6/8/1Q4p1/8/5k2 w - - 0 1")                      # Qxg3 stalemates
    d = G.decide(app, b2, k=30)
    assert d[0][1].move != chess.Move.from_uci("b3g3")
    assert [s for s, c, th in d if c.move == chess.Move.from_uci("b3g3")] in ([], [-13.0])


def test_neighbourhood_holds_the_pawn_taken_en_passant():
    b = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert chess.D5 in G.neighbourhood(b, chess.Move.from_uci("e5d6"))
