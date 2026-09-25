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
            assert c.value <= c.bound + 1e-9, (b.san(c.move), c.value, c.bound)      # the frame's material can never exceed the enemy worth inside it
    assert on["best"].value == off["best"].value
    assert on["opened"] <= off["opened"]
    assert any(c.move == chess.Move.from_uci("d6h2") and c.value is not None and c.value > 0 for c in on["choices"])
    d = G.decide(app, b, k=3)
    assert d[0][1].move == chess.Move.from_uci("d6h2")                          # ours minus their best reply: the sacrifice wins the decision


def test_neighbourhood_holds_the_pawn_taken_en_passant():
    b = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert chess.D5 in G.neighbourhood(b, chess.Move.from_uci("e5d6"))
