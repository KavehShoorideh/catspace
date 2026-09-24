"""The square ledger: active and near squares from guard terms, explained; the actor's own safety is not a tactic."""
from pathlib import Path

import chess
import pytest

from catspace.incubator.analyst.chains import domain_of
from catspace.incubator.analyst.lemma_apply import Applier, load_library
from catspace.incubator.analyst.squares import control, ledger, term_squares


def test_term_squares_name_other_pieces_and_the_landing_square():
    ids = [chess.D6, chess.H2, chess.F3]
    b = chess.Board("2kr3r/ppp2ppp/2nb1n2/7q/3P2b1/2P2N2/PP1NBPPP/R1BQ1RK1 b - - 0 1")
    a = (b, chess.Move.from_uci("d6h2"))
    term = (("attacked", (1,), False), ("lands_attacked", (), True))
    assert term_squares(term, ids, a, chess.D6) == {chess.H2}                 # slot 1 and the landing square; the actor itself is not a place


def test_ledger_marks_the_deflection_square_and_explains_it():
    lib, prov = Path("data/derived/lemmas/library_games_frame6k.json"), Path("data/derived/lemmas/proved_games_frame6k.json")
    if not lib.exists():
        pytest.skip("no library")
    app = Applier(domain_of(lib), load_library(lib, prov))
    b = chess.Board("2kr3r/ppp2ppp/2nb1n2/7q/3P2b1/2P2N2/PP1NBPPP/R1BQ1RK1 b - - 0 1")
    L = ledger(app, b)
    assert L["embeddings"] > 100 and L["active"] and L["near"]
    top_near = sorted(L["near"], key=lambda s: -L["near"][s])[:10]
    assert chess.F3 in top_near or chess.H2 in top_near or chess.E2 in top_near      # the squares the combination runs through
    for sq, rows in L["why"].items():
        for lid, stake, term, kind, uci in rows:
            assert kind in ("active", "near") and stake >= 1.0
    c = {m: control(b, m, L["near"]) for m in b.legal_moves}
    assert len(c) == len(list(b.legal_moves)) and any(v != 0 for v in c.values())
