"""The learned threat (Kaveh 2026-10-14): the mechanics only -- the teacher's label, the frame's ranking, the ledger's hook. Not the learned words."""
import shutil

import chess
import chess.engine
import pytest

from catspace.incubator.analyst.discovery import ledger as L
from catspace.incubator.analyst.discovery import watch as W

needs_sf = pytest.mark.skipif(shutil.which("stockfish") is None, reason="no stockfish")


def test_the_pass_board_and_the_no_pass_in_check():
    assert W.pass_board("4k3/8/8/3n4/8/8/3R4/4K3 b - - 0 1").turn == chess.WHITE
    assert W.pass_board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1") is None          # the mover is in check: no pass


@needs_sf
def test_the_label_of_a_hanging_piece_is_its_value_and_a_repair_is_not_a_threat():
    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    try:
        r = W.threat_labels("4k3/8/8/3n4/8/8/3R4/4K3 b - - 0 1", eng)            # Black to move; the rook takes the loose knight if Black passes
        sizes = {m["uci"]: m["size"] for m in r["moves"]}
        assert sizes["d2d5"] == 3 and max(v for u, v in sizes.items() if u != "d2d5") <= 0
        r = W.threat_labels("4k3/3r4/8/8/8/3B4/8/6K1 b - - 0 1", eng)            # Black's rook attacks the loose bishop: O saving it is no threat
        sizes = {m["uci"]: m["size"] for m in r["moves"]}
        assert sizes["d3f1"] <= 0 and sizes["d3e2"] <= 0
    finally:
        eng.quit()


def test_the_frame_ranks_against_the_null_and_the_ledger_hook_switches():
    frame = {"terms": [{"term": "(see O)", "family": "raw", "sign": -1, "gloss": "what the other side wins by the exchange on their other piece"},
                       {"term": "(count edge)", "family": "raw", "sign": -1, "gloss": "the squares on the edge"}]}
    pb = W.pass_board("4k3/8/8/3n4/8/8/3R4/4K3 b - - 0 1")
    rec = dict(fen="4k3/8/8/3n4/8/8/3R4/4K3 b - - 0 1", pass_fen=pb.fen(), moves=[dict(uci=m.uci(), size=0) for m in pb.legal_moves])
    above, null = W.frame_ranking(frame, rec)
    assert all(t > null for _, t in above) and [t for _, t in above] == sorted([t for _, t in above], reverse=True)
    fen = "5k1r/1p1n1p2/p1qb1p1p/4p3/2rP2PP/1PPQ1PB1/P3NK2/1R5R b - - 0 1"
    rules = L.new(fen, 200).trace(2, "rules")
    assert rules["learned"] is False
    defined = L.new(fen, 200).trace(2, "defined")           # the threat DEFINED (material lost or mate) and sized by the ledger
    assert defined["learned"] is True and defined["looks"] > 0 and defined["positions"] >= 1
    assert "next move b3" in "\n".join(defined["trace"]) or "Rc4" in "\n".join(defined["trace"])


def test_the_sizer_finds_the_threat_and_gives_every_threat_its_reason():
    import analyst_core as A
    L.load_frame  # (the frame no longer orders the threats: the threat is defined)
    rep = A.Ledger("5k1r/1p1n1p2/p1qb1p1p/4p3/2rP2PP/2PQ1PB1/PP2NK2/1R5R b - - 0 1", 1000).threat_report(True)
    assert rep["threats"] and rep["threats"][0]["san"] == "b3" and rep["threats"][0]["size"] >= 4 and "TRAPPED" in rep["threats"][0]["detail"]
    rep = A.Ledger("4k3/8/8/3n4/8/8/3R4/4K3 b - - 0 1", 1000).threat_report(True)            # a plain capture has its reason
    assert rep["threats"][0]["san"] == "Rxd5" and rep["threats"][0]["size"] == 3 and "takes" in rep["threats"][0]["detail"] and "guarded by" in rep["threats"][0]["detail"]
    # a thread carried through a game: the overloaded guard of the knight on e5 (the live finding) names the capture that cashes it in
    a = "1r1qk2r/p3bpp1/2p1p2p/2p3P1/4PPn1/2NPpN2/PPP1Q1P1/1R3RK1 w k - 0 15"
    led = A.Ledger(a, 1000); led.run(); led.push("f3e5")
    t = led.threat_report(True, True)["threats"]
    assert t and t[0]["san"] == "Nxe5" and t[0]["thread"] and "no longer sees Ne5" in t[0]["thread"]


def test_a_checking_move_is_read_on_the_true_board_not_as_zeros():
    fen = "4k3/8/8/8/8/8/R7/4K3 w - - 0 1"
    after = W._true_after(fen, "a2a8")                                  # Ra8+: the other side is in check, to move -- a legal board
    terms = W.threat_terms()[0]
    j = terms.index("material_us")
    assert W._tmatrix([after], True)[0][j] == 5                         # read with us = White, the checked side to move
    assert not W._tmatrix([W._flip_after(fen, "a2a8")], True)[0].any()  # the old reading (the turn handed back) is a board the evaluator refuses: all zeros, a free pass for checks


def test_one_threat_per_thing_that_would_be_lost():
    import analyst_core as A
    fen = "1r1qk2r/p3bpp1/2p1p2p/2p1N1P1/4PPn1/2NPp3/PPP1Q1P1/1R3RK1 w k - 1 15"
    th = A.Ledger(fen, 1000).threat_report(True)["threats"]
    targets = [t["target"] for t in th]
    assert len(targets) == len(set(targets)) and all(t["size"] >= 1 for t in th)        # no target listed twice
    g5 = [t for t in th if t["target"] == "g5"]
    assert not g5 or g5[0]["san"] == "hxg5"                              # shown with the move that cashes it in, not under a quiet move that leaves it standing
