"""The word `worth`: the declared value of the piece on a square, whoever owns it (plan 42.2)."""
import analyst_core as A
from catspace.incubator.analyst.discovery import base_vocabulary as BV

FEN_W = "4k3/8/8/3q4/8/2P5/8/3QK3 w - - 0 1"   # white: Qd1, Pc3, Ke1; black: Qd5, Ke8


def test_worth_reads_the_piece_either_side():
    BV.install_terms()
    got = {s: A.vocab_eval(FEN_W, f"(worth {s})") for s in ("d1", "c3", "e1", "d5", "e8", "a1")}
    print("OBS worth white to move", got)
    assert got == {"d1": 9, "c3": 1, "e1": 0, "d5": 9, "e8": 0, "a1": 0}


def test_worth_from_the_other_side_of_the_board():
    BV.install_terms()
    fen_b = "4k3/8/8/3q4/8/2P5/8/3QK3 b - - 0 1"        # the same board, black to move: names are from black's side (mirrored)
    got = {s: A.vocab_eval(fen_b, f"(worth {s})") for s in ("d1", "d8", "d4", "e8", "a8")}
    print("OBS worth black to move (mirrored names)", got)
    assert got == {"d1": 0, "d8": 9, "d4": 9, "e8": 0, "a8": 0}


def test_worth_renders_in_lean_and_words():
    s = BV.lean(("worth", "a1"))
    g = BV.gloss(("worth", "a1"))
    print("OBS", s, "|", g)
    assert "worthOf p" in s and "declared value" in g
