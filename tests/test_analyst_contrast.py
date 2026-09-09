"""contrast.py: a move explained against an alternative -- what it allows, prevents, gives up or creates."""
import chess

from catspace.incubator.analyst.memo import Memo
from catspace.incubator.analyst.contrast import contrast


def test_the_scholars_mate_by_contrast():
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3")
    c = contrast(b, b.parse_san("Nf6"), b.parse_san("g6"), Memo())
    assert c.a.mate_threat == "Qxf7#" and c.b.mate_threat is None
    assert any(l.startswith("Nf6 allows Qxf7#, mate; g6 does not") for l in c.lines)
    # the other way round the same fact reads as prevention
    c2 = contrast(b, b.parse_san("g6"), b.parse_san("Nf6"), Memo())
    assert c2.b.mate_threat == "Qxf7#" and c2.a.mate_threat is None and any("prevents" in l or "does not" in l for l in c2.lines)


def test_the_game_move_against_the_missed_knight():
    b = chess.Board("r4rk1/2q1bppp/n1p1pn2/2PpN3/8/P1N1P3/5PPP/R2Q1RK1 b - - 0 16")
    c = contrast(b, b.parse_san("Nxc5"), b.parse_san("Qxe5"), Memo())
    assert c.b.base > c.a.base + 2                                     # the knight is worth more than the pawn, one reply deep
    assert any("Qxe5 sure gain" in l and "Nxc5 sure gain" in l for l in c.lines)
    assert "Qxc3 (+4)" in " ".join(c.gives_up)                         # the queen's other prize, kept by Qxe5 and lost by Nxc5


def test_a_quiet_difference_says_so():
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    c = contrast(b, b.parse_san("Bc4"), b.parse_san("Bb5"), Memo())
    assert not c.allows and not c.prevents and not c.a.mate_threat


def test_a_created_threat_names_the_reply_that_meets_it():
    from catspace.incubator.analyst.engine import Engine
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3")
    with Engine(depth=12) as eng:
        c = contrast(b, b.parse_san("Qh5"), b.parse_san("Nc3"), Memo(), eng)
    assert "Qxf7# (mate)" in " ".join(c.creates)
    assert c.a.refuted.startswith("g6 meets it (blocks the line to f7)") and "attacks the queen on h5" in c.a.refuted
    assert any("but g6 meets it" in l for l in c.lines)
