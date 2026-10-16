"""The word `maxworth`: the largest declared value among the pieces on a set of squares (plan 44)."""
import chess
import analyst_core as A
from catspace.incubator.analyst.discovery import base_vocabulary as BV

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
FENS = ["4k3/8/8/3q4/8/2P5/8/3QK3 w - - 0 1",
        "r3k2r/pp3ppp/2n1b3/3q4/3P4/2N1BN2/PP3PPP/R2QK2R w KQkq - 0 1",
        "8/5pk1/6p1/8/2B5/1R6/5PK1/8 w - - 0 1"]


def py_maxworth(fen, squares):
    b = chess.Board(fen)
    return max([VAL[b.piece_type_at(chess.parse_square(s))] for s in squares if b.piece_type_at(chess.parse_square(s))] or [0])


def test_maxworth_agrees_with_python_on_three_positions():
    BV.install_terms()
    # a set spelled by explicit squares: the union of single squares through `or` of `at`
    for fen in FENS:
        for squares in (["a1", "b1", "c1", "d1", "e1"], ["c3", "d5", "e8", "c4"], ["h1", "h2"], ["f7", "g7", "g6", "b3", "f2"]):
            term = "(maxworth " + _union(squares) + ")"
            got = A.vocab_eval(fen, term)
            want = py_maxworth(fen, squares)
            print("OBS maxworth", fen.split()[0], squares, "engine", got, "python", want)
            assert got == want


def _union(squares):
    t = f"(at {squares[0]})"
    for s in squares[1:]:
        t = f"(or {t} (at {s}))"
    return t


def test_maxworth_empty_and_kings_only_is_zero():
    BV.install_terms()
    fen = "4k3/8/8/3q4/8/2P5/8/3QK3 w - - 0 1"
    got = [A.vocab_eval(fen, f"(maxworth {_union(s)})") for s in (["a1"], ["e1", "e8"], ["e1", "e8", "a1"])]
    print("OBS maxworth empty / kings only", got)
    assert got == [0, 0, 0]


def test_maxworth_of_a_computed_set_and_the_rendering():
    BV.install_terms()
    fen = "4k3/8/8/3q4/8/2P5/8/3QK3 w - - 0 1"
    assert A.vocab_eval(fen, "(maxworth ours)") == 9      # ours: Qd1, Pc3 (king excluded)
    assert A.vocab_eval(fen, "(maxworth (and ours (not (at d1))))") == 1
    s = BV.lean(("maxworth", "ours"))
    g = BV.gloss(("maxworth", "ours"))
    print("OBS", s, "|", g)
    assert "maxWorthOf p" in s and "largest declared value" in g
