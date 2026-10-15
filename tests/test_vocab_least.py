"""The words `least` and `most` (Lean leastOver / mostOver): the worst and the best over every legal reply."""
import chess
import analyst_core as A
from catspace.incubator.analyst.discovery import base_vocabulary as BV

FEN = "4k3/8/4p3/3Q4/8/8/8/4K3 b - - 1 1"     # black to move; the e6 pawn can take the queen on d5


def _expected(fen, body_abs):
    """The body (absolute squares, read with white to move: the replies of a black move) on each reply position."""
    b = chess.Board(fen)
    vals = []
    for m in list(b.legal_moves):
        b.push(m)
        vals.append(A.vocab_eval(b.fen(), body_abs))
        b.pop()
    return vals


def test_least_and_most_read_the_legal_replies():
    BV.install_terms()
    vals = _expected(FEN, "(see d5)")
    lo, hi = A.vocab_eval(FEN, "(least (see d4))"), A.vocab_eval(FEN, "(most (see d4))")   # black names abs d5 as d4
    print(f"OBS replies {len(vals)} values {sorted(set(vals))} least {lo} most {hi} (spec spelling d5: {A.vocab_eval(FEN, '(least (see d5))')})")
    assert (lo, hi) == (min(vals), max(vals))
    assert lo < hi                                  # the capture on d5 and a quiet move differ


def test_after_with_least_evaluates():
    BV.install_terms()
    v = A.vocab_eval(FEN, "(after e6 d5 (least (see d5)))")
    print(f"OBS (after e6 d5 (least (see d5))) {v}")
    assert isinstance(v, int)


def test_no_legal_move_reads_zero():
    BV.install_terms()
    mate = "R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1"
    assert A.vocab_eval(mate, "(least (see a8))") == 0 and A.vocab_eval(mate, "(most (see a8))") == 0


def test_lean_renders_the_quantifier():
    s = BV.lean(("least", ("see", "a1")))
    s2 = BV.lean(("most", ("see", "a1")))
    print("OBS", s, s2, "|", BV.gloss(("least", ("see", "a1"))))
    assert "leastOver p" in s and "mostOver p" in s2
