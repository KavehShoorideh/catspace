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


def test_after_plays_the_kernels_move_when_legal():
    import chess
    BV.install_terms()
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    # castling: the king move e1->g1 plays the rook too (the kernel's move)
    g = A.vocab_eval(fen, "(after e1 g1 (count (at f1)))")
    b3 = chess.Board(fen); b3.push_san("O-O")
    # a king step: the right is gone, so Ke2 then Ke1 back leaves black's rook a8 unable to be ... count black king-side castle is not ours;
    # compare the white rook h1's moves after Ke2 (black to move): same as python-chess' count of h1 moves when it is white's turn again
    got = A.vocab_eval(fen, "(after e1 e2 (count (moves h1)))")
    b2 = chess.Board(fen); b2.push_san("Ke2"); b2.turn = chess.WHITE
    exp = sum(1 for m in b2.legal_moves if m.from_square == chess.H1)
    print(f"OBS castle f1 {g}; after Ke2 rook h1 moves {got}, python-chess {exp}")
    assert g == 1 and b3.piece_at(chess.F1).symbol() == "R" and got == exp
    # en passant: the kernel sets the e.p. square after e2e4, so the pawn on d4 has 2 moves (push and capture)
    fen2 = "4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1"
    v = A.vocab_eval(fen2, "(after e2 e4 (count (moves d4)))")
    b4 = chess.Board(fen2); b4.push_san("e4")
    exp4 = sum(1 for m in b4.legal_moves if m.from_square == chess.D4)
    print(f"OBS after e2e4 (count (moves d4)) {v}, python-chess {exp4}")
    assert v == exp4 == 2
