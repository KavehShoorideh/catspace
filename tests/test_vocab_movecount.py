"""The words `movecount` (legal moves of the side to move) and `check` (1 when it is in check), plan 47."""
import chess
import analyst_core as A
from catspace.incubator.analyst.discovery import base_vocabulary as BV

FENS = ["4k3/8/8/3q4/8/2P5/8/3QK3 w - - 0 1",
        "r3k2r/pp3ppp/2n1b3/3q4/3P4/2N1BN2/PP3PPP/R2QK2R b KQkq - 0 1",
        "8/2R5/6pk/p3p2p/4Nq2/5n1P/P4RP1/7K b - - 1 41",
        "8/5pk1/6p1/8/2B5/1R6/5PK1/8 w - - 0 1"]
MATED = "8/2R5/6pk/p3p2p/4N3/5n1P/P4RPq/7K w - - 0 42"      # 00cZ4 after Qh2#


def test_movecount_and_check_agree_with_python_on_four_positions():
    BV.install_terms()
    for fen in FENS:
        b = chess.Board(fen)
        got = (A.vocab_eval(fen, "movecount"), A.vocab_eval(fen, "check"))
        want = (b.legal_moves.count(), int(b.is_check()))
        print("OBS movecount/check", fen.split()[0], "engine", got, "python", want)
        assert got == want


def test_a_mated_position_reads_zero_moves_in_check():
    BV.install_terms()
    b = chess.Board(MATED)
    assert b.is_checkmate()
    got = (A.vocab_eval(MATED, "movecount"), A.vocab_eval(MATED, "check"))
    print("OBS mated position movecount/check", got)
    assert got == (0, 1)


def test_under_after_the_words_read_their_side():
    BV.install_terms()
    fen = FENS[2]      # black to move: squares are named from the mover side (ranks upside down), so Qf4-h2 is f5 h7
    got = (A.vocab_eval(fen, "(after f5 h7 movecount)"), A.vocab_eval(fen, "(after f5 h7 check)"))
    print("OBS after Qh2 movecount/check", got)
    assert got == (0, 1)


def test_rendering_and_gloss():
    for w, lean in (("movecount", "moveCountOf p"), ("check", "checkOf p")):
        s, g = BV.lean(w), BV.gloss(w)
        print("OBS", s, "|", g)
        assert lean in s and g
