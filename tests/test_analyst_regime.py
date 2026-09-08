"""regime.py: the board as a sum of sub-games when it earns it -- light cones, races, frozen regions, the tablebase
on the live remainder, and no split where there is none."""
import chess

from catspace.incubator.analyst.memo import Memo
from catspace.incubator.analyst.regime import decompose, race, cone, bound_cone, frozen, pawn_path


def test_the_rook_ending_splits_into_a_locked_queenside_and_a_won_kingside():
    # Kaveh's screenshot after ...Kxh5: the a4 rook is out of play, the queenside is locked, the white king cannot
    # leave the f-pawn's square, the c4 rook serves both sides, and the remainder is K+R+P v K: a tablebase loss for White
    b = chess.Board("8/8/1p3p2/p6k/RPr5/P5K1/8/8 w - - 0 2")
    dec = decompose(b, Memo())
    assert dec.split and len([g for g in dec.subgames if g.lemmas]) == 2
    q = next(g for g in dec.subgames if g.frozen)
    assert {chess.A3, chess.A4, chess.A5, chess.B4, chess.B6} <= q.pieces and any(getattr(l, "kind", "") == "trapped" and l.target == chess.A4 for l in q.lemmas)
    assert q.reservoir["w"] == 0 and "Rxa5+" in q.losing["w"] and "b5" in q.losing["w"] and "b5" in q.losing["b"]
    k = next(g for g in dec.subgames if not g.frozen)
    assert {chess.F6, chess.G3, chess.H5, chess.C4} <= k.pieces and k.shared == {chess.C4} and k.men == 4 and k.solved == "loss"
    assert dec.verdict.startswith("loss for White")
    assert any("out of play" in l for l in dec.lines) and any("Leave it as it is" in l for l in dec.lines)
    assert any("with the black rook on c4" in l and "tablebase: loss for White" in l for l in dec.lines)
    assert not any(w in l for l in dec.lines for w in ("queenside", "kingside"))    # named by its pieces, never by a wing
    # the white king's cone stays inside the f-pawn's square: no square west of the d-file within 3 moves
    assert all(chess.square_file(s) >= 3 for s in dec.cones[chess.G3])
    # the a4 rook has no cone at all
    assert dec.cones[chess.A4] == {chess.A4: 0}


def test_no_split_in_a_middlegame_and_exact_verdicts_on_small_boards():
    b = chess.Board("r4rk1/2q1bppp/npp1pn2/2PpN3/3P4/P1N1P3/5PPP/R2Q1RK1 b - - 5 15")
    dec = decompose(b, Memo())
    assert not dec.split and dec.lines == []                                 # one game: nothing to say here
    b = chess.Board("8/5k2/8/8/8/8/PP3K2/8 w - - 0 1")
    dec = decompose(b, Memo())
    assert not dec.split and dec.verdict == "win for White (tablebase)"


def test_races_and_cones():
    b = chess.Board("8/5k2/8/8/8/8/PP3K2/8 w - - 0 1")
    assert pawn_path(b, chess.B2) == (chess.B3, chess.B4, chess.B5, chess.B6, chess.B7, chess.B8)
    r = race(b, chess.B2)
    assert r.steps == 5 and r.catcher == chess.F7 and not r.promotes          # five with the double step; the king is inside the square
    # a pawn that is not passed is no runner
    b = chess.Board("r4rk1/2q1bppp/npp1pn2/2PpN3/3P4/P1N1P3/5PPP/R2Q1RK1 b - - 5 15")
    assert pawn_path(b, chess.B6) is None
    # a runner the king cannot catch
    b = chess.Board("8/8/8/8/8/k7/6P1/K7 w - - 0 1")
    r = race(b, chess.G2)
    assert r.promotes and r.catcher == chess.A3
    # the cone of a rook hemmed in by its own pawns is its square alone
    b = chess.Board("8/8/1p3p2/p6k/RPr5/P5K1/8/8 w - - 0 2")
    assert cone(b, chess.A4) == {chess.A4: 0}
    fr, res, los = frozen(b, {chess.A3, chess.A4, chess.A5, chess.B4, chess.B6, chess.C4}, 2, Memo())
    assert fr and res == {"w": 0, "b": 5}
