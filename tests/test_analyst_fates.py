"""The pony and the juicer: a knight attacks a bishop the rook defends; the frame's outcomes are fates, and a goal
('their bishop captured by my knight') is answered with who achieves it."""
import chess

from catspace.incubator.analyst import fates as F
from catspace.incubator.analyst.chess_domain import domain

FEN = "3r2k1/8/3b4/8/4N3/8/8/7K w - - 0 1"       # white knight e4 attacks the bishop d6; the rook d8 defends it (king h1: off the c5-g1 diagonal, or Nf2 would be pinned)
IDS = [chess.E4, chess.D6, chess.D8]              # pony, juicer, rook


def test_outcomes_are_fates_with_captors():
    dom = domain(oracle="frame")
    b = chess.Board(FEN)
    table = F.outcomes_table(dom, b, IDS, chess.E4)
    top = table[0]
    trade = next(o for o in table if o.move == chess.Move.from_uci("e4d6"))
    assert trade.fates[1].alive is False and trade.fates[1].captor == 0 and trade.fates[1].ply == 1   # the juicer, taken by the pony
    assert trade.fates[0].alive is False and trade.fates[0].captor == 2 and trade.fates[0].ply == 2   # the pony, taken by the rook
    assert trade.value == 0.0                                                                          # materially even
    assert trade.achieved_by(1) == 0
    quiet = next(o for o in table if o.move == chess.Move.from_uci("e4f2"))          # g3 would be on the bishop's diagonal
    assert all(f.alive for f in quiet.fates) and quiet.counts[0][0] > 0
    assert "taken by the pony" in F.describe(trade, b, ["the pony", "the juicer", "the rook"])
    assert isinstance(top.signature, dict) and top.signature


def test_goal_is_achievable_by_the_pony_but_not_without_losing_it():
    b = chess.Board(FEN)
    ans = F.achievable(b, IDS, chess.E4, F.captured_by(1, 0))
    assert ans["yes"] and ans["move"] == chess.Move.from_uci("e4d6")
    ans2 = F.achievable(b, IDS, chess.E4, F.both(F.captured(1), F.survives(0)))
    assert not ans2["yes"]                                                                             # the rook always takes back


def test_intermezzi_are_forced_passes_at_each_ply():
    """Kaveh 2026-09-24: an intermezzo is a move spent outside the frame. If black's reply (ply 2) is spent elsewhere,
    the pony takes the juicer and lives: +3; if white's next move (ply 3) is spent elsewhere, the trade stands."""
    b = chess.Board(FEN)
    im = F.intermezzi(b, IDS, chess.Move.from_uci("e4d6"))
    by_ply = {ply: (who, v, f) for ply, who, v, f in im}
    assert by_ply[2][0] is chess.BLACK and by_ply[2][1] == 3.0 and by_ply[2][2][0].alive           # black's tempo elsewhere: the pony survives with the bishop
    assert by_ply[3][0] is chess.WHITE and by_ply[3][1] == 0.0 and not by_ply[3][2][0].alive       # white's tempo elsewhere after Nxd6: the rook takes back anyway
    dom = domain(oracle="frame")
    o = F.outcome(dom, b, IDS, chess.Move.from_uci("e4d6"), with_intermezzi=True)
    assert len(o.intermezzi) == 3


def test_intermezzo_by_composition_needs_no_big_frame():
    """Frame A (the pony takes the juicer) with a two-ply intermezzo inserted at black's reply: black spends ply 2
    elsewhere (say a check), white answers it at ply 3, and A resumes -- the rook still takes back at ply 4, so A's
    part is 0; the composed value is 0 plus whatever frame B earned. A one-ply intermezzo at ply 2 alone leaves the
    pony alive (+3) because the depth runs out before the rook's reply."""
    b = chess.Board(FEN)
    v2, f2 = F.with_passes(b, IDS, chess.Move.from_uci("e4d6"), {2})
    assert v2 == 3.0 and f2[0].alive
    v23, f23 = F.with_passes(b, IDS, chess.Move.from_uci("e4d6"), {2, 3}, depth=4)
    assert v23 == 0.0 and not f23[0].alive and f23[0].ply == 4
    c = F.compose(b, IDS, chess.Move.from_uci("e4d6"), k=2, b_value=1.0, b_plies=2, depth=4)
    assert c["value"] == 1.0 and c["a_value"] == 0.0 and c["at"] == 2


def test_en_passant_in_the_frame_episode():
    """Review 2026-09-24: a pass after a double pawn push is legal (the en-passant right lapses; a bare turn flip kept
    it and read as illegal), and a pawn taken en passant was taken by the frame pawn that moved, not by an outsider."""
    from catspace.incubator.analyst.chess_domain import passed
    dom = domain(oracle="frame")
    with_ep, without = chess.Board("4k3/8/8/3n4/4P3/8/8/4K3 b - e3 0 1"), chess.Board("4k3/8/8/3n4/4P3/8/8/4K3 b - - 0 1")
    assert passed(with_ep) is not None and passed(with_ep).turn == chess.WHITE
    ids = [chess.D5, chess.E4]
    assert F.pass_value(with_ep, ids, chess.BLACK, 2) == F.pass_value(without, ids, chess.BLACK, 2)
    b = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    o = F.outcome(dom, b, [chess.E5, chess.D5], chess.Move.from_uci("e5d6"))
    assert o.fates[1].alive is False and o.fates[1].captor == 0


def test_achievable_searches_as_deep_as_the_outcome():
    """The knight fork wins the rook on ply 3; the outcome sees it, so the goal is achievable (d - 1 plies missed it)."""
    dom = domain(oracle="frame")
    b = chess.Board("r3k3/8/8/1N6/8/8/8/6K1 w - - 0 1"); ids = [chess.B5, chess.A8]
    assert F.outcome(dom, b, ids, chess.Move.from_uci("b5c7")).fates[1].alive is False
    a = F.achievable(b, ids, chess.B5, F.captured(1))
    assert a["yes"] and a["move"] == chess.Move.from_uci("b5c7")
