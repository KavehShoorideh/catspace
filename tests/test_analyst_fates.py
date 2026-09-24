"""The pony and the juicer: a knight attacks a bishop the rook defends; the frame's outcomes are fates, and a goal
('their bishop captured by my knight') is answered with who achieves it."""
import chess

from catspace.incubator.analyst import fates as F
from catspace.incubator.analyst.chess_domain import domain

FEN = "3r2k1/8/3b4/8/4N3/8/8/6K1 w - - 0 1"       # white knight e4 attacks the bishop d6; the rook d8 defends it
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
