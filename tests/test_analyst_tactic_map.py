"""The tactic map (catspace/incubator/analyst/tactic_map.py): the value of token states, and the rule preferring, among
tied moves, the one that leaves us a winning tactic and them none -- on constructed positions whose properties are
asserted first. Tables: the c300 census (fitted in the repo's data)."""
import chess
import pytest

from catspace.incubator.analyst import frame_player as FP
from catspace.incubator.analyst import tactic_map as TM


@pytest.fixture()
def tm():
    m = TM.TacticMap("c300", elo_us=1800, elo_them=1800)
    assert m.t["hold"]["pin|all"] > 0 and m.t["exposed"]["pin|all"] < 0
    return m


def test_value_of_token_states(tm):
    assert tm.value_of_tokens([], []) == 0.0
    assert tm.value_of_tokens([("pin", 0, "")], []) > 0
    assert tm.value_of_tokens([], [("pin", 0, "")]) < 0
    assert 0 < tm.value_of_tokens([("pin", 1, "no strike")], []) < tm.value_of_tokens([("pin", 0, "")], [])
    assert tm.value_of_tokens([("pin", 0, "")], [("pin", 0, "")]) == pytest.approx(tm.value_of_tokens([("pin", 0, "")], []) + tm.value_of_tokens([], [("pin", 0, "")]))


def test_the_rule_prefers_the_move_that_wins_a_tactic(tm):
    # White Ra1 can go to a8 (a skewer/pin on the king e8 with the knight ... ) -- simpler: Rd1 can pin the knight d5 to the king d8
    b = chess.Board("3k4/8/8/3n4/8/8/8/R3K3 w - - 0 1")
    pin_move, other = chess.Move.from_uci("a1d1"), chess.Move.from_uci("a1b1")
    assert pin_move in b.legal_moves and other in b.legal_moves
    nb = b.copy(); nb.push(pin_move)
    assert nb.is_pinned(chess.BLACK, chess.D5)                            # the knight is pinned to its king
    assert not nb.attackers(chess.BLACK, chess.D5)                       # and undefended: the pin wins it
    assert tm.value_after(b, pin_move) > tm.value_after(b, other)
    sc = TM.map_fire(b, [pin_move.uci(), other.uci()])
    assert sc[pin_move.uci()][0] > sc[other.uci()][0]


def test_the_rule_is_registered_as_a_conjecture_that_reads_the_ties():
    r = FP.RULE_BOOK["tactic_map"]()
    assert r.tier == "conjecture" and r.reads_ledger
    assert "tactic_map" in FP.RULE_ORDER


def test_value_after_does_not_double_charge_the_recapture(tm):
    # Nxf6 is an even trade (knight for knight): the ledger already subtracts the recapture; the map must not again
    b = chess.Board("7k/ppp2pp1/5n1p/8/4N3/3P4/PPP2PPP/6K1 w - - 0 1")
    assert chess.F6 in b.attacks(chess.E4) and b.piece_at(chess.F6) == chess.Piece(chess.KNIGHT, chess.BLACK)
    nb = b.copy(); nb.push_san("Nxf6"); assert nb.attackers(chess.BLACK, chess.F6)       # it is recaptured
    trade, quiet = tm.value_after(b, chess.Move.from_uci("e4f6")), tm.value_after(b, chess.Move.from_uci("g1h1"))
    assert abs(trade - quiet) < 0.5


def test_a_checking_fork_is_not_read_as_tactic_free(tm):
    b = chess.Board("r3k3/pp3ppp/8/3N4/5B2/8/PPP2PPP/6K1 w - - 0 1")
    m = chess.Move.from_uci("d5c7")
    nb = b.copy(); nb.push(m); assert nb.is_check() and chess.A8 in nb.attacks(chess.C7)     # the royal fork, with check
    assert tm.value_after(b, m) >= tm.value_after(b, chess.Move.from_uci("g1h1")) - 0.1


def test_a_stalemating_move_is_worth_nothing(tm):
    b = chess.Board("k7/8/8/2K5/8/8/8/1Q6 w - - 0 1")
    m = chess.Move.from_uci("b1b6")
    nb = b.copy(); nb.push(m); assert nb.is_stalemate()
    assert tm.value_after(b, m) == 0.0
