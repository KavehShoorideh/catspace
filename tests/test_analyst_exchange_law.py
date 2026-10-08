"""The exchange law's prototype (catspace/incubator/analyst/exchange_law.py): the static exchange on textbook frames, the
hypotheses on constructed positions whose properties are asserted first, and the bridge (exact search) agreeing with the
law where it holds and beating it where a hypothesis fails (the tactic)."""
import chess

from catspace.incubator.analyst import exchange_law as X


def test_static_exchange_textbook_values():
    # a pawn on d5 defended by a pawn, attacked by a knight: Nxd5 exd5 -- the knight is lost for a pawn: -2
    b = chess.Board("4k3/8/4p3/3p4/8/4N3/8/4K3 w - - 0 1")
    assert chess.E3 in b.attackers(chess.WHITE, chess.D5) and chess.E6 in b.attackers(chess.BLACK, chess.D5)
    assert X.see(b, chess.D5) == (1 - 3, 2)                                    # Nxd5 exd5: a knight for a pawn, the capture's value
    # a knight defended once, attacked by a rook and a bishop: Bxc6 bxc6 Rxc6 -> +3 -3 ... bishop for knight then the rook takes the pawn: +1
    b2 = chess.Board("4k3/1p6/2n5/8/4B3/8/8/2R1K3 w - - 0 1")
    assert chess.E4 in b2.attackers(chess.WHITE, chess.C6) and chess.C1 in b2.attackers(chess.WHITE, chess.C6) and chess.B7 in b2.attackers(chess.BLACK, chess.C6)
    v, n = X.see(b2, chess.C6)
    assert v == 1 and n == 3                                                   # +3 (knight) -3 (bishop) +1 (pawn): the pawn's recapture is their choice; they stop
    # an undefended rook attacked by a knight: +5 in one capture
    b3 = chess.Board("4k3/8/8/3r4/8/4N3/8/4K3 w - - 0 1")
    assert X.see(b3, chess.D5) == (5, 1)


def test_the_law_holds_on_an_honest_frame_and_the_bridge_agrees():
    b = chess.Board("4k3/8/8/3n4/8/4N3/8/K7 w - - 0 1")                        # Nxd5 wins a knight; no check, no arrival possible
    assert not any(chess.Board("4k3/8/8/3n4/8/4N3/8/K7 b - - 0 1").gives_check(m) for m in chess.Board("4k3/8/8/3n4/8/4N3/8/K7 b - - 0 1").legal_moves)
    v = X.law(b, chess.D5)
    assert v["holds"], v["failing"]
    assert v["value"] == 3
    assert X.bridge(b, chess.D5, 2 * v["captures"] - 1) >= v["value"]


def test_a_pinned_defender_is_the_exception():
    # White Ra1, Nc3; Black Ke8, Rd5?? build: the knight on e3 attacks d5; the black bishop on e6 defends d5 but is pinned by the rook e1 to the king e8
    b = chess.Board("4k3/8/4b3/3p4/8/2N5/8/K3R3 w - - 0 1")
    assert chess.E6 in b.attackers(chess.BLACK, chess.D5) and b.is_pinned(chess.BLACK, chess.E6)   # the defender is pinned
    assert chess.C3 in b.attackers(chess.WHITE, chess.D5)
    v = X.law(b, chess.D5)
    assert not v["holds"] and "noPin" in v["failing"]                          # the law does not fire: a named exception
    assert v["value"] == -2                                                    # the naive exchange: take and lose the knight for a pawn
    assert X.bridge(b, chess.D5, 1) >= 1                                       # the exact search: the pawn is simply won -- the pin tactic


def test_a_check_in_between_is_the_exception():
    b = chess.Board("4k3/8/8/3p4/8/4N3/7r/4K3 w - - 0 1")                       # the rook h2 can check from h1: an in-between move exists
    assert not b.is_check() and any(chess.Board("4k3/8/8/3p4/8/4N3/7r/4K3 b - - 0 1").gives_check(m) for m in chess.Board("4k3/8/8/3p4/8/4N3/7r/4K3 b - - 0 1").legal_moves)
    v = X.law(b, chess.D5)
    assert not v["holds"] and "noCheck" in v["failing"]


def test_the_composed_tier_subtracts_the_intermezzo():
    """Nxd5 wins a knight; Black's bishop then takes the loose h2 pawn: the strict law does not fire (noOtherCapture),
    the composed tier claims 3 - 1 = 2 and the exact search agrees."""
    from catspace.incubator.analyst.exchange_law import bridge, intermezzo, law
    b = chess.Board("4k3/8/3b4/3n4/8/4N3/7P/1K6 w - - 0 1")
    assert not b.attackers(chess.BLACK, chess.D5) and not b.attackers(chess.WHITE, chess.H2) and chess.D6 in b.attackers(chess.BLACK, chess.H2)
    v = law(b, chess.D5)
    assert v["value"] == 3 and intermezzo(b, chess.D5) == 1 and v["composed"] == 2
    assert not v["holds"] and v["failing"] == ["noOtherCapture"] and v["composed_holds"]
    assert bridge(b, chess.D5, 3) >= 2
