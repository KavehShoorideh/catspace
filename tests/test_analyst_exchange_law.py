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


def test_an_xray_needs_a_participant_in_front_and_a_discovered_defender_is_an_arrival():
    from catspace.incubator.analyst.exchange_law import hypotheses
    b = chess.Board("4k3/8/8/3n4/3P4/4N3/8/1K1R4 w - - 0 1")        # the d4 pawn (not a participant) blocks our rook's line to d5
    assert chess.D4 in [t for t in chess.SQUARES if chess.BB_SQUARES[t] & chess.between(chess.D1, chess.D5)] and not b.attackers(chess.BLACK, chess.D5)
    assert hypotheses(b, chess.D5)["noXray"] is True                   # nothing reveals the rook: no x-ray
    b2 = chess.Board("4k3/8/8/3n4/8/3R4/8/1K1Q4 w - - 0 1")           # our rook d3 captures d5 and reveals the queen behind it
    assert chess.D3 in b2.attackers(chess.WHITE, chess.D5) and chess.D1 not in b2.attackers(chess.WHITE, chess.D5)
    assert hypotheses(b2, chess.D5)["noXray"] is False
    b3 = chess.Board("3rk3/3b4/8/3n4/8/4N3/8/1K6 w - - 0 1")          # their bishop d7 blocks their rook's line; Bd7 moving reveals a defender
    assert chess.D7 not in b3.attackers(chess.BLACK, chess.D5)
    assert hypotheses(b3, chess.D5)["noArrival"] is False


def test_the_xray_chain_is_recursive():
    """Alekhine's gun: queen behind two rooks -- the second rook is revealed by the first, the queen by the second; a
    bishop (not a rook) between breaks the chain."""
    from catspace.incubator.analyst.exchange_law import hypotheses
    gun = chess.Board("4k3/8/8/3n4/8/3R4/3R4/1K1Q4 w - - 0 1")
    assert chess.D3 in gun.attackers(chess.WHITE, chess.D5) and chess.D2 not in gun.attackers(chess.WHITE, chess.D5)
    assert hypotheses(gun, chess.D5)["noXray"] is False
    broken = chess.Board("4k3/8/8/3n4/8/3R4/3B4/1K1Q4 w - - 0 1")     # a bishop on d2 never attacks d5 along the file
    assert hypotheses(broken, chess.D5)["noXray"] is True
    diagonal = chess.Board("4k3/8/8/3n4/2P5/1B6/Q7/1K6 w - - 0 1")    # pawn c4 takes d5, reveals the bishop b3, which reveals the queen a2
    assert chess.C4 in diagonal.attackers(chess.WHITE, chess.D5) and chess.B3 not in diagonal.attackers(chess.WHITE, chess.D5)
    assert hypotheses(diagonal, chess.D5)["noXray"] is False


def test_the_static_exchange_with_xrays_counts_the_gun():
    """Alekhine's gun against a knight defended by a bishop: the plain count loses the exchange (Rxd5 Bxd5), the x-ray
    count nets a pawn (Rxd5 Bxd5 Rxd5), and the exact search agrees."""
    from catspace.incubator.analyst.exchange_law import bridge, law, see, see_x
    gun = chess.Board("4k3/8/4b3/3n4/3R4/3R4/3Q4/1K6 w - - 0 1")
    assert chess.E6 in gun.attackers(chess.BLACK, chess.D5) and list(gun.attackers(chess.WHITE, chess.D5)) == [chess.D4]
    assert see(gun, chess.D5)[0] == -2 and see_x(gun, chess.D5)[0] == 1 and bridge(gun, chess.D5, 5) >= 1 and law(gun, chess.D5)["xray_holds"]   # Rxd5 Bxd5 Rxd5
    plain = chess.Board("4k3/8/8/3n4/8/4N3/8/1K6 w - - 0 1")
    assert see_x(plain, chess.D5) == see(plain, chess.D5) == (3, 1)


def test_every_exchange_carries_its_variant_names():
    from catspace.incubator.analyst.exchange_law import law
    assert law(chess.Board("4k3/8/8/3n4/8/4N3/8/1K6 w - - 0 1"), chess.D5)["variants"] == ["plain exchange"]
    assert "intermezzo exchange" in law(chess.Board("4k3/8/3b4/3n4/8/4N3/7P/1K6 w - - 0 1"), chess.D5)["variants"]
    assert "x-ray exchange" in law(chess.Board("4k3/8/4b3/3n4/3R4/3R4/3Q4/1K6 w - - 0 1"), chess.D5)["variants"]
    assert "king exchange" in law(chess.Board("8/8/8/3n4/4K3/8/8/7k w - - 0 1"), chess.D5)["variants"]
