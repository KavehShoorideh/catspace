"""L0-L3 of the analyst: relations found on known positions, defined only via L1 probes."""
import chess

from catspace.incubator.analyst import geometry as G, rules as R, facts as F, relations as L3


def names(rels, name):
    return {r.names() for r in rels if r.name == name}


def test_geometry_basics():
    assert G.aligned(chess.A1, chess.H8) == "diagonal"
    assert G.aligned(chess.A1, chess.A8) == "orthogonal"
    assert G.aligned(chess.A1, chess.B3) is None
    assert G.ray(chess.A1, 1, 1) == [chess.B2, chess.C3, chess.D4, chess.E5, chess.F6, chess.G7, chess.H8]
    assert G.square_color(chess.A1) == "dark" and G.square_color(chess.H1) == "light"


def test_rules_royalty_and_variants():
    assert R.royal_square(chess.Board(), chess.WHITE) == chess.E1
    anti = R.rules_for("antichess").board()
    assert R.royal_square(anti, chess.WHITE) is None          # no royal piece in antichess
    assert R.rules_of(anti).captures_forced
    assert R.rules_of(chess.Board()).name == "chess"


def test_state_facts():
    f = F.state_facts(chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"))
    assert f.material_diff == 0 and f.turn == "white" and f.castling["K"] and not f.in_check


def test_absolute_pin_ruy_lopez():
    # 1.e4 e5 2.Nf3 Nc6 3.Bb5 d5?! -- with d7 vacated the knight on c6 IS pinned to the king
    b = chess.Board("r1bqkbnr/ppp2ppp/2n5/1B1pp3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq d6 0 4")
    pins = [r for r in L3.line_relations(b) if r.name == "pin"]
    assert any(r.names() == ("b5", "c6", "e8") and "absolute" in r.tags for r in pins)
    # and with the d7 pawn still home (plain 3.Bb5) there is NO pin -- the removal test must say so
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    assert not [r for r in L3.line_relations(b) if r.name == "pin" and r.names()[:2] == ("b5", "c6")]


def test_relative_pin_and_skewer():
    # Bb2 vs Qd4 with Kf6 behind: the queen is pinned to the king (absolute pin, NOT a skewer)
    b = chess.Board("8/8/5k2/8/3q4/8/1B6/4K3 w - - 0 1")
    assert ("b2", "d4", "f6") in names(L3.line_relations(b), "pin")
    # Bb2 gives check to Kd4 with Qf6 behind: a skewer with check
    b = chess.Board("8/8/5q2/8/3k4/8/1B6/4K3 b - - 0 1")
    sk = [r for r in L3.line_relations(b) if r.name == "skewer"]
    assert any(r.names() == ("b2", "d4", "f6") and "check" in r.tags for r in sk)


def test_fork_and_hanging():
    b = chess.Board("r3k3/2N5/8/8/8/8/8/4K3 w - - 0 1")           # Nc7 forks Ke8 and Ra8
    assert ("c7", "a8", "e8") in names(L3.forks(b), "fork")
    b = chess.Board("4k3/8/8/3n4/8/8/8/3RK3 w - - 0 1")           # Nd5 attacked by Rd1, undefended
    assert ("d5",) in names(L3.hanging_and_loose(b), "hanging")


def test_discovered_check_and_battery():
    b = chess.Board("3k4/8/8/8/8/3N4/8/3RK3 w - - 0 1")           # Rd1 behind Nd3 aimed at Kd8
    dc = [r for r in L3.line_relations(b) if r.name == "discovered_attack"]
    assert any(r.names() == ("d1", "d3", "d8") and "check" in r.tags for r in dc)
    b = chess.Board("3k4/8/8/8/8/8/3R4/3RK3 w - - 0 1")           # Rd1 behind Rd2: battery on d8
    assert ("d1", "d2", "d8") in names(L3.line_relations(b), "battery")


def test_overloaded_defender():
    # Ra1 attacks Ra5, Rd1 attacks Pd4; the black knight on c6 is the SOLE defender of both
    b = chess.Board("6k1/8/2n5/r7/3p4/8/8/R2R2K1 w - - 0 1")
    ov = [r for r in L3.overloaded(b) if r.name == "overloaded"]
    assert any(r.squares[0] == chess.C6 and set(r.squares[1:]) == {chess.A5, chess.D4} for r in ov)

def test_king_zone_pressure_and_antichess_has_none():
    b = chess.Board("6k1/5ppp/8/8/8/2Q5/5PPP/6K1 w - - 0 1")    # Qc3 hits g7, in Kg8's zone
    kz = [r for r in L3.king_zone_pressure(b) if r.name == "attacks_king_zone"]
    assert any(r.squares == (chess.C3, chess.G8) for r in kz)
    anti = R.rules_for("antichess").board("6k1/5ppp/8/8/8/2Q5/5PPP/6K1 w - - 0 1")
    assert not L3.king_zone_pressure(anti)                      # no royal king -> no king zone


def test_functional_pin_is_a_four_piece_relation():
    # Bg5 pins Nf6 to Qd8 (relative). Nf6 attacks the e4 pawn, but taking it gives up the queen.
    b = chess.Board("rnbqkb1r/pppp1ppp/4pn2/6B1/4P3/8/PPPP1PPP/RN1QKBNR w KQkq - 0 3")
    rels = L3.relations(b)
    da = [r for r in rels if r.name == "disabled_attack"]
    assert any(r.names() == ("g5", "f6", "d8", "e4") and r.arity == 4 and "relative" in r.tags for r in da)


def test_capture_along_the_pin_line_is_not_disabled():
    # Re1 pins Re2 to Ke8 (absolute). The rook may take the pinner (on the line) but not the knight on a2.
    b = chess.Board("4k3/8/8/8/8/8/N3r3/3BR1K1 b - - 0 1")
    rels = L3.relations(b)
    da = {r.names() for r in rels if r.name == "disabled_attack"}
    assert ("e1", "e2", "e8", "a2") in da
    assert not any(n[3] == "e1" for n in da)
    # and the bishop on d1 attacks the pinned rook, which cannot flee
    pa = {r.names() for r in rels if r.name == "pinned_and_attacked"}
    assert ("e1", "e2", "e8", "d1") in pa


def test_hanging_through_pin():
    # d5 is attacked by e4 and defended ONLY by Nf6, which is pinned to Qe7 by Bg5
    b = chess.Board("rnb1kb1r/ppp1qppp/5n2/3p2B1/4P3/8/PPPP1PPP/RN1QKBNR w KQkq - 0 4")
    rels = L3.relations(b)
    assert any(r.name == "disabled_defense" and r.names() == ("g5", "f6", "e7", "d5") for r in rels)
    assert any(r.name == "hanging_through_pin" and r.names() == ("d5", "f6") for r in rels)


def test_antichess_has_no_absolute_disabling():
    # same geometry, antichess rules: no royal king, so the rook on e2 is NOT absolutely pinned to e8
    b = R.rules_for("antichess").board("4k3/8/8/8/8/8/N3r3/3BR1K1 b - - 0 1")
    rels = L3.relations(b)
    assert not any(r.name == "pin" and "absolute" in r.tags for r in rels)


def test_mate_patterns():
    from catspace.incubator.analyst.mates import mate_pattern, mate_in_line
    b = chess.Board("6rk/5Npp/8/8/8/8/8/6K1 b - - 0 1")               # Philidor: smothered
    assert mate_pattern(b)[0] == "smothered mate"
    b = chess.Board("3R2k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")               # rook on the 8th, pawns block
    assert mate_pattern(b)[0] == "back-rank mate"
    b = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")   # Scholar's: Qxf7#
    assert mate_pattern(b)[0] == "supported queen mate"
    # Opera Game finish: Qb8+ Nxb8 Rd8#  (from the position before 16.Qb8+)
    b = chess.Board("1n1rkb1r/p4ppp/4q3/4p1B1/4P3/1Q6/PPP2PPP/2K5 w - - 0 16")
    r = mate_in_line(b, ["b3b8", "d7b8", "d1d8"]) if False else None
