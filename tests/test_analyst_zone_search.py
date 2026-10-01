"""The zone search (catspace/incubator/analyst/zone_search.py): alpha-beta inside an idea's zone, the pass searched."""
import chess

from catspace.incubator.analyst.fates import clear_caches
from catspace.incubator.analyst.zone_search import ZoneSearch
from catspace.incubator.analyst.frame_planner import runners


def test_kaveh_kf5_is_refuted_by_the_rook_sacrifice():
    # Kaveh 2026-09-28: the e5 pawn shields the king walk; Black gives a rook for it and the other rook skewers king and queen
    clear_caches()
    b = chess.Board("6k1/6p1/8/2r1P2Q/6K1/3P2P1/2r5/8 w - - 1 38")
    v, line, depth = ZoneSearch(20000).rank(b, [b.parse_san("Kf5")])[b.parse_san("Kf5")]
    assert v <= -4 and depth == 8
    assert [m.uci() for m in line[:5]] == ["g4f5", "c5e5", "f5e5", "c2c5", "e5e6"]
    got = ZoneSearch(20000).rank(b, [b.parse_san("Kf5"), b.parse_san("Qe8+")])
    assert got[b.parse_san("Qe8+")][0] > got[b.parse_san("Kf5")][0]
    assert got[b.parse_san("Kf5")][2] < got[b.parse_san("Qe8+")][2]   # more than the margin behind: not deepened


def test_rank_compares_every_candidate_at_the_same_depth():
    clear_caches()
    b = chess.Board("8/k2P4/N1n1N1p1/4p3/4P1rp/8/8/3K4 w - - 3 46")
    ms = [b.parse_san(s) for s in ("Nb4", "d8=Q", "Nac5")]
    got = ZoneSearch(20000).rank(b, ms)
    assert len({d for _, _, d in got.values()}) == 1
    assert got[ms[0]][0] > got[ms[1]][0]                      # the puzzle's Nb4 over the premature promotion


def test_a_threat_shown_by_the_pass_brings_its_defences_into_the_zone():
    # 0ROfv: after Kh8 Black threatens Qxh2#; White's pass is mated, but Rf2 (bearing on h2, outside the zone) holds.
    # Without the threat's squares joining the zone, Kh8 read as a forced mate (+1000)
    clear_caches()
    b = chess.Board("r5k1/p4B1p/2P3p1/4n2n/NP3p1q/1Q3Pb1/1P1B3P/R4R1K b - - 0 25")
    kh8, nxf7 = b.parse_san("Kh8"), b.parse_san("Nxf7")
    got = ZoneSearch(20000).rank(b, [kh8, nxf7])
    assert got[kh8][0] < 100
    assert got[nxf7][0] > got[kh8][0]


def test_the_checking_sacrifice_mate_is_found():
    # 0K3Ll: Rg1+ Kxg1 Rg8+ Bg4 Rxg4# (a regression guard: missed in the quiet benchmark, found here; the cause of that
    # miss is under study -- the check exemption in rank() is not what finds it)
    clear_caches()
    b = chess.Board("2kr4/pb3p1p/1p2p3/1P6/2PN4/P3P3/4BPrP/BN3R1K b - - 1 18")
    rg1, rxf2 = b.parse_san("Rg1+"), b.parse_san("Rxf2+")
    got = ZoneSearch(20000).rank(b, [rg1, rxf2])
    assert got[rg1][0] >= 900 and got[rg1][0] > got[rxf2][0]


def test_runners_are_passed_pawns_near_promotion_with_their_path():
    b = chess.Board("8/k2P4/N1n1N1p1/4p3/4P1rp/8/8/3K4 w - - 3 46")
    r = runners(b)
    assert {chess.D7, chess.D8} <= r                          # White's d-pawn, a step away
    assert {chess.H4, chess.H3, chess.H2, chess.H1} <= r      # Black's h-pawn, three steps away, nothing ahead of it
    assert chess.E4 not in r and chess.G6 not in r and chess.E5 not in r


def test_a_pass_is_a_null_move_in_the_line():
    clear_caches()
    b = chess.Board("6k1/6p1/8/2r1P2Q/6K1/3P2P1/2r5/8 w - - 1 38")
    v, line, _ = ZoneSearch(20000).value(b, b.parse_san("Kf4"))
    bb = b.copy()
    for m in line:                                            # every move of the line, passes included, replays
        assert not m or m in bb.legal_moves
        bb.push(m)


def test_pawn_moves_join_the_zone_when_the_planner_asks():
    import chess
    from catspace.incubator.analyst.zone_search import ZoneSearch, TacticalSpec
    b = chess.Board("4k3/8/8/8/8/8/P6R/4K3 w - - 0 1")
    zone = frozenset({chess.H2, chess.E8, chess.E1})
    assert chess.Move.from_uci("a2a4") not in ZoneSearch(spec=TacticalSpec()).moves(b, zone)
    assert chess.Move.from_uci("a2a4") in ZoneSearch(spec=TacticalSpec(pawn_moves=True)).moves(b, zone)
