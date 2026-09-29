"""The zone search (catspace/incubator/analyst/zone_search.py): alpha-beta inside an idea's zone, the pass searched."""
import chess

from catspace.incubator.analyst.fates import clear_caches
from catspace.incubator.analyst.zone_search import ZoneSearch, runners


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
