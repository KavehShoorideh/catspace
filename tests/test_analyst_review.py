"""Engine-free pieces of the review pipeline: exchange nets, the static exchange evaluation edge cases, mate units."""
import chess

from catspace.incubator.analyst.review import exchange_net
from catspace.incubator.analyst.facts import see
from catspace.incubator.analyst.engine import Line
from catspace.incubator.analyst.verbalize import cp_str


def test_exchange_net_counts_our_own_capture():
    # AUjG3sl4 before 18...Bxd4: Bxd4 Nxd4 Rxd4 is bishop for knight AND pawn: +1 for Black, not an even trade
    b = chess.Board("3r2k1/p1p2ppp/2p1pb2/2P5/2qP4/P7/1r2NPPP/R2Q1RK1 b - - 0 18")
    mv = b.parse_san("Bxd4")
    assert exchange_net(b, mv, ["e2d4", "d8d4"]) == 1
    # AsXfL6OX 11...Bxf3 12.Nxf3: bishop for knight, even
    b = chess.Board("r3kbnr/pbpp1ppp/1p6/5qN1/8/5NP1/PPPBPP1P/R2QK2R b KQkq - 5 11")
    assert exchange_net(b, b.parse_san("Bxf3"), ["g5e4", "f5e4"]) is None          # the reply lands elsewhere: not this exchange
    assert exchange_net(b, b.parse_san("Bxf3"), ["g5f3", "g8f6"]) == 0             # knight takes back: even
    # the reply's own capture that nobody answers is worth what it took
    b2 = chess.Board("4k3/8/8/3p4/8/8/8/3RK3 w - - 0 1")
    assert exchange_net(b2, b2.parse_san("Rxd5"), [], require_recapture=False) == 1
    assert exchange_net(b2, b2.parse_san("Rxd5"), []) is None


def test_see_edge_cases():
    b = chess.Board("4k3/8/8/8/8/8/8/r1Kr4 w - - 0 1")          # Kxd1 would walk into the a1 rook's line: not a capture
    assert see(b, chess.D1, chess.WHITE) == 0.0
    b = chess.Board("4k3/8/8/8/8/8/8/2Kr4 w - - 0 1")
    assert see(b, chess.D1, chess.WHITE) == 5.0
    b = chess.Board("4k3/8/4p3/3n4/8/8/3R4/3R2K1 w - - 0 1")     # battery: Rxd5 exd5 Rxd5 = 3 - 5 + 1 = -1 for the starter
    assert see(b, chess.D5, chess.WHITE) == -1.0


def test_mate_is_in_moves():
    # python-chess Mate(n) is n MOVES; the prose must not halve it
    assert cp_str(9985, True, 15) == "#15"
    line = Line("a1a8", ("a1a8",), 9999, 1)
    assert line.mate == 1
