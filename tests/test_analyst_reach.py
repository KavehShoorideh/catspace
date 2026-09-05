"""reach.py: flip conditions, safe reach from the rules, the progress map; the same placement under antichess differs."""
import chess

from catspace.incubator.analyst.reach import reach, flip_conditions, progress, progress_lines
from catspace.incubator.analyst import rules as R


def test_reach_follows_the_move_graph():
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
    p = reach(b, chess.B1, chess.C7)
    assert p is not None and [chess.square_name(s) for s in p.squares] == ["c3", "d5"]     # the knight hops to a square that hits c7
    assert reach(b, chess.F1, chess.C7) is None                                           # a light-squared bishop never attacks c7
    assert reach(b, chess.C1, chess.C7) is None                                           # boxed in by its own pawns
    # a landing square must be safe: the knight will not stop on a square a pawn takes it for free
    b = chess.Board("4k3/8/8/8/8/2p5/8/1N2K3 w - - 0 1")   # Nb1: c3 has a black pawn; d2 is attacked by the c3 pawn
    p = reach(b, chess.B1, chess.E4)                        # to attack e4 it could go d2 (unsafe) or a3-c4? a3 attacks c4/b5; c4? ...
    assert p is None or chess.D2 not in p.squares


def test_flip_and_progress_statuses():
    # AUjG3sl4 after 16.c4, Black to move: d4 and c4 fall now; c5 is held but its defender d4 falls first
    b = chess.Board("1r1r2k1/p1p2ppp/b3pb2/2Pq4/2PP4/8/P2NNPPP/R2Q1RK1 b - - 0 16")
    ts = {t.square: t for t in progress(b, chess.BLACK, top=64)}
    assert ts[chess.D4].status == "falls" and ts[chess.C4].status == "falls"
    assert ts[chess.C5].status == "undermine" and ts[chess.C5].remove == chess.D4
    # AsXfL6OX before 18.fxg3, Black: the f3 knight is held; g5-g4 joins in one move
    b = chess.Board("2kr1bnr/p1pp4/1p3p2/2P2qp1/8/4BNp1/PP2PP1P/2RQ1RK1 w - - 0 18")
    ts = {t.square: t for t in progress(b, chess.BLACK, top=64)}
    t = ts[chess.F3]
    assert t.status == "joinable" and t.joiner.piece == chess.G5 and t.joiner.squares == (chess.G4,)
    fl = flip_conditions(b, chess.F3, chess.BLACK)
    assert any(f.kind == "add_attacker" and f.max_value >= 1 for f in fl)
    lines = progress_lines(b, chess.BLACK)
    assert any("joins from g4 in 1 move" in l for l in lines)


def test_progress_changes_with_the_rules():
    # chess: a king may not take a pawn a pawn defends (into check); antichess has no royal king, so it falls
    fen = "8/8/8/2k5/8/2p5/1p6/1K6 w - - 0 1"
    ch = {t.square: t.status for t in progress(chess.Board(fen), chess.WHITE, top=64)}
    ab = R.rules_for("antichess").board(); ab.set_fen(fen)
    an = {t.square: t.status for t in progress(ab, chess.WHITE, top=64)}
    assert ch[chess.B2] in ("held", "unreachable") and an[chess.B2] == "falls"
