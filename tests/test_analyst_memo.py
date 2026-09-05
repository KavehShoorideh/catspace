"""The concept store: a fact keyed by its footprint is served from any position whose pieces on that footprint are
the same, and never served wrongly (memo answers == fresh answers over positions and their children)."""
import random
import chess

from catspace.incubator.analyst.memo import Memo, attack_reads, pin_line
from catspace.incubator.analyst.defense import defense, _defense
from catspace.incubator.analyst.reach import reach, _reach


def test_footprints_are_small_and_shaped():
    b = chess.Board("1r1r2k1/p1p2ppp/b3pb2/2Pq4/2PP4/8/P2NNPPP/R2Q1RK1 b - - 0 16")
    m = attack_reads(b, chess.C4)
    assert chess.C4 in chess.SquareSet(m) and chess.D5 in chess.SquareSet(m) and chess.A6 in chess.SquareSet(m)
    assert chess.H8 not in chess.SquareSet(m)                 # not on any line into c4
    # a knight on the ray cuts it: nothing beyond a non-participant matters
    b2 = chess.Board("4k3/8/8/8/2n5/8/8/R3K3 w - - 0 1")     # rook a1, black knight c4? use a file: a1 rook, a4 knight, a7 target
    b2 = chess.Board("4k3/p7/8/8/n7/8/8/R3K3 w - - 0 1")
    m = attack_reads(b2, chess.A7)
    assert chess.A4 in chess.SquareSet(m) and chess.A1 not in chess.SquareSet(m)
    # pin line: king e1, piece e4 -> e2, e3, e4, e5..e8 (the far side), not d-file squares
    b3 = chess.Board("4k3/8/8/8/4N3/8/8/4K3 w - - 0 1")
    pl = chess.SquareSet(pin_line(b3, chess.WHITE, chess.E4))
    assert chess.E8 in pl and chess.E2 in pl and chess.D4 not in pl


def test_memo_serves_across_positions_and_never_wrongly():
    random.seed(3)
    b = chess.Board("2kr1bnr/p1pp4/1p3p2/2P2qp1/8/4BNp1/PP2PP1P/2RQ1RK1 w - - 0 18")
    memo = Memo()
    for sq in chess.scan_forward(b.occupied):
        defense(b, sq, memo=memo)
    r0 = reach(b, chess.G5, chess.F3, 3, memo=memo)
    assert r0 is not None and r0.squares == (chess.G4,)
    misses0 = memo.stats["miss"]
    # a move far from the g5-g4-f3 story: the store serves the reach (no recomputation)
    c = b.copy(); c.push_san("a3")
    r1 = reach(c, chess.G5, chess.F3, 3, memo=memo)
    assert r1 == r0 and memo.stats["miss"] == misses0
    # every child: served answers equal fresh answers
    for mv in list(b.legal_moves)[:12]:
        c = b.copy(stack=False); c.push(mv)
        for sq in chess.scan_forward(c.occupied):
            a, f = defense(c, sq, memo=memo), _defense(c, sq)
            assert (a is None) == (f is None)
            if a:
                assert a.effective == f.effective and a.see == f.see and [x.reason for x in a.disqualified] == [x.reason for x in f.disqualified]
        assert reach(c, chess.G5, chess.F3, 3, memo=memo) == _reach(c, chess.G5, chess.F3, 3, None)
    assert memo.hit_rate() > 0.3


def test_memo_persists(tmp_path):
    b = chess.Board("2kr1bnr/p1pp4/1p3p2/2P2qp1/8/4BNp1/PP2PP1P/2RQ1RK1 w - - 0 18")
    path = str(tmp_path / "memo.sqlite")
    m = Memo(path)
    defense(b, chess.B6, memo=m); reach(b, chess.G5, chess.F3, 3, memo=m)
    m.save()
    m2 = Memo.load(path)
    assert len(m2) == len(m)
    defense(b, chess.B6, memo=m2); reach(b, chess.G5, chess.F3, 3, memo=m2)
    assert m2.stats["hit"] == 2 and m2.stats["miss"] == 0
