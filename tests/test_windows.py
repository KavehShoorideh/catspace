"""Phase A of ANALYST_PLAN 41.7: observables and the explanation tree. Numbers printed as OBS lines."""
import os
import shutil

import pytest

from catspace.incubator.analyst.discovery import ledger as LG
from catspace.incubator.analyst.discovery import observables as OB
from catspace.incubator.analyst.discovery import teachers as TE
from catspace.incubator.analyst.discovery import windows as W

needs_corpus = pytest.mark.skipif(not os.path.exists(LG.LICHESS_PGN) or not shutil.which("zstd"), reason="Lichess corpus / zstd absent")
needs_sf = pytest.mark.skipif(not shutil.which("stockfish"), reason="stockfish absent")


@needs_corpus
def test_observable_kinds_on_a_known_game():
    game, _ = LG.find_game(0)                                   # lichess.org/yL7lY2kx
    obs = OB.stream(game)
    counts = {k: sum(o.kind == k for o in obs) for k in OB.ADAPTER.kinds}
    print(f"OBS observable kinds, game 0: {counts}")
    assert counts == dict(capture=4, promotion=0, check=3, castle=2, mate=0, resignation=1)
    assert all(o.fen_before and o.fen_after and o.source == "watched" for o in obs)
    cap = next(o for o in obs if o.kind == "capture")
    assert cap.delta >= 1 and cap.victim in ("white", "black") and cap.side != cap.victim


@needs_sf
def test_tree_has_one_branch_that_avoids_and_one_that_is_same():
    fen = "4k3/8/8/3q4/4P3/8/8/4K3 b - - 0 1"                    # the black queen is attacked by the e4 pawn
    D = OB.DOMAIN
    obs = OB.Observable("t", 0, "capture", "white", "black", 9, fen, "", "watched")
    t = TE.StockfishTeacher(depth=8, multipv=30, margin={"default": 100000})
    try:
        cls = {}
        for mv, cp in t.reasonable(fen, "x"):
            o = D.outcome(fen, mv, t.rollout(fen, mv, 4), cp, obs, 4)
            cls[D.san(fen, [mv])[0]] = o["cls"]
    finally:
        t.close()
    print(f"OBS hand-made queen-attacked position: {sum(v == 'SAME' for v in cls.values())} SAME, {sum(v == 'AVOIDED' for v in cls.values())} AVOIDED of {len(cls)} branches")
    assert "SAME" in cls.values() and "AVOIDED" in cls.values()
    assert cls["Qd6"] == "AVOIDED" and cls["Kd7"] == "SAME"


class FakeTeacher:
    """a non-engine teacher through the protocol: three reasonable moves at every ply."""
    def reasonable(self, fen, band):
        return [("m0", 0), ("m1", 0), ("m2", 0)]

    def score(self, fen, side):
        return 0

    def rollout(self, fen, move, plies):
        return [move]

    def close(self):
        pass


class FakeDomain:
    kinds = material_kinds = ()

    def side_to_move(self, fen):
        return "white"

    def apply(self, fen, mv):
        return fen

    def san(self, fen, mvs):
        return list(mvs)

    def outcome(self, fen, mv, line, cp, obs, plies):
        return dict(cls="SAME" if mv == "m0" else "AVOIDED", other=None, swing=0, line=[], end_fen=fen, end_bal=0, mated=None)


def test_branch_cap_ends_the_backward_walk():
    fens, moves = [f"f{i}" for i in range(11)], ["m0"] * 10
    obs = OB.Observable("t", 9, "capture", "white", "black", 1, "", "", "watched")
    t = W.build_tree(FakeTeacher(), FakeDomain(), fens, moves, obs, "x", cap=8)
    print(f"OBS stop rule: cap 8, 3 per ply -> {len(t['plies'])} plies, {t['reasonable']} branches, stop {t['stop']}")
    assert (len(t["plies"]), t["reasonable"], t["stop"]) == (2, 6, "cap")
    t = W.build_tree(FakeTeacher(), FakeDomain(), fens, moves, obs, "x", cap=1000)
    assert (len(t["plies"]), t["stop"]) == (10, "move one")
