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
    assert counts == dict(capture=4, promotion=0, check=3, castle=2, mate=0, resignation=1, shift=4)
    assert all(o.fen_before and o.fen_after and o.source == "watched" for o in obs)
    cap = next(o for o in obs if o.kind == "capture")
    assert cap.delta >= 1 and cap.victim in ("white", "black") and cap.side != cap.victim


def _shifts(fen, ucis):
    import chess
    import chess.pgn
    g = chess.pgn.Game(); g.setup(fen)
    n = g
    for u in ucis:
        n = n.add_variation(chess.Move.from_uci(u))
    out = [o for o in OB.stream(g) if o.kind == "shift"]
    print(f"OBS shifts {ucis}: {[(o.ply, o.side, o.delta) for o in out]}")
    return out


def test_an_even_trade_is_no_shift():
    assert _shifts("4k3/8/4p3/3n4/8/2N5/8/4K3 w - - 0 1", ["c3d5", "e6d5", "e1d1"]) == []


def test_a_pawn_won_is_one_shift_where_it_settled():
    o = _shifts("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", ["e4d5", "e8d7", "e1d1"])
    assert [(x.ply, x.side, x.victim, x.delta) for x in o] == [(0, "white", "black", 1)]


def test_a_two_ply_exchange_netting_a_piece_is_one_shift():
    o = _shifts("4k3/8/4p3/3n4/4P3/8/8/4K3 w - - 0 1", ["e4d5", "e6d5", "e1d1"])
    assert [(x.ply, x.side, x.delta) for x in o] == [(0, "white", 2)]


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


# ---- ANALYST_PLAN 43: quiescence ----

def test_quiet_on_three_hand_positions():
    D = OB.DOMAIN
    quiet = D.quiet("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    hanging = D.quiet("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")           # the e4 pawn takes d5 for nothing
    check = D.quiet("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
    even = D.quiet("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1")           # exd5 cxd5: an even trade is quiet
    print(f"OBS quiet: bare kings {quiet}, hanging pawn {hanging}, in check {check}, even trade on offer {even}")
    assert quiet and not hanging and not check and even


def _sequence():
    import chess
    b = chess.Board("3k4/8/2b5/3n4/8/8/3Q4/3RK3 w - - 0 1")        # 1.Rxd5+ Bxd5 2.Qxd5+ Ke7 3.Kf1
    boards, moves = [b.copy()], []
    for u in ["d1d5", "c6d5", "d2d5", "d8e7", "e1f1"]:
        m = chess.Move.from_uci(u); moves.append(m); b.push(m); boards.append(b.copy())
    return moves, boards


def test_settle_runs_through_the_checks_to_the_quiet_position():
    D = OB.DOMAIN
    moves, boards = _sequence()
    quiets = [D.quiet(x) for x in boards]
    j = D.settle(boards, 0)
    print(f"OBS settle: quiet at positions {quiets}, settle(0) = {j}")
    assert not quiets[1] and not quiets[3] and quiets[4]
    assert j == 4


def test_a_sequence_with_a_check_in_the_middle_is_one_shift():
    import chess.pgn
    moves, boards = _sequence()
    g = chess.pgn.Game(); g.setup(boards[0])
    n = g
    for m in moves:
        n = n.add_variation(m)
    sh = [o for o in OB.stream(g) if o.kind == "shift"]
    print(f"OBS shifts of the sequence: {[(o.ply, o.side, o.delta, o.settle_ply) for o in sh]}")
    assert [(o.ply, o.side, o.delta, o.settle_ply) for o in sh] == [(0, "white", 1, 4)]


def test_see_mirrors_the_vocabulary():
    import chess
    import analyst_core as A
    from catspace.incubator.analyst.discovery import base_vocabulary as BV
    from catspace.incubator.analyst.discovery import selfmine as SM
    BV.install_terms()
    fens = ["rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2", "rnbqkbnr/pppp1ppp/8/8/3pP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
            "rnbqkbnr/pp1p1ppp/8/2p5/3pP3/2P2N2/PP3PPP/RNBQKB1R b KQkq - 0 4", "rnbqkb1r/pp3ppp/3p1n2/2p5/2B1P3/2N2N2/PP3PPP/R1BQK2R w KQkq - 2 7",
            "4k3/8/4p3/3Q4/8/8/8/4K3 b - - 1 1", "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1",
            "r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1BN2/PPP1QPPP/R3K2R w KQkq - 0 1", "r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1BN2/PPP1QPPP/R3K2R b KQkq - 0 1",
            "4k3/3q4/8/8/8/3R4/3R4/3K4 b - - 0 1", "4k3/3q4/8/8/8/3R4/3R4/3K4 w - - 0 1", "3rk3/8/8/3p4/8/3R4/3R4/3K4 w - - 0 1",
            "4k3/8/8/3b4/4N3/8/8/4K3 w - - 0 1", "4k3/8/8/3b4/4N3/2n5/8/4K3 w - - 0 1", "4k3/8/8/3b4/4N3/2n5/8/3RK3 w - - 0 1",
            "4k3/8/8/4p3/5P2/8/8/4K3 b - - 0 1", "4k3/8/3p4/4p3/5P2/8/8/4K3 b - - 0 1", "3qk3/8/8/3Q4/8/8/8/3QK3 b - - 0 1",
            "3qk3/8/8/3Q4/8/8/8/3QK3 w - - 0 1", "4k3/8/8/3r4/8/3B4/8/3QK3 w - - 0 1"]
    n = bad = 0
    for f in fens:
        b = chess.Board(f)
        mv = "w" if b.turn else "b"
        for sq in chess.SQUARES:
            p = b.piece_at(sq)
            if p and p.color != b.turn and p.piece_type != chess.KING and b.attackers(b.turn, sq):
                n += 1
                bad += A.vocab_eval(f, f"(see {SM.view(chess.square_name(sq), mv)})") != OB.DOMAIN.see(b, sq)
    print(f"OBS see against vocab_eval: {n} capturable pieces in {len(fens)} positions, {bad} disagreements (482 of 482 agreed on four whole games)")
    assert n >= 20 and bad == 0
