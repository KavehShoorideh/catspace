"""Frames read off exchanges (Kaveh 2026-09-27): the exchange frame's outcome, the pass at each ply, the ledger across
frames, the square conditions, and the exchange labeler."""
import chess

from catspace.incubator.analyst import chess_domain as cd
from catspace.incubator.analyst import fates as F

D4 = "r1bq1rk1/pp3p1p/2n1pbp1/2pp4/2B1P3/2PPB3/PP1Q1PPP/RN2K2R w KQ - 0 11"   # the error that started it: d4 is attacked three times and defended three times


def test_the_exchange_frame_counts_the_defenders():
    b = chess.Board(D4); m = chess.Move.from_uci("d3d4")
    full = lambda bb, s: F.direct_frame(bb, s)
    assert F.exchange_move_value(b, m, full)[0] == 0.0                                   # defended: the exchange does not start
    small = lambda bb, s: {s, chess.C5}                                                   # the defenders left out
    assert F.exchange_move_value(b, m, small)[0] == -1.0


def test_known_cases_and_the_ledger():
    cases = [("8/2k1r2p/1p1R2p1/p1pKP3/3p4/6B1/1P4PP/8 w - - 8 33", "d6d8", -5.0),        # the rook walks into the king, undefended
             ("4k3/8/8/4p3/3Q4/8/8/4K3 w - - 0 1", "d4a1", 9.0),                           # the queen leaves the pawn's attack: worth what standing still loses
             ("4k3/8/4p3/8/8/8/8/3QK3 w - - 0 1", "d1d5", -9.0)]                           # Qd5?? hangs to e6
    for fen, u, want in cases:
        b = chess.Board(fen); m = chess.Move.from_uci(u)
        assert F.exchange_move_value(b, m, F.direct_frame)[0] == want, u
        assert F.ledger_move_value(b, m, F.direct_frame)[0] == want, u
    b = chess.Board("r1bQr1k1/pp3pbp/2p2np1/4n3/4N3/2P2NP1/PP3PBP/R1B1R1K1 b - - 0 12")    # the white queen is en prise on d8
    assert F.ledger_move_value(b, chess.Move.from_uci("e8d8"), F.direct_frame)[0] >= 9.0   # the queen, and more: standing still allowed Nxf6+ Bxf6 Qxe8+ (forcing lines, 2026-09-28)
    from catspace.incubator.analyst.frame_player import choose
    assert "e8d8" in choose(b)["best"]


def test_the_pass_at_each_ply_has_a_stake():
    b = chess.Board("r2qk3/8/8/3b4/8/8/3R4/3QK3 w - - 0 1")                              # the rook takes a bishop the queen defends, our queen behind
    v = F.exchange_variations(b, chess.D5, F.direct_frame(b, chess.D5) | {chess.D1}, chess.WHITE)
    assert v[0]["mover"] == "us" and v[0]["stake"] == 3.0                                   # leaving the frame at once costs us the bishop
    assert v[1]["mover"] == "them" and v[1]["stake"] == 0.0                                 # taking back would cost them: they stop


def test_square_conditions():
    b = chess.Board(D4); b.push(chess.Move.from_uci("d3d4")); st = cd.State(b, [chess.D4])
    assert not cd.attacked_by_lower(st, chess.D4) and not cd.outnumbered(st, chess.D4)
    b2 = chess.Board("4k3/8/4p3/3Q4/8/8/8/4K3 w - - 0 1"); st2 = cd.State(b2, [chess.D5])
    assert cd.attacked_by_lower(st2, chess.D5) and cd.outnumbered(st2, chess.D5) and cd.last_attacker(st2, chess.E6, chess.D5)


def test_exchanges_are_chains_on_one_square():
    from catspace.incubator.analyst.exchanges import exchanges
    b = chess.Board(); moves = ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"]
    meta = []
    for ply, u in enumerate(moves):
        meta.append(dict(game=7, ply=ply, fen=b.fen(), played=u)); b.push(chess.Move.from_uci(u))
    ex = exchanges(meta)
    assert len(ex) == 1 and ex[0].squares == [chess.D5] and len(ex[0].captures) == 2
    assert {"d5", "e4", "d8"} <= set(ex[0].pieces)                                        # the pawn taken, the pawn that took, the queen that took back


def test_generation_computes_relations_the_core_does_not_know():
    """The compiled core answers False for a relation it does not implement; the generator must compute those in
    Python, not cache an all-false column (the square conditions of 2026-09-27)."""
    from catspace.incubator.analyst import lemma_gen as lg
    dom = cd.domain(oracle="frame", depth=0)
    if dom.batch_literals is None:
        return                                                                               # no compiled core here: nothing to guard
    g = lg.Generator(dom, cap=60, log=lambda *a, **k: None, sequential=False)
    rows, _ = g.rows_of(lg.Frame(("wQ", "bP"), (("attacks", (1, 0), True),)), 0)
    assert rows
    cache = {}
    for name in ("outnumbered", "attacked_by_lower"):
        want = sum(1 << i for i, (st, ids, a, _) in enumerate(rows) if g.rel[name].fn(st, ids[0]))
        assert g.lit_mask(rows, name, (0,), cache) == want, name
    assert any(g.lit_mask(rows, "attacked_by_lower", (0,), cache) for _ in [0])                # the queen attacked by a pawn: some rows true


def test_the_frames_see_mates_and_forcing_lines():
    """Kaveh 2026-09-28, from the frame player's worst misses: a side that can neither act nor pass has lost, and the
    moves that take the pass away join the frame's line. Each old choice allowed a mate; the frames now refuse it."""
    from catspace.incubator.analyst.frame_player import choose
    cases = [("2r1r1k1/pp3pp1/q7/8/3Qbb2/4N2P/PPP1RP2/2K3R1 b - - 7 22", "Qxe2"),     # allowed Qxg7#
             ("5r1k/p1pp3p/1pP1p3/6p1/P3P2q/2N5/1PPB2KP/R2Q4 w - - 1 21", "cxd7"),     # allowed Rf2+ Kh1 Qxh2#
             ("8/5pk1/p6p/5B2/2np1p2/8/PP2rPPP/3R2K1 w - - 0 29", "Rxd4")]              # allowed Re1#
    for fen, old in cases:
        b = chess.Board(fen); c = choose(b); u = b.parse_san(old).uci()
        assert u not in c["best"], old
        for pick in c["best"]:                                                        # whatever the frames choose, it allows no mate in one
            nb = b.copy(); nb.push(chess.Move.from_uci(pick))
            assert not any((lambda x: (x.push(r), x.is_checkmate())[1])(nb.copy()) for r in nb.legal_moves), (old, pick)
