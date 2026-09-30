"""The standing king frame (catspace/incubator/analyst/king_watch.py): members, conditions, operators, no horizon."""
import chess

import catspace.incubator.analyst.king_watch as KW
from catspace.incubator.analyst.piece_reach import clear_store


def test_the_back_rank_guard_is_a_condition_with_a_deflection():
    # Qxc8+ Bxc8 Rd8#: the rook on c8 guards d8 against Rd1 and e8 against Qc6 -- one piece, two jobs
    KW.clear_store(); clear_store()
    b = chess.Board("2r2k1r/2p1ppbp/2Q1bqpn/p7/8/2P1BPN1/PP4PP/3R1RK1 w - - 1 18")
    w = KW.watch(b, chess.BLACK)
    open1 = {m.square for m in w.members if m.kind == "open" and m.moves == 1}
    assert {chess.C6, chess.D1} <= open1
    guard = [c for c in w.conditions if c.kind == "guards" and c.square == chess.C8 and c.member == chess.D1]
    assert guard and any(op[0] == "deflect" for op in guard[0].operators)


def test_a_rook_behind_three_pieces_is_a_latent_member():
    # Kaveh: "a king and the opponent's rook on the same file, a few pieces in between ... all three have to go"
    KW.clear_store(); clear_store()
    b = chess.Board("4r1k1/8/4p3/4n3/8/4B3/8/4K3 w - - 0 1")
    w = KW.watch(b, chess.WHITE)
    m = next(m for m in w.members if m.square == chess.E8 and m.kind == "latent")   # (it may also check by an open route)
    assert sorted(m.blockers) == sorted([chess.E6, chess.E5, chess.E3])
    assert {c.square for c in w.conditions if c.kind == "blocks" and c.member == chess.E8} == {chess.E6, chess.E5, chess.E3}


def test_no_horizon_a_far_knight_is_a_member_with_its_distance():
    KW.clear_store(); clear_store()
    b = chess.Board("7k/8/8/8/8/8/8/N6K w - - 0 1")
    w = KW.watch(b, chess.BLACK)
    m = next(m for m in w.members if m.square == chess.A1)
    assert m.kind == "open" and m.moves >= 4


def test_served_watches_equal_recomputed_ones_along_a_game():
    old = KW.VERIFY
    KW.VERIFY = True
    try:
        KW.clear_store(); clear_store()
        b = chess.Board()
        for u in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5", "c2c3", "d7d6", "e1g1", "e8g8"]:
            b.push_uci(u)
            KW.watch(b, chess.WHITE); KW.watch(b, chess.BLACK)
            KW.watch(b, chess.WHITE); KW.watch(b, chess.BLACK)     # served, and verified against a fresh computation
    finally:
        KW.VERIFY = old
