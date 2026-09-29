"""Can the piece make it there (catspace/incubator/analyst/piece_reach.py): per-piece reach, the goal frame, footprints."""
import random

import chess

import catspace.incubator.analyst.piece_reach as PR


def test_the_rule_of_the_square_falls_out():
    b = chess.Board("8/8/8/P7/8/5k2/8/K7 w - - 0 1")                  # the king on f3 is outside the pawn's square
    assert PR.goal_frame(b, chess.A5, jobs=False).unopposed
    b = chess.Board("8/8/8/P7/2k5/8/8/K7 b - - 0 1")                  # on c4 with the move it catches the pawn
    g = PR.goal_frame(b, chess.A5, jobs=False)
    assert not g.unopposed and g.members[0].square == chess.C4


def test_a_discovered_check_makes_the_king_a_member():
    # the black king steps off the a5-e1 diagonal: the bishop checks, White loses a move, and the pawn is stopped
    b = chess.Board("8/8/6P1/b7/8/2k5/8/4K3 b - - 0 1")
    g = PR.goal_frame(b, chess.G6, jobs=False)
    assert any(m.square == chess.C3 and "uncovers" in m.how for m in g.members)


def test_the_rook_can_stop_the_pawn_but_leaves_its_job():
    # 2ckKj: after c7 the rook on d5 can reach the c-pawn's path in time (Rd8), but it then gives up its threat on b5
    b = chess.Board("8/p7/2P5/1P1rk3/R7/6p1/8/6K1 w - - 3 43")
    g = PR.goal_frame(b, chess.C6)
    m = next(m for m in g.members if m.square == chess.D5)
    assert m.moves <= m.allowed and m.job is not None and m.job > 0
    assert {s for s, _ in g.outside} == {chess.A7, chess.E5, chess.G3}


def test_a_reach_is_served_until_a_square_it_read_changes():
    PR.clear_store()
    b = chess.Board("8/p7/2P5/1P1rk3/R7/6p1/8/6K1 w - - 3 43")
    r0 = PR.piece_reach(b, chess.D5, k=1)                             # its rays from d5 and from every square one move away
    assert chess.A7 in r0.footprint and chess.D3 in r0.footprint      # (from d7 it scans the 7th rank to a7)
    outside = next(s for s in chess.SQUARES if s not in r0.footprint and b.piece_at(s) is None)
    far = b.copy(); far.set_piece_at(outside, chess.Piece(chess.KNIGHT, chess.WHITE))
    before = dict(PR.STATS)
    PR.piece_reach(far, chess.D5, k=1)                                # a piece on a square it never read: served
    assert PR.STATS["served"] == before["served"] + 1
    near = b.copy(); near.set_piece_at(chess.D3, chess.Piece(chess.KNIGHT, chess.WHITE))
    PR.piece_reach(near, chess.D5, k=1)                               # d3 is on the rook's file: recomputed
    assert PR.STATS["computed"] == before["computed"] + 1


def test_unopposed_never_lies_and_verify_mode_finds_no_leak():
    rng = random.Random(7)
    old = PR.VERIFY
    PR.VERIFY = True
    try:
        PR.clear_store()
        checked = 0
        while checked < 150:
            b = chess.Board(None)
            ks = rng.sample(range(64), 3)
            ps = chess.square(rng.randrange(8), rng.randrange(4, 7))
            if ps in ks:
                continue
            b.set_piece_at(ks[0], chess.Piece(chess.KING, chess.WHITE)); b.set_piece_at(ks[1], chess.Piece(chess.KING, chess.BLACK))
            b.set_piece_at(ks[2], chess.Piece(rng.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK]), chess.BLACK))
            b.set_piece_at(ps, chess.Piece(chess.PAWN, chess.WHITE))
            b.turn = rng.choice([chess.WHITE, chess.BLACK])
            if not b.is_valid() or b.is_game_over():
                continue
            g = PR.goal_frame(b, ps, jobs=False)
            if g.unopposed:
                assert PR._gets_through(b, ps, chess.WHITE, max(t for _, t in g.path), {}), b.fen()
            checked += 1
    finally:
        PR.VERIFY = old
