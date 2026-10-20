"""Plan 54 (B5.3): the python-chess spread. Every fast path gives what the plain one gave -- graph.mates (a push only when the move could give check), graph.with_piece
(bitboard methods), regress._moves (the per-root cache), facts.board_piece (the per-square memo), reasoned._threat_moves (the exchange after every quiet move in one Rust call)."""
import random

import chess
import pytest

import analyst_core as A
from catspace.incubator.analyst.discovery import facts as FS
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import reasoned as RS
from catspace.incubator.analyst.discovery import regress as R

FENS = [
    "2r2rk1/1p4bp/6p1/pP1qpp2/6PP/P1n1BQ2/2B2R2/5RK1 w - - 1 29",
    "2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 6 62",
    "r3k2r/1P4P1/8/3pP3/8/8/1p4p1/R3K2R w KQkq d6 0 7",
    "7k/5qp1/4R2p/p4Q2/1p6/2P3R1/PP4PP/5K2 b - - 2 33",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "5rk1/5ppp/8/8/8/8/1Q3PPP/6K1 w - - 0 1",
    "rnbqkbnr/ppppp2p/8/5pp1/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq g6 0 3",
]


def playouts(n_games=60, plies=70, seed=7):
    rnd = random.Random(seed)
    out = []
    for _ in range(n_games):
        b = chess.Board()
        for _ in range(plies):
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(rnd.choice(ms))
            out.append(b.copy(stack=False))
    return out


POSITIONS = [chess.Board(f) for f in FENS] + playouts()


def test_mates_equals_push_and_is_checkmate():
    n_mate = n_check = 0
    for b in POSITIONS:
        for m in list(b.legal_moves):
            b.push(m)
            exp = b.is_checkmate()
            n_check += b.is_check()
            b.pop()
            before = b.fen()
            assert G.mates(b, m) is exp, (b.fen(), m)
            assert b.fen() == before
            n_mate += exp
    print(f"OBS mates: {len(POSITIONS)} positions, {n_check} checking moves, {n_mate} mating moves, all equal")
    assert n_check > 100


def test_mates_on_known_mates():
    for fen, uci in (("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"), ("7k/6pp/8/8/8/8/8/5RK1 w - - 0 1", "f1f8"), ("k7/8/1K6/8/8/8/8/7R w - - 0 1", "h1h8"),
                     ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7")):
        assert G.mates(chess.Board(fen), chess.Move.from_uci(uci))


def test_with_piece_fast_equals_plain():
    rnd = random.Random(3)
    for b in POSITIONS[:80]:
        for _ in range(6):
            at, frm = rnd.randrange(64), rnd.randrange(64)
            pc = chess.Piece(rnd.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]), rnd.choice([True, False]))
            for rf in (None, frm):
                G.FAST_PIECE = True
                x = G.with_piece(b, pc, at, rf)
                G.FAST_PIECE = False
                y = G.with_piece(b, pc, at, rf)
                G.FAST_PIECE = True
                assert (x.fen(), x.promoted, x.pawns, x.occupied_co[True], x.occupied_co[False]) == (y.fen(), y.promoted, y.pawns, y.occupied_co[True], y.occupied_co[False])


def test_moves_cache_equals_plain():
    G.clear_caches()
    try:
        for b in POSITIONS[:120]:
            for us in ("w", "b", True, False):
                for skip in (None, chess.E1, chess.A2):
                    R.MOVES_CACHE = False
                    plain = R._moves(b, us, skip)
                    R.MOVES_CACHE = True
                    assert R._moves(b, us, skip) == plain
                    assert R._moves(b, us, skip) == plain                   # the second answer is the cached one
    finally:
        R.MOVES_CACHE = True
        G.clear_caches()


def test_board_piece_memo_equals_plain():
    for b in POSITIONS[:60]:
        FS.PIECE_MEMO = True
        f = FS.board_piece(b)
        FS.PIECE_MEMO = False
        g = FS.board_piece(b)
        FS.PIECE_MEMO = True
        assert [f(s) for s in range(64)] == [g(s) for s in range(64)] == [f(s) for s in range(64)]


def test_threat_moves_rust_equals_python_and_see_after_equals_see():
    n = 0
    try:
        for b in POSITIONS:
            us = "w" if b.turn else "b"
            hang = RS.hanging_of_ours(b, us)
            if not hang:
                continue
            RS.THREAT_RUST = False
            old = RS._threat_moves(b, us, hang)
            RS.THREAT_RUST = True
            assert RS._threat_moves(b, us, hang) == old, b.fen()
            big = max(hang, key=lambda s: RS._worth(b, s))
            ms = list(b.legal_moves)
            got = A.see_after(b.fen(), [m.uci() for m in ms], big)
            for m, g in zip(ms, got):
                c = b.copy()
                c.push(m)
                assert g == RS._see(c.fen(), big, us), (b.fen(), m)
            n += 1
    finally:
        RS.THREAT_RUST = True
    print(f"OBS threat_moves: {n} positions with a hanging piece, Rust and Python paths equal")
    assert n >= 10
