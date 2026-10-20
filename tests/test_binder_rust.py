"""Plan 54 (A): `analyst_core.verify_route` reads the binder's schemas exactly as the Python `binder._read` / `_read2` did."""
import chess
import pytest

from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import binder as B

FENS = [
    "2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 6 62",                          # 0LRTF, white
    "2r2rk1/1p4bp/6p1/pP1qpp2/6PP/P1n1BQ2/2B2R2/5RK1 w - - 1 29",    # 07Z0P, white, a middlegame with checks and captures
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",                           # a back-rank mate in one
    "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1",                           # the same for black (squares named from black's side)
    "5r1k/6b1/8/4QppB/p7/5P2/P4P1P/6K1 b - - 1 32",                   # black; Bxe5 reads 68 but our material is short at a leaf (the SAN in a note)
    "7k/5qp1/4R2p/p4Q2/1p6/2P3R1/PP4PP/5K2 b - - 2 33",               # black; depth-two notes naming two moves
]


def routes(board):
    seen = []
    for m in board.legal_moves:
        if m.promotion or board.is_castling(m):
            continue
        pr = (chess.square_name(m.from_square), chess.square_name(m.to_square))
        if pr not in seen:
            seen.append(pr)
    return seen


@pytest.mark.parametrize("fen", FENS)
def test_verify_route_equals_read(fen):
    import analyst_core as A
    BV.install_terms()
    board = chess.Board(fen)
    us = "w" if board.turn else "b"
    n = 0
    for pair in routes(board):
        for s in [s for s in B.SCHEMAS]:
            old = B._read(s, fen, us, pair)
            (name, val, note, sec), = A.verify_route(fen, pair[0], pair[1], us, [s.name], [])
            if "\x01" in note:
                note = B._SAN_TOKEN.sub(lambda m: " ".join(B.PA._san_pairs(fen, [(t[:2], t[2:]) for t in m.group(1).split(",")])), note)
            assert (name, val, note) == (s.name, old[0], old[1]), (fen, pair, s.name)
            n += 1
    assert n > 0


@pytest.mark.parametrize("fen", FENS)
def test_verify_tried_rust_equals_python(fen):
    board = chess.Board(fen)
    us = "w" if board.turn else "b"
    out = {}
    try:
        for flag in (False, True):
            B.BINDER_RUST = flag
            res = []
            for m in board.legal_moves:
                bind, tried = B.verify_tried(m, board, us)
                res.append((m.uci(), tried, None if bind is None else (bind.schema, bind.value, bind.san, bind.assignment, bind.note)))
            out[flag] = res
    finally:
        B.BINDER_RUST = True
    assert out[False] == out[True]


def test_alive_notes_carry_the_san():
    """The alive-test failures name the move in SAN (filled in from python-chess); both paths say the same words."""
    fen = "7k/5qp1/4R2p/p4Q2/1p6/2P3R1/PP4PP/5K2 b - - 2 33"
    board = chess.Board(fen)
    B.BINDER_RUST = True
    notes = [n for m in board.legal_moves for _, _, n in B.verify_tried(m, board, "b")[1]]
    assert any("Qxf5+ bxc3+ reads 54 but our material is below 13" in n or "bind delayed_root_max_d2: " in n for n in notes)
    assert not any("\x01" in n for n in notes)


# ------------------------------------------------------------------------------------------------------------------ 54 (B4): second_moves and the verify cache

QUEEN_TRAP = "rnb1kb2/1p2pprp/2p2n2/pB4p1/2p2P1P/PP2P1PN/3qQK2/RNB4R w q - 2 13"


@pytest.mark.parametrize("fen", FENS + [QUEEN_TRAP, "4k3/8/8/8/8/5q2/PP6/RN2K3 w - - 0 1", "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"])
def test_second_moves_rust_equals_python_for_every_route_and_trap_square(fen):
    board = chess.Board(fen)
    old = B.SECOND_RUST
    try:
        for m in board.legal_moves:
            for trap in [None] + list(chess.SQUARES):
                B.SECOND_RUST = False
                a = B.second_moves(board, m, trap)
                B.SECOND_RUST = True
                b = B.second_moves(board, m, trap)
                assert a == b, (fen, m.uci(), trap, a, b)
    finally:
        B.SECOND_RUST = old


def test_verify_cache_answers_the_same_and_is_cleared_at_a_fresh_root():
    from catspace.incubator.analyst.discovery import graph as G
    board = chess.Board(FENS[2])
    m = chess.Move.from_uci("a1a8")
    B.VERIFY_CACHE = False
    a = B.verify_tried(m, board, "w")
    B.VERIFY_CACHE = True
    try:
        G.clear_caches()
        b = B.verify_tried(m, board, "w")
        assert len(B._VERIFIED) == 1
        c = B.verify_tried(m, board, "w")
        assert a == b == c and c[0] is not b[0]            # equal, and a copy
        G.clear_caches()
        assert not B._VERIFIED
    finally:
        B.VERIFY_CACHE = True
