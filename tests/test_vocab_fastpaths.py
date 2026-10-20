"""Plan 54 (B5): the evaluator's fast paths read what the plain definitions read -- a `least` / `most` over a body of the material alone (no board built per move),
a constant body, the zero test of a sum (cheapest summand first), and `(and A hit)` (the scan only inside A) -- against python-chess as the independent reference."""
import chess
import pytest

import analyst_core as A
from catspace.incubator.analyst.discovery import base_vocabulary as BV

FENS = [
    "2r2rk1/1p4bp/6p1/pP1qpp2/6PP/P1n1BQ2/2B2R2/5RK1 w - - 1 29",
    "2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 6 62",                           # a pawn on the seventh: promotions (queen and under)
    "r3k2r/1P4P1/8/3pP3/8/8/1p4p1/R3K2R w KQkq d6 0 1",               # castling, en passant, promotions for both sides
    "7k/5qp1/4R2p/p4Q2/1p6/2P3R1/PP4PP/5K2 b - - 2 33",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1",                              # mated: no legal move
]
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def mat(b, color):
    return sum(VAL[p.piece_type] for p in b.piece_map().values() if p.color == color)


def balance(b, us):
    return max(mat(b, us), 0) + 64 - min(max(mat(b, not us), 0), max(mat(b, us), 0) + 64)


def reference(b, us, depth, leaf):
    """least / most alternately (least first) `depth` times, then the leaf on the board; no legal move reads 0."""
    if depth == 0:
        return leaf(b, us)
    vals = []
    for m in list(b.legal_moves):
        b.push(m)
        vals.append(reference(b, us, depth - 1, leaf))
        b.pop()
    if not vals:
        return 0
    return min(vals) if depth % 2 == 1 else max(vals)


@pytest.mark.parametrize("fen", FENS)
def test_material_only_bodies(fen):
    BV.install_terms()
    b = chess.Board(fen)
    us = b.turn
    # (least (most (least (min balance (sum material_us 4))))): the order the reference nests is least, most, least from the outside in
    leaf = lambda bb, u: min(balance(bb, u), mat(bb, u) + 4)
    got = A.vocab_eval(fen, "(least (most (least (min balance (sum material_us 4)))))")
    # the reference's outermost is `least` when depth is odd: depth 3 -> least, most, least
    assert got == reference(b, us, 3, leaf), fen
    got2 = A.vocab_eval(fen, "(most (least (max material_them (diff balance 70))))")
    leaf2 = lambda bb, u: max(mat(bb, not u), abs(balance(bb, u) - 70))
    # depth 2 -> outermost `most`, then `least`
    assert got2 == reference_mostleast(b, us, leaf2), fen


def reference_mostleast(b, us, leaf):
    vals = []
    for m in list(b.legal_moves):
        b.push(m)
        inner = [leaf_after(b, us, leaf, mm) for mm in list(b.legal_moves)]
        vals.append(min(inner) if inner else 0)
        b.pop()
    return max(vals) if vals else 0


def leaf_after(b, us, leaf, m):
    b.push(m)
    v = leaf(b, us)
    b.pop()
    return v


@pytest.mark.parametrize("fen", FENS)
def test_constant_body_and_zero_sum_and_masked_hit(fen):
    BV.install_terms()
    b = chess.Board(fen)
    us = b.turn
    has = any(b.legal_moves)
    assert A.vocab_eval(fen, "(least 1)") == (1 if has else 0) and A.vocab_eval(fen, "(most 7)") == (7 if has else 0)
    # (count (and ours hit)): our pieces (the king too, as `ours` is) that they attack, our king lifted
    c = b.copy()
    c.remove_piece_at(c.king(us))
    n_hit = sum(1 for s, p in b.piece_map().items() if p.color == us and c.is_attacked_by(not us, s))
    assert A.vocab_eval(fen, "(count (and ours hit))") == n_hit
    assert A.vocab_eval(fen, "(count (and hit ours))") == n_hit
    promo = A.vocab_eval(fen, "(count (and theirpawns (rank a2)))")
    total = n_hit + promo + (1 if has else 0)
    zero = A.vocab_eval(fen, "(eq (sum (sum (count (and ours hit)) (count (and theirpawns (rank a2)))) (least 1)) 0)")
    assert zero == (1 if total == 0 else 0)
