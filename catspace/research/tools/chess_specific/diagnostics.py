"""catspace/diagnostics.py -- exact concept computations and field-health instruments.

DIAGNOSTIC ONLY (DECISIONS.md sec 4, Kaveh 2026-07-22): everything in this module is an
instrument or a data-generation label -- NEVER a play-time value or policy input. It
measures whether the LEARNED system has a property; it does not substitute for it.

Concepts:  escape_volume (cornering; constrain(king) -- generalize to any piece),
           mate_pattern / mate_labels (classic human mate-pattern classifier),
           material_count.
Instruments: eff_rank (collapse gate), field d_pairs helpers live on FieldModel
           (catspace.fields) since they need a loaded field.
"""
from __future__ import annotations

from collections import deque

import chess
import numpy as np


# --------------------------------------------------------------------- concepts
def escape_volume(b: chess.Board, color: chess.Color = chess.BLACK) -> int:
    """How many squares `color`'s king can still reach (flood-fill), blocked by
    occupancy and opponent-attacked squares. The 'box'; shrinks toward mate."""
    k = b.king(color)
    if k is None:
        return 0
    opp = not color
    seen = {k}; dq = deque([k])
    while dq:
        s = dq.popleft()
        for nb in chess.SquareSet(chess.BB_KING_ATTACKS[s]):
            if nb in seen or b.piece_at(nb) is not None:
                continue
            if b.is_attacked_by(opp, nb):
                continue
            seen.add(nb); dq.append(nb)
    return len(seen)


def material_count(b: chess.Board, color: chess.Color) -> int:
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    return sum(vals.get(p.piece_type, 0) for p in b.piece_map().values() if p.color == color)


def mate_pattern(b: chess.Board) -> str:
    """Classic human mate patterns for a Black-mated board (rules only, audit-clean):
    double / smothered / backrank / ladder / qkiss / ksupport / <piece>-other."""
    bk = b.king(chess.BLACK); wk = b.king(chess.WHITE)
    checkers = list(b.checkers())
    if len(checkers) > 1:
        return "double"
    chk = checkers[0]; ct = b.piece_type_at(chk)
    f, r = chess.square_file(bk), chess.square_rank(bk)
    adj = list(chess.SquareSet(chess.BB_KING_ATTACKS[bk]))
    own_blocked = [s for s in adj if (p := b.piece_at(s)) is not None and p.color == chess.BLACK]
    if ct == chess.KNIGHT and len(own_blocked) == len(adj):
        return "smothered"
    if ct in (chess.ROOK, chess.QUEEN) and r == 7 and chess.square_rank(chk) == 7:
        front = [s for s in adj if chess.square_rank(s) == 6]
        if front and all(b.piece_at(s) is not None and b.piece_at(s).color == chess.BLACK for s in front):
            return "backrank"
    if ct in (chess.ROOK, chess.QUEEN) and (r in (0, 7) or f in (0, 7)):
        heavies = [s for s in list(b.pieces(chess.ROOK, chess.WHITE)) + list(b.pieces(chess.QUEEN, chess.WHITE))
                   if s != chk]
        if r in (0, 7) and any(chess.square_rank(s) == (1 if r == 0 else 6) for s in heavies):
            return "ladder"
        if f in (0, 7) and any(chess.square_file(s) == (1 if f == 0 else 6) for s in heavies):
            return "ladder"
    if ct == chess.QUEEN and chess.square_distance(chk, bk) == 1:
        return "qkiss"
    if ct in (chess.ROOK, chess.QUEEN) and chess.square_distance(wk, bk) <= 2:
        return "ksupport"
    return {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
            chess.ROOK: "rook-other", chess.QUEEN: "queen-other"}[ct]


def mate_labels(b: chess.Board):
    """(material, pattern, king_loc) for a checkmate board (Black mated)."""
    mat = "".join(sorted(p.symbol() for p in b.piece_map().values()))
    bk = b.king(chess.BLACK)
    f, r = chess.square_file(bk), chess.square_rank(bk)
    corner = (f in (0, 7)) and (r in (0, 7))
    loc = "corner" if corner else ("edge" if (f in (0, 7) or r in (0, 7)) else "center")
    return mat, mate_pattern(b), loc


# ------------------------------------------------------------------ instruments
def eff_rank(X: np.ndarray) -> float:
    """Entropy-of-singular-values effective rank -- the standard collapse gate
    (check_representational_collapse). Run on every field's embeddings."""
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))
