"""mates.py -- MATE PATTERNS named from L1 probes over the mated position: which piece checks,
what covers the king's flight squares, whether the king is boxed by its own men. Objective-free
(a checkmate is a rules fact); used by L6 to say "back-rank mate: the rook checks along the 8th
and the pawns block the king" instead of "best Rd8#"."""
from __future__ import annotations

import chess

from . import rules as R

NAME = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}


def _pc(board: chess.Board, sq: int) -> str:
    p = board.piece_at(sq)
    return f"the {'white' if p.color else 'black'} {NAME[p.piece_type]} on {chess.square_name(sq)}"


def mate_pattern(board: chess.Board) -> tuple[str, str] | None:
    """(pattern, reason) for a board where the side to move is checkmated; None otherwise."""
    if not board.is_checkmate():
        return None
    c = board.turn
    k = R.royal_square(board, c)
    if k is None:
        return None
    checkers = list(board.checkers())
    ring = [s for s in chess.SQUARES if chess.square_distance(s, k) == 1]
    own_blocked = [s for s in ring if (p := board.piece_at(s)) and p.color == c]
    covered = [s for s in ring if s not in own_blocked and board.attackers(not c, s)]
    coverers = sorted({a for s in covered for a in board.attackers(not c, s)} - set(checkers))
    chk = checkers[0]
    ctype = board.piece_at(chk).piece_type
    back_rank = 0 if c == chess.WHITE else 7
    last = board.peek() if board.move_stack else None
    discovered = last is not None and last.to_square != chk and len(checkers) >= 1 and all(x != last.to_square for x in checkers)
    if len(checkers) == 2:
        return "double-check mate", f"two pieces check at once ({_pc(board, checkers[0])} and {_pc(board, checkers[1])}); the king cannot capture or block both"
    if ctype == chess.KNIGHT and len(own_blocked) == len(ring):
        return "smothered mate", f"the king is boxed in by its own pieces and {_pc(board, chk)} checks"
    if chess.square_rank(k) == back_rank and ctype in (chess.ROOK, chess.QUEEN) and chess.square_rank(chk) == back_rank:
        blockers = [s for s in own_blocked if chess.square_rank(s) != back_rank]
        why = f"{_pc(board, chk)} checks along the back rank"
        if blockers:
            why += "; the king's own " + ", ".join(NAME[board.piece_at(s).piece_type] + " on " + chess.square_name(s) for s in blockers) + (" block" if len(blockers) > 1 else " blocks") + " its escape"
        if coverers:
            why += "; " + ", ".join(_pc(board, a) for a in coverers[:2]) + (" cover" if len(coverers[:2]) > 1 else " covers") + " the rest"
        return "back-rank mate", why
    if discovered:
        return "discovered-check mate", f"{_pc(board, last.to_square)} moved away and {_pc(board, chk)} delivers the check"
    if ctype == chess.QUEEN and chess.square_distance(chk, k) == 1:
        guards = [a for a in board.attackers(not c, chk)]
        return "supported queen mate", f"the queen mates in contact on {chess.square_name(chk)}" + (f", protected by {_pc(board, guards[0])}" if guards else "")
    edge = chess.square_rank(k) in (0, 7) or chess.square_file(k) in (0, 7)
    if edge and ctype in (chess.ROOK, chess.QUEEN):
        return "edge mate", f"the king is cut off on the edge; {_pc(board, chk)} checks and " + (", ".join(_pc(board, a) for a in coverers[:2]) + " cover the flight squares" if coverers else "the flight squares are covered")
    return f"{NAME[ctype]} mate", f"{_pc(board, chk)} checks; flight squares " + ", ".join(chess.square_name(s) for s in covered) + (" are covered by " + ", ".join(_pc(board, a) for a in coverers[:2]) if coverers else "") + (("; " + ", ".join(chess.square_name(s) for s in own_blocked) + " blocked by own pieces") if own_blocked else "")


def mate_in_line(board: chess.Board, pv) -> tuple[int, str, str] | None:
    """Replay a line; if it ends in mate, (plies_to_mate, pattern, reason)."""
    t = board.copy(stack=False)
    for i, u in enumerate(pv, 1):
        m = chess.Move.from_uci(u) if isinstance(u, str) else u
        if m not in t.legal_moves:
            return None
        t.push(m)
        if t.is_checkmate():
            pat = mate_pattern(t)
            return (i, *pat) if pat else None
    return None
