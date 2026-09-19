"""PIECE IDENTITY ACROSS PLIES (Kaveh 2026-09-18: "my bishop tends to be inactive in this line; their kingside
knight active" -- an action concept names a PIECE, so the piece must be followed through the game).

Pure board mechanics: a piece id is (origin square, type at origin); a move carries the id from its from-square
to its to-square; a capture ends the captured id; castling carries the rook; en passant ends the pawn behind;
promotion keeps the id and changes the type. A piece's FATE within h plies of ply t: kept (still on the board),
exchanged (it captured something and was then captured, or was captured after capturing, within the window),
lost (captured without capturing), promoted. Attributes for the readout: the square colour and the wing, which
are board coordinates, not rules."""
from __future__ import annotations

from dataclasses import dataclass

import chess

KEPT, EXCHANGED, LOST, PROMOTED = 0, 1, 2, 3
FATE_NAMES = ("kept", "exchanged", "lost", "promoted")


def move_between(b0: chess.Board, b1: chess.Board) -> chess.Move | None:
    """The legal move of b0 that produces b1's placement and turn (None when b1 is not one move away)."""
    target = (b1.board_fen(), b1.turn)
    for m in b0.legal_moves:
        nb = b0.copy(stack=False); nb.push(m)
        if (nb.board_fen(), nb.turn) == target:
            return m
    return None


def boards_from_fens(fens: list[str]) -> tuple[list[chess.Board], list[chess.Move | None]]:
    boards = [chess.Board(f) for f in fens]
    moves = [move_between(boards[k], boards[k + 1]) for k in range(len(boards) - 1)]
    return boards, moves


def track(start: chess.Board, moves: list[chess.Move]) -> list[dict[int, tuple[int, int]]]:
    """ids[k] = {square: (origin, type_at_origin)} at ply k (before moves[k]); len = len(moves) + 1.
    A capture ends the victim's id; the pieces themselves are identified by where they started."""
    ids = {sq: (sq, p.piece_type) for sq, p in start.piece_map().items()}
    out = [dict(ids)]
    b = start.copy(stack=False)
    for m in moves:
        if m is None:
            break
        pid = ids.pop(m.from_square, None)
        if b.is_en_passant(m):
            cap = m.to_square + (-8 if b.turn == chess.WHITE else 8)
            ids.pop(cap, None)
        ids.pop(m.to_square, None)                                   # the victim, if any
        if b.is_castling(m):
            rook_from, rook_to = _castle_rook(m, b.turn)
            rid = ids.pop(rook_from, None)
            if rid is not None:
                ids[rook_to] = rid
        if pid is not None:
            ids[m.to_square] = pid
        b.push(m)
        out.append(dict(ids))
    return out


def _castle_rook(m: chess.Move, side: chess.Color) -> tuple[int, int]:
    rank = 0 if side == chess.WHITE else 7
    if chess.square_file(m.to_square) > chess.square_file(m.from_square):
        return chess.square(7, rank), chess.square(5, rank)
    return chess.square(0, rank), chess.square(3, rank)


@dataclass
class Fate:
    kind: int                                 # KEPT | EXCHANGED | LOST | PROMOTED
    square: int | None                        # where the piece stands at t+h when kept
    other_type: int | None = None             # the type of the piece it was exchanged for / taken by
    at: int | None = None                     # the ply the fate happened


def fates(start: chess.Board, moves: list[chess.Move], t: int, h: int) -> dict[tuple[int, int], Fate]:
    """For every piece alive at ply t, its fate by ply min(t+h, end)."""
    ids = track(start, moves)
    b = start.copy(stack=False)
    for m in moves[:t]:
        b.push(m)
    alive = {pid: sq for sq, pid in ids[t].items()}
    took = {}                                                          # pid -> the type it captured within the window
    out = {}
    end = min(t + h, len(moves), len(ids) - 1)
    bb = b.copy(stack=False)
    for k in range(t, end):
        m = moves[k]
        if m is None:
            break
        mover = ids[k].get(m.from_square)
        victim_sq = m.to_square + (-8 if bb.turn == chess.WHITE else 8) if bb.is_en_passant(m) else m.to_square
        victim = ids[k].get(victim_sq) if (bb.is_capture(m)) else None
        if victim is not None and victim in alive and victim not in out:
            other = bb.piece_at(m.from_square).piece_type if bb.piece_at(m.from_square) else None
            kind = EXCHANGED if victim in took else LOST
            out[victim] = Fate(kind, None, other, k)
        if mover is not None and victim is not None and mover in alive:
            took[mover] = bb.piece_at(victim_sq).piece_type if bb.piece_at(victim_sq) else None
            if mover in out and out[mover].kind == LOST:
                pass
        if m.promotion and mover in alive and mover not in out:
            out[mover] = Fate(PROMOTED, m.to_square, m.promotion, k)
        bb.push(m)
    # a piece that captured and was later captured in the window is an exchange even if its capture came first
    for pid, f in list(out.items()):
        if f.kind == LOST and pid in took:
            f.kind = EXCHANGED
            f.other_type = took[pid] if f.other_type is None else f.other_type
    for pid in alive:
        if pid not in out:
            sq = next((s for s, p in ids[end].items() if p == pid), None)
            out[pid] = Fate(KEPT, sq)
            if pid in took:
                out[pid].other_type = took[pid]                        # kept, having captured: the readout can say "took a knight and stayed"
    return out


def square_colour(sq: int) -> int:
    """1 = light, 0 = dark."""
    return (chess.square_file(sq) + chess.square_rank(sq)) % 2


def wing(sq: int) -> int:
    """0 queenside (a-c), 1 centre (d-e), 2 kingside (f-h)."""
    f = chess.square_file(sq)
    return 0 if f <= 2 else (1 if f <= 4 else 2)


def describe(board: chess.Board, sq: int, side: chess.Color | None = None) -> str:
    """'our light-squared bishop on c1' / 'their kingside knight on f6' -- attributes only where they distinguish."""
    p = board.piece_at(sq)
    if p is None:
        return chess.square_name(sq)
    own = "our" if side is None or p.color == side else "their"
    name = chess.piece_name(p.piece_type)
    attr = ""
    if p.piece_type == chess.BISHOP:
        attr = "light-squared " if square_colour(sq) else "dark-squared "
    elif p.piece_type in (chess.KNIGHT, chess.ROOK):
        attr = ("queenside ", "central ", "kingside ")[wing(sq)]
    return f"{own} {attr}{name} on {chess.square_name(sq)}"
