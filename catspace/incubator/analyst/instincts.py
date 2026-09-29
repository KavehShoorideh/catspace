"""FIRST-INSTINCT MOVES (Kaveh 2026-09-28: "there are some kind of obvious frames, one-ply single-piece frames: if my
piece is under attack, my first instinct is to retreat it to a square that is safe, or safe and protected; if you take
my piece that is protected, my first instinct is to take back ... these are single-frame best-move ideas; when other
frames come in it's not obvious that this was the right move"; "if they could take back ... they could make the next
obvious move, one of those first-instinct moves -- and now we would need planning to win").

The instincts of the side to move, each read off one frame, in the order a player reaches for them:

    recapture              take back on the square the opponent just captured on
    take                   capture a piece whose frame the side wins now
    retreat (protected)    move an attacked piece to a square where its frame no longer loses it, and that is defended
    retreat                the same, to a safe square that is not defended
    defend / block         another piece's move after which the attacked piece no longer falls
    escape check           under check, every evasion

The INSTINCTIVE OPPONENT (`frame_player --ledger two_ply_instinct`) replies only with these -- or passes. Kaveh's
definition: what it takes to beat that opponent is the planning layer's job.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

from .fates import passed
from .frame_goals import frame_on

ORDER = {"escape check": 0, "recapture": 1, "take": 2, "retreat (protected)": 3, "retreat": 4, "defend": 5, "block": 5}


@dataclass
class Instinct:
    move: chess.Move
    kind: str
    square: int                     # the piece or square the instinct is about
    text: str


def threatened(b: chess.Board) -> list:
    """The side to move's pieces the opponent's frames win now: [(square, what they win)], most valuable first."""
    pb = passed(b)
    if pb is None:
        return []
    out = []
    for s, pc in b.piece_map().items():
        if pc.color != b.turn or pc.piece_type == chess.KING or not b.attackers(not b.turn, s):
            continue
        g = frame_on(pb, s)[0]
        if g > 0:
            out.append((s, g))
    return sorted(out, key=lambda t: -t[1])


def instincts(b: chess.Board) -> list:
    """The first-instinct moves of the side to move, labelled, in the order a player reaches for them."""
    side = b.turn; out = []; seen = set()
    nm = lambda s: f"the {chess.piece_name(b.piece_at(s).piece_type)} on {chess.square_name(s)}"

    def add(m, kind, s, text):
        if m not in seen:
            seen.add(m); out.append(Instinct(m, kind, s, text))

    if b.is_check():
        for m in b.legal_moves:
            add(m, "escape check", b.king(side), f"{b.san(m)} escapes the check")
        return out
    last = b.peek() if b.move_stack else None
    if last is not None and b.piece_at(last.to_square) is not None and b.piece_at(last.to_square).color != side:
        for m in b.generate_legal_captures(to_mask=chess.BB_SQUARES[last.to_square]):
            add(m, "recapture", last.to_square, f"{b.san(m)} takes back on {chess.square_name(last.to_square)}")
    for s, pc in b.piece_map().items():                              # a piece the side's frames win now
        if pc.color == side or pc.piece_type == chess.KING or not b.attackers(side, s):
            continue
        if frame_on(b, s)[0] > 0:
            for m in b.generate_legal_captures(to_mask=chess.BB_SQUARES[s]):
                add(m, "take", s, f"{b.san(m)} takes {nm(s)}")
    for s, g in threatened(b):                                       # a piece of ours that falls: save it
        for m in b.generate_legal_moves(from_mask=chess.BB_SQUARES[s]):
            nb = b.copy(stack=False); nb.push(m)
            if nb.is_check() or frame_on(nb, m.to_square)[0] <= 0:
                prot = bool(nb.attackers(side, m.to_square))
                add(m, "retreat (protected)" if prot else "retreat", s,
                    f"{b.san(m)} takes {nm(s)} to {'a protected' if prot else 'a safe'} square")
        for m in b.legal_moves:
            if m.from_square == s or m in seen:
                continue
            if b.gives_check(m):
                continue                                          # a check buys a tempo; it is tactics, not an instinct
            nb = b.copy(stack=False); nb.push(m)
            if nb.piece_at(s) is not None and frame_on(nb, s)[0] <= 0:
                att = [a for a in b.attackers(not side, s)]
                kind = "block" if any(m.to_square in chess.SquareSet(chess.between(a, s)) for a in att) else "defend"
                add(m, kind, s, f"{b.san(m)} {'blocks the attack on' if kind == 'block' else 'defends'} {nm(s)}")
    out.sort(key=lambda i: ORDER.get(i.kind, 9))
    return out
