"""HYPOTHETICAL WHAT-IFS (Kaveh 2026-09-28: "what if there was a knight here? what if there was a pawn here? what if
there was another attacker here? what if we removed this attacker?"). A what-if is an EDIT of the board -- not a move --
asked of the frames: the edited board's frame on the target, less the real one. The edits, for the side to move:

    add(t, v)       one more attacker of value v (1, 3, 5, 9) bearing on the enemy piece on t
    remove(t, d)    the enemy defender on d gone from the frame of t (a pin, a deflection, a block, a capture all do it)
    relocate(p, q)  the own piece on p standing on q (two quiet moves away at most)
    guard(t, v)     the defensive mirror: one more enemy attacker of value v on the own piece on t (what we must pre-empt)

A move REALIZES an edit when it starts to make it true: a piece worth at most v comes to bear on t (or takes the
first step of a two-move route there); after it the defender on d no longer counts for t; it is the relocated piece's
first step toward q; it adds a counted defender to t or moves the piece on t (guard). The edits answer the questions;
which to ask, and in what order, is learned (`whatif_priority`).
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

from .frame_goals import counted_defenders, frame_on
from .frame_placement import placements, roles, _moved, _turned
from .reach import reach

TYPE_OF = {1: chess.PAWN, 3: chess.KNIGHT, 5: chess.ROOK, 9: chess.QUEEN}
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}


@dataclass
class Edit:
    kind: str                       # "add" | "remove" | "relocate" | "guard"
    target: int                     # the square the edit is about (target piece, or the relocated piece's square)
    arg: int                        # add/guard: the attacker's value; remove: the defender's square; relocate: the square q
    gain: float                     # the frames' answer: what the edit is worth to the side to move (guard: what it would cost)
    hops: int = 1                   # relocate: quiet moves to q


def _value_for(b: chess.Board, t: int, side: chess.Color) -> float:
    bb = _turned(b, side)
    return 0.0 if bb is None else frame_on(bb, t)[0]


def _with_attacker(b: chess.Board, t: int, v: int, side: chess.Color) -> chess.Board | None:
    """The board with one more `side` piece of value v bearing on t, on the nearest empty square it attacks t from."""
    pt = TYPE_OF[v]
    cands = []
    for q in chess.SQUARES:
        if b.piece_at(q) is not None:
            continue
        if pt == chess.PAWN:
            att = chess.BB_PAWN_ATTACKS[side][q]
        else:
            probe = b.copy(stack=False); probe.set_piece_at(q, chess.Piece(pt, side)); att = int(probe.attacks_mask(q))
        if att & chess.BB_SQUARES[t]:
            cands.append((chess.square_distance(q, t), q))
    if not cands:
        return None
    q = min(cands)[1]
    nb = b.copy(stack=False); nb.set_piece_at(q, chess.Piece(pt, side))
    return nb


def edits(b: chess.Board) -> list:
    """Every hypothetical edit for the side to move, answered by the frames (only the ones worth something)."""
    side = b.turn; out = []
    for t, pc in b.piece_map().items():
        if pc.piece_type == chess.KING:
            continue
        if pc.color != side:
            base = _value_for(b, t, side)
            for v in (1, 3, 5, 9):
                nb = _with_attacker(b, t, v, side)
                if nb is not None:
                    g = _value_for(nb, t, side) - base
                    if g > 0:
                        out.append(Edit("add", t, v, g))
            for d in counted_defenders(b, t):
                nb = b.copy(stack=False); nb.remove_piece_at(d)
                k = nb.king(side)
                if k is not None and nb.is_attacked_by(not side, k):
                    continue
                g = _value_for(nb, t, side) - base
                if g > 0:
                    out.append(Edit("remove", t, d, g))
        else:
            base = _value_for(b, t, not side)
            for v in (1, 3, 5, 9):
                nb = _with_attacker(b, t, v, not side)
                if nb is not None:
                    g = _value_for(nb, t, not side) - base
                    if g > 0:
                        out.append(Edit("guard", t, v, g))
            now = max(roles(b, t).values(), default=0.0)
            for q, h, w, r in placements(b, t, 2):                # its best single role there: the opponent saves the biggest target,
                best = max(r.values(), default=0.0)               # so roles are alternatives, not a sum (the rook on e7 was '+15')
                if best > now:
                    out.append(Edit("relocate", t, q, best - now, h))
    return out


def realizes(b: chess.Board, m: chess.Move, e: Edit) -> int:
    """How far the move goes toward making the edit true: 0 not at all; 1 it completes it; 2 it is the first of two steps."""
    side = b.turn; pc = b.piece_at(m.from_square)
    nb = b.copy(stack=False); nb.push(m)
    if e.kind == "add":
        if VALUE[pc.piece_type] > e.arg or pc.piece_type == chess.KING or m.from_square == e.target:
            return 0
        if e.target in nb.attacks(m.to_square) and e.target not in b.attacks(m.from_square):
            return 1
        at = _moved(b, m.from_square, m.to_square) if not b.is_capture(m) else None
        if at is not None and e.target not in b.attacks(m.from_square):
            p = reach(at, m.to_square, e.target, 1)
            if p is not None and p.hops == 1:
                return 2
        return 0
    if e.kind == "remove":
        if m.to_square == e.arg:
            return 1                                              # it takes the defender
        pb = nb.copy(stack=False); pb.turn = side; pb.ep_square = None
        if pb.piece_at(e.target) is not None and e.arg not in counted_defenders(pb, e.target) and nb.piece_at(e.arg) is not None:
            return 1                                              # pinned, blocked or tied: it no longer counts
        return 0
    if e.kind == "relocate":
        if m.from_square != e.target:
            return 0
        if m.to_square == e.arg:
            return 1
        at = _moved(b, m.from_square, m.to_square)
        return 2 if any(q == e.arg for q, h, _, _ in placements(at, m.to_square, 1)) else 0
    if e.kind == "guard":
        if m.from_square == e.target:
            return 1                                              # the piece steps away
        pb = nb.copy(stack=False); pb.turn = side; pb.ep_square = None
        if pb.piece_at(e.target) is not None and len(counted_defenders(pb, e.target)) > len(counted_defenders(b, e.target)):
            return 1
        return 0
    return 0


KINDS = ("add", "remove", "relocate", "guard")


def move_features(b: chess.Board, m: chess.Move, es: list) -> list:
    """For each edit kind: the best gain the move completes, the best it starts (two steps), and how many it touches."""
    out = []
    for k in KINDS:
        done = [e.gain for e in es if e.kind == k and realizes(b, m, e) == 1]
        half = [e.gain for e in es if e.kind == k and realizes(b, m, e) == 2]
        out += [max(done, default=0.0), max(half, default=0.0), len(done) + len(half)]
    return out


FEATURES = [f"{k}_{s}" for k in KINDS for s in ("completes", "starts", "count")]
