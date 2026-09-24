"""FRAMES AS LIVE OBJECTS (Kaveh 2026-09-24): an idea is a set of pieces and a line; its value is the frame's own
episode along the line; its LOGIC is the relations among its pieces (the signature). A board change re-derives an
idea logically: only the relations that changed are named, and only ideas whose pieces are touched are recomputed
("the same tactic works, but it only wins a pawn now"). The LISTENER answers "which frames does this piece
affect?" and, when one frame's line moves a piece, marks the other frames holding that piece for recalculation,
with the reason. Nothing here knows what a fork or a check is: labels are the caller's."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import chess

from .chess_domain import State, _ids_after, frame_episode_scalar, frame_scalar


@dataclass
class Idea:
    name: str                      # a label from outside the mechanism ("check response", "fork")
    ids: list                      # the frame's pieces, by square now (None once captured)
    move: chess.Move | None        # the idea's first move (None: a standing frame)
    value: float | None = None     # the frame episode's delta for the move, from the current position
    signature: dict = field(default_factory=dict)   # (relation, slots) -> bool: the logic the value rests on
    depth: int = 0                 # episode depth (0: the frame's size, capped)
    history: list = field(default_factory=list)     # (touched_by, changed relations, old value, new value)


def signature(dom, board: chess.Board, ids: list) -> dict:
    """Every base state relation over the frame's pieces, by slot: the logical state of the frame."""
    st = State(board.copy(stack=False), list(ids))
    out = {}
    for r in dom.relations:
        if r.on_action or r.name.startswith(("after_", "then_", "some_", "threat_")) or r.arity > 2:
            continue
        for sl in itertools.permutations(range(len(ids)), r.arity):
            args = [ids[s] for s in sl]
            if any(a is None for a in args):
                out[(r.name, sl)] = None
                continue
            try:
                out[(r.name, sl)] = bool(r.fn(st, *args))
            except Exception:
                out[(r.name, sl)] = None
    return out


def value_of(board: chess.Board, ids: list, move: chess.Move, depth: int = 0, max_depth: int = 5) -> float | None:
    """The frame episode's delta for `move`: material of the mover's side counted over the frame's pieces, after a
    line of up to `depth` plies (0: the frame's size), minus the material before."""
    live = [i for i in ids if i is not None]
    if move.from_square not in live or move not in board.legal_moves:
        return None
    b = board.copy(stack=False)
    st = State(b, list(ids))
    before = frame_scalar(st, move.from_square)
    nb = b.copy(stack=False); nb.push(move)
    after = State(nb, _ids_after(list(ids), b, move))
    d = min(max_depth, depth if depth > 0 else len(live))
    return frame_episode_scalar(after, b.piece_at(move.from_square).color, d) - before


class Listener:
    """Frames indexed by the pieces they hold; a move re-derives every frame it touches and names why."""

    def __init__(self, dom, board: chess.Board):
        self.dom = dom; self.board = board.copy(stack=False); self.ideas: dict = {}

    def register(self, name: str, ids: list, move: chess.Move | None, depth: int = 0) -> Idea:
        idea = Idea(name, list(ids), move, depth=depth)
        idea.signature = signature(self.dom, self.board, idea.ids)
        idea.value = value_of(self.board, idea.ids, move, depth) if move else None
        self.ideas[name] = idea
        return idea

    def affected(self, square: int) -> list:
        """Which frames does this piece affect? The names of every idea holding it."""
        return [n for n, i in self.ideas.items() if square in i.ids]

    def apply(self, move: chess.Move, by: str | None = None) -> list:
        """Play `move` on the listener's board; every idea holding a piece the move touches (moved, captured, or
        whose relations changed) is re-derived: its signature diffed, its value recomputed only if something
        changed. Returns [(name, touched, changed relations, old value, new value)]."""
        b0 = self.board.copy(stack=False)
        self.board.push(move)
        report = []
        for name, idea in self.ideas.items():
            new_ids = _ids_after(idea.ids, b0, move)
            moved = [(a, c) for a, c in zip(idea.ids, new_ids) if a != c]
            idea.ids = new_ids
            if idea.move is not None and idea.move.from_square in [a for a, _ in moved]:
                idea.move = chess.Move(dict(moved)[idea.move.from_square], idea.move.to_square) if dict(moved)[idea.move.from_square] is not None else None
            sig = signature(self.dom, self.board, idea.ids)
            changed = [(k, idea.signature.get(k), v) for k, v in sig.items() if idea.signature.get(k) != v]
            touched = bool(moved) or bool(changed)
            old = idea.value
            if touched and idea.move is not None:
                idea.value = value_of(self.board, idea.ids, idea.move, idea.depth)
            idea.signature = sig
            if touched:
                idea.history.append((by, changed, old, idea.value))
            report.append((name, touched, changed, old, idea.value))
        return report

    def why(self, name: str) -> str:
        """The last re-derivation of an idea, in words: who touched it, which relations flipped, the value before and after."""
        idea = self.ideas[name]
        if not idea.history:
            return f"{name}: untouched, value {idea.value}"
        by, changed, old, new = idea.history[-1]
        rel = "; ".join(f"{r}({','.join(chess.square_name(idea.ids[s]) if idea.ids[s] is not None else '-' for s in sl)}) {a}->{b}" for (r, sl), a, b in sorted(changed, key=lambda c: (c[0][0] != "defends", c[0][0] != "attacks", c[0][0])))
        return f"{name}: touched by {by or 'the move'}; {rel or 'no relation changed'}; value {old} -> {new}"
