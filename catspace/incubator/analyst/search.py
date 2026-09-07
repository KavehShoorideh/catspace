"""search.py -- the concept-driven search pathway (Kaveh 2026-09-05): expand a position and evaluate every node by
its CONCEPTS -- the defense ledgers, both sides' progress maps, the material -- not by a scalar, sharing every
concept through the store (memo.py) so a fact computed in one node is served in every sibling, child or later
run whose footprint carries the same pieces.

    python -m catspace.incubator.analyst.search --fen "<fen>" [--depth 2 --k 4 --memo data/derived/analyst_memo.sqlite]

What a node knows: which pieces fall now (for each side), what each side can make progress on, the material.
What an edge says: the DELTA of concepts -- which targets a move made winnable or safe, which plans it opened or
killed. That delta is the plan layer's answer to 'what did this move do' (ANALYST_PLAN §14)."""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import chess

from . import rules as R
from .defense import all_defenses, Defense
from .facts import value, material
from .memo import Memo
from .reach import progress, Target, target_phrase


@dataclass
class ConceptState:
    fen: str
    material: dict                       # {"w": .., "b": ..}
    falling: dict                        # {"w": [(sq, value)], "b": [...]} pieces of that side that fall now
    progress: dict                       # {"w": [Target], "b": [Target]} what each side can make progress on
    ledgers: dict[int, Defense]

    def falling_value(self, color: str) -> float:
        return sum(v for _, v in self.falling[color])

    def status_map(self, color: str) -> dict[int, str]:
        return {t.square: t.status for t in self.progress[color]}


def concept_state(board: chess.Board, memo: Memo | None = None, hops: int = 2) -> ConceptState:
    ledgers = all_defenses(board, memo=memo)
    falling = {"w": [], "b": []}
    for sq, d in ledgers.items():
        if d.falls:
            falling["w" if board.piece_at(sq).color else "b"].append((sq, value(board, sq)))
    prog = {"w": progress(board, chess.WHITE, max_hops=hops, top=64, memo=memo),
            "b": progress(board, chess.BLACK, max_hops=hops, top=64, memo=memo)}
    return ConceptState(board.fen(), {"w": material(board, chess.WHITE), "b": material(board, chess.BLACK)}, falling, prog, ledgers)


ORDER = {"falls": 0, "joinable": 1, "undermine": 2, "held": 3, "unreachable": 4}


def delta(board: chess.Board, before: ConceptState, after: ConceptState, mover: chess.Color) -> list[str]:
    """What the move did to the concepts, from the mover's side: targets that became winnable or safe, plans
    opened or closed, material that changed hands."""
    me, them = ("w", "b") if mover else ("b", "w")
    out = []
    a_before, a_after = before.status_map(me), after.status_map(me)      # my targets
    for sq in set(a_before) | set(a_after):
        s0, s1 = a_before.get(sq), a_after.get(sq)
        if s0 == s1 or s1 is None:
            continue
        p = after.ledgers.get(sq)
        name = _name(board, sq)
        if s1 == "falls" and s0 != "falls":
            out.append(f"{name} now falls")
        elif s1 == "joinable" and s0 in (None, "held", "unreachable"):
            out.append(f"{name} became reachable")
        elif s1 in ("held", "unreachable") and s0 in ("falls", "joinable", "undermine"):
            out.append(f"{name} is safe again for them")
    d_before, d_after = before.status_map(them), after.status_map(them)   # their targets = my pieces
    for sq in set(d_before) | set(d_after):
        s0, s1 = d_before.get(sq), d_after.get(sq)
        if s0 == s1 or s1 is None:
            continue
        name = _name(board, sq)
        if s1 == "falls" and s0 != "falls":
            out.append(f"my {name} now falls")
        elif s1 == "joinable" and s0 in (None, "held", "unreachable"):
            out.append(f"my {name} became reachable for them")
        elif s1 in ("held", "unreachable") and s0 in ("falls", "joinable", "undermine"):
            out.append(f"my {name} is safe now")
    dm = (after.material[me] - after.material[them]) - (before.material[me] - before.material[them])
    if dm:
        out.append(f"material {dm:+g}")
    return out


def _name(board: chess.Board, sq: int) -> str:
    p = board.piece_at(sq)
    return f"{chess.piece_name(p.piece_type) if p else 'piece'} on {chess.square_name(sq)}"


@dataclass
class Node:
    board: chess.Board
    state: ConceptState
    move: str | None = None              # SAN that led here
    delta: list[str] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)
    cp: int | None = None                # the engine's number, when an engine is given (never the evaluation, only the order)


def candidates(board: chess.Board, state: ConceptState, engine=None, k: int = 4) -> list[chess.Move]:
    """Moves worth expanding: the engine's top-k when given; always the captures of falling pieces, the first hops
    of every joiner path, and checks."""
    out = []
    if engine is not None:
        for l in engine.analyse(board, multipv=k):
            out.append(chess.Move.from_uci(l.move))
    me = "w" if board.turn else "b"
    for t in state.progress[me]:
        if t.status == "falls" and t.ledger:
            for a in t.ledger.effective_attackers:
                mv = chess.Move(a, t.square)
                if mv in board.legal_moves:
                    out.append(mv); break
        elif t.status == "joinable" and t.joiner and t.joiner.squares:
            mv = chess.Move(t.joiner.piece, t.joiner.squares[0])
            if mv in board.legal_moves:
                out.append(mv)
    for mv in board.legal_moves:
        if board.gives_check(mv) and len(out) < k + 6:
            out.append(mv)
    seen, uniq = set(), []
    for mv in out:
        if mv not in seen:
            seen.add(mv); uniq.append(mv)
    return uniq[:k + 6]


def search(board: chess.Board, depth: int = 2, k: int = 4, memo: Memo | None = None, engine=None, hops: int = 2) -> Node:
    root = Node(board.copy(stack=False), concept_state(board, memo, hops))
    _expand(root, depth, k, memo, engine, hops)
    return root


def _expand(node: Node, depth: int, k: int, memo, engine, hops: int):
    if depth == 0 or node.board.is_game_over():
        return
    for mv in candidates(node.board, node.state, engine, k):
        b = node.board.copy(stack=False); san = b.san(mv); b.push(mv)
        st = concept_state(b, memo, hops)
        child = Node(b, st, san, delta(b, node.state, st, not b.turn))
        node.children.append(child)
        _expand(child, depth - 1, k, memo, engine, hops)


def render(node: Node, indent: int = 0, top: int = 3) -> list[str]:
    pad = "  " * indent
    out = []
    if node.move:
        out.append(f"{pad}{node.move}: " + ("; ".join(node.delta) if node.delta else "no concept changed"))
    side = "w" if node.board.turn else "b"
    lines = [target_phrase(node.board, t) for t in sorted(node.state.progress[side], key=lambda t: (ORDER[t.status], -value(node.board, t.square)))
             if t.status in ("falls", "joinable", "undermine")][:top]
    for l in lines:
        out.append(f"{pad}  · {'White' if node.board.turn else 'Black'} to move: {l}")
    for c in node.children:
        out.extend(render(c, indent + 1, top))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True); ap.add_argument("--depth", type=int, default=2); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--memo", default="data/derived/analyst_memo.sqlite", help="the concept store; '' for none")
    ap.add_argument("--engine", action="store_true", help="order candidates with Stockfish")
    ap.add_argument("--hops", type=int, default=2)
    a = ap.parse_args(argv)
    memo = Memo.load(a.memo) if a.memo else None
    n0 = len(memo) if memo else 0
    eng = None
    if a.engine:
        from .engine import Engine
        eng = Engine(depth=12)
    t0 = time.perf_counter()
    root = search(chess.Board(a.fen), a.depth, a.k, memo, eng, a.hops)
    dt = time.perf_counter() - t0
    if eng:
        eng.close()
    print("\n".join(render(root)))
    nodes = 1 + sum(1 for _ in _walk(root))
    print(f"[search] {nodes} nodes in {dt:.1f}s" + (f" | {memo.report()} | store {n0} -> {len(memo)} entries" if memo else ""))
    if memo:
        memo.save()


def _walk(node: Node):
    for c in node.children:
        yield c
        yield from _walk(c)


if __name__ == "__main__":
    main()
