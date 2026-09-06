"""goal_search.py -- search TOWARD A GOAL that is not mate (Kaveh 2026-09-06): "win the f7 pawn: this, this, this...
no; this, this... no; give up, next goal."  Proof-number search (Allis 1994) over an AND/OR tree: OR nodes are our
moves, AND nodes the opponent's; a node is proven when the goal predicate holds against every reply within the
horizon. Complete: every legal move is generated on both sides, so a wrong prior costs time, never the answer.
The prior comes from the planner: our plan steps and the opponent's counter-goals are expanded first and start
with lower proof / disproof numbers. The goal predicate is the ledger (the target is gone and our material is up by
its value against the opponent's best recapture sequence, judged at our turn), and every PROOF is confirmed by the
engine at the proof's leaves before it is believed -- the horizon stays honest. Give-up is native: when the root's
disproof number reaches zero (refuted within the horizon) or the node budget runs out, the goal is abandoned with
the hardest defence found, and the goal loop (pursue) moves to the next target of the progress map."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import chess

from . import rules as R
from .defense import defense, all_defenses
from .facts import value, material
from .plans import derive, Goal
from .reach import progress

INF = float("inf")


@dataclass
class Node:
    board: chess.Board
    ply: int                         # plies from the root
    move: chess.Move | None          # the move that led here
    is_or: bool                      # True: our move next (OR); False: the opponent's (AND)
    pn: float = 1.0
    dn: float = 1.0
    children: list["Node"] | None = None
    parent: "Node | None" = None
    target: int | None = None        # where the target piece stands now (it may have moved)
    note: str = ""


@dataclass
class Result:
    goal: str
    proven: bool
    disproven: bool
    nodes: int
    seconds: float
    line: list[str]                  # the principal line (our moves and the opponent's hardest replies) when proven
    refutation: list[str]            # the hardest defence when not proven
    tried: list[str] = field(default_factory=list)   # our first moves tried at the root, in order
    engine_calls: int = 0


class GoalSearch:
    """PN search for one goal: win the enemy piece standing on `target` for `side`, within `horizon` plies."""

    def __init__(self, board: chess.Board, side: chess.Color, target: int, memo=None, engine=None, horizon: int = 6,
                 budget: int = 3000, engine_depth: int = 10, tolerance_cp: int = 40):
        self.root_board = board.copy(stack=False); self.root_board.turn = side
        self.side, self.target0, self.memo, self.engine = side, target, memo, engine
        self.horizon, self.budget, self.engine_depth, self.tol = horizon, budget, engine_depth, tolerance_cp
        p = board.piece_at(target)
        self.target_piece = p
        self.target_value = value(board, target)
        self.mat0 = material(board, side) - material(board, not side)
        self.root_cp = None
        self.nodes = 0; self.vetoes = 0; self.expansions = 0; self.engine_calls = 0
        self.tried: list[str] = []

    # ---- the goal predicate
    def achieved(self, n: Node) -> bool | None:
        """At OUR turn (the opponent has replied): True when the target is gone -- captured by us -- and the engine
        says the position is not worse than the root by more than the tolerance (their counterplay elsewhere is the
        engine's to weigh; without an engine, the material must be up by the target and nothing of ours may fall).
        False at the horizon with the target still standing, or in a hopeless line. None: undecided."""
        b = n.board
        if b.is_game_over():
            return False
        if not n.is_or:
            return None
        net = (material(b, self.side) - material(b, not self.side)) - self.mat0
        if net <= -(self.target_value + 3):                            # a hopeless line: down a piece beyond the target's worth
            return False                                               # (a heuristic floor, stated: a deeper combination could still exist)
        if n.target is None:                                           # the target has been taken
            if self.engine is not None:
                ok = self.engine_ok(n)
            else:
                own_falls = sum(value(b, sq) for sq, d in all_defenses(b, memo=self.memo).items()
                                if d.falls and b.piece_at(sq).color == self.side)
                ok = net - own_falls >= self.target_value - 0.5
            if ok:
                return True
            return False if n.ply >= self.horizon else None            # not yet: we may still consolidate before the horizon
        if n.ply >= self.horizon:
            return False
        return None

    def engine_ok(self, n: Node) -> bool:
        """The engine's verdict on a claimed gain. The material is already in hand (net >= the target); the engine
        only has to say there is no tactical refutation: from our side to move the evaluation must not have fallen
        below the root's by more than the tolerance. (It must not be asked to have RISEN: a root evaluation already
        anticipates a pawn that is falling, so cashing it moves the number very little.)"""
        if self.root_cp is None:
            l = self.engine.best(self.root_board, depth=self.engine_depth)
            self.root_cp = l.cp if l else 0
        l = self.engine.best(n.board, depth=self.engine_depth)
        if l is None:
            return True
        self.engine_calls += 1
        return l.cp >= self.root_cp - self.tol

    # ---- move ordering: the planner's prior at shallow plies, a cheap target-centred order deeper (prior only:
    # every legal move is still generated, and at AND nodes every reply must be answered for a proof)
    def ordered_moves(self, n: Node) -> list[tuple[chess.Move, float]]:
        b = n.board
        legal = list(b.legal_moves)
        prior: dict[chess.Move, float] = {}
        if n.target is not None and n.ply <= 2:
            if n.is_or:
                for pl in derive(b, self.side, Goal("win", n.target), self.memo, depth=1)[:4]:
                    if pl.steps and pl.steps[0].move in b.legal_moves:
                        prior[pl.steps[0].move] = min(prior.get(pl.steps[0].move, 9), float(pl.hops))
            else:
                for pl in derive(b, not self.side, Goal("defend", n.target), self.memo, depth=1)[:4]:
                    if pl.steps and pl.steps[0].move in b.legal_moves:
                        prior[pl.steps[0].move] = min(prior.get(pl.steps[0].move, 9), float(pl.hops))
        t = n.target
        def key(m):
            p = prior.get(m)
            if p is not None:
                return (0, p)
            if t is not None and m.to_square == t:
                return (1, -value(b, m.from_square) if n.is_or else 0)          # capture the target / recapture on it
            if t is not None and not n.is_or and m.from_square == t:
                return (1, 1)                                                    # the target runs
            if b.is_capture(m):
                return (2, -value(b, m.to_square))
            if t is not None and n.is_or:
                nb = b.copy(stack=False); nb.push(m)
                if t in nb.attacks(m.to_square):
                    return (2, 0)                                                # a move that attacks the target
            if b.gives_check(m):
                return (3, 0)
            return (4, 0)
        legal.sort(key=key)
        return [(m, prior.get(m, INF)) for m in legal]

    def init_numbers(self, child: Node, prior: float, moved_attacks: bool) -> tuple[float, float]:
        """The prior as proof/disproof numbers (the ledger's guess of how hard the goal is from here)."""
        if child.is_or:                                        # our move next: how close is the target to falling?
            if child.target is None:
                return 1.0, 1.0
            d = defense(child.board, child.target, memo=self.memo)
            if d is not None and d.falls:
                return 1.0, 2.0
            return (2.0 if prior < INF else 3.0), 1.0
        # their move next (after our move): a plan move or an attack on the target is cheaper to prove
        base = 1.0 if child.target is None else (2.0 if (prior < INF or moved_attacks) else 4.0)
        return base, 1.0

    def expand(self, n: Node):
        n.children = []
        for m, p in self.ordered_moves(n):
            b = n.board.copy(stack=False); b.push(m)
            t = n.target
            if t is not None:
                if m.to_square == t and n.is_or:
                    t = None                                   # we took it
                elif m.from_square == t and not n.is_or:
                    t = m.to_square                            # it ran
                elif m.to_square == t and not n.is_or:
                    t = m.to_square                            # they recaptured on its square: win what stands there now
            child = Node(b, n.ply + 1, m, not n.is_or, parent=n, target=t)
            moved_attacks = n.is_or and t is not None and t in b.attacks(m.to_square)
            child.pn, child.dn = self.init_numbers(child, p, moved_attacks)
            a = self.achieved(child)
            if a is True:
                child.pn, child.dn = 0.0, INF
            elif a is False:
                child.pn, child.dn = INF, 0.0
            n.children.append(child)
            self.nodes += 1
        if not n.children:                                     # no legal moves: the side to move is stuck
            n.pn, n.dn = (INF, 0.0)
        self.expansions += 1
        self.update(n)

    def update(self, n: Node):
        while n is not None:
            if n.children:
                if n.is_or:
                    n.pn = min(c.pn for c in n.children); n.dn = sum(c.dn for c in n.children)
                else:
                    n.pn = sum(c.pn for c in n.children); n.dn = min(c.dn for c in n.children)
                n.dn = min(n.dn, INF); n.pn = min(n.pn, INF)
            n = n.parent

    def select(self, n: Node) -> Node:
        while n.children:
            if n.is_or:
                n = min(n.children, key=lambda c: c.pn)
            else:
                n = min(n.children, key=lambda c: c.dn)
        return n

    def run(self) -> Result:
        t0 = time.perf_counter()
        root = Node(self.root_board, 0, None, True, target=self.target0)
        goal_txt = f"win the {chess.piece_name(self.target_piece.piece_type)} on {chess.square_name(self.target0)}"
        while root.pn != 0 and root.dn != 0 and self.expansions < self.budget:
            leaf = self.select(root)
            if leaf.children is not None:
                break
            if leaf.pn == 0 or leaf.dn == 0:
                self.update(leaf); continue
            self.expand(leaf)
        self.tried = [root.board.san(c.move) for c in (root.children or []) if c.pn < INF][:6]
        line = self.principal_line(root) if root.pn == 0 else []
        ref = self.hardest_defence(root) if root.pn != 0 else []
        return Result(goal_txt, root.pn == 0, root.dn == 0, self.nodes, time.perf_counter() - t0, line, ref, self.tried, self.engine_calls)

    def principal_line(self, root: Node) -> list[str]:
        out = []; n = root
        while n.children:
            if n.is_or:
                c = next((c for c in n.children if c.pn == 0), None)
            else:                                              # the opponent's hardest reply inside the proof: the deepest proven child
                c = max((c for c in n.children if c.pn == 0), key=lambda c: self.depth_of(c), default=None)
            if c is None:
                break
            out.append(n.board.san(c.move)); n = c
        return out

    def depth_of(self, n: Node) -> int:
        if not n.children:
            return 0
        return 1 + max(self.depth_of(c) for c in n.children)

    def hardest_defence(self, root: Node) -> list[str]:
        """The line where our most promising try was refuted: follow min-pn at OR nodes, min-dn at AND nodes."""
        out = []; n = root
        while n.children:
            c = min(n.children, key=lambda c: c.pn) if n.is_or else min(n.children, key=lambda c: c.dn)
            out.append(n.board.san(c.move)); n = c
        return out


# ---------------------------------------------------------------- the goal loop
def pursue(board: chess.Board, side: chess.Color, memo=None, engine=None, budget: int = 3000, horizon: int = 6,
           max_goals: int = 4) -> list[Result]:
    """Try the progress map's targets most valuable first; each gets a budget; a proven goal ends the loop."""
    out = []
    order = {"falls": 0, "joinable": 1, "undermine": 2, "held": 3, "unreachable": 4}
    targets = [t for t in progress(board, side, top=64, memo=memo) if t.status in ("falls", "joinable", "undermine", "held")]
    targets.sort(key=lambda t: (order[t.status], -value(board, t.square)))
    for t in targets[:max_goals]:
        gs = GoalSearch(board, side, t.square, memo=memo, engine=engine, horizon=horizon, budget=budget)
        r = gs.run(); out.append(r)
        if r.proven:
            break
    return out


def result_text(r: Result) -> str:
    if r.proven:
        return f"{r.goal}: proven in {r.nodes} nodes, {r.engine_calls} engine checks ({r.seconds:.1f}s): {' '.join(r.line)}"
    how = f"refuted within the horizon ({r.nodes} nodes)" if r.disproven else f"given up after {r.nodes} nodes"
    return f"{r.goal}: {how} ({r.seconds:.1f}s); tried {', '.join(r.tried) or 'nothing'}; hardest defence: {' '.join(r.refutation)}"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True); ap.add_argument("--side", default=""); ap.add_argument("--target", default="", help="square; empty = the goal loop")
    ap.add_argument("--budget", type=int, default=3000); ap.add_argument("--horizon", type=int, default=6); ap.add_argument("--engine", action="store_true")
    a = ap.parse_args(argv)
    b = chess.Board(a.fen); side = b.turn if not a.side else a.side.lower().startswith("w")
    eng = None
    if a.engine:
        from .engine import Engine
        eng = Engine(depth=12)
    from .memo import Memo
    memo = Memo()
    if a.target:
        rs = [GoalSearch(b, side, chess.parse_square(a.target), memo=memo, engine=eng, horizon=a.horizon, budget=a.budget).run()]
    else:
        rs = pursue(b, side, memo=memo, engine=eng, budget=a.budget, horizon=a.horizon)
    for r in rs:
        print(result_text(r))
    if eng:
        eng.close()


if __name__ == "__main__":
    main()
