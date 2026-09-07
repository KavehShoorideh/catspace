"""goal_search.py -- search TOWARD A GOAL that is not mate (Kaveh 2026-09-06): "win the f7 pawn: this, this, this...
no; this, this... no; give up, next goal."  Proof-number search (Allis 1994) over an AND/OR tree: OR nodes are our
moves, AND nodes the opponent's; a node is proven when the goal predicate holds against every reply within the
horizon. Complete: every legal move is generated on both sides, so a wrong prior costs time, never the answer.
The prior comes from the planner: our plan steps and the opponent's counter-goals are expanded first and start
with lower proof / disproof numbers. The goal predicate at our turn: the target is gone, our material is up by its
value on the raw count (a pending capture may extend the horizon a few plies to cash it), and the engine, looking
to the same absolute horizon from every leaf, says the position is not worse than the root by more than the
tolerance -- the horizon stays honest. Give-up is native: when the root's disproof number reaches zero (refuted
within the horizon) or the node budget runs out, the goal is abandoned with the hardest defence found, and the goal
loop (pursue) moves to the next target of the progress map.

    python -m catspace.incubator.analyst.goal_search --fen "<fen>" --target c5 --engine --trace --budget 600

The trace (one line per expansion) is the transcript of the search: see docs/ANALYST_GOAL_SEARCH.md."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

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
    cp: float | None = None          # the judge's number at a proven leaf (engine cp, or the settled count in pawns x100)


@dataclass
class Result:
    goal: str
    proven: bool
    disproven: bool
    nodes: int
    seconds: float
    line: list[str]                  # the principal line (our moves and the opponent's best replies by the judge) when proven
    refutation: list[str]            # the hardest defence when not proven
    tried: list[str] = field(default_factory=list)   # our first moves tried at the root, in order
    engine_calls: int = 0
    work: list[str] = field(default_factory=list)    # the line inside the proof that took the most refuting (their deepest try)
    refinements: list["Refinement"] = field(default_factory=list)   # the sketch revised after a refutation, and what came of it
    diagnosis: str = ""              # what the refuting reply did to us, when the goal was not proven
    served: str = ""                 # 'transposition' | 'sketch' when the concept store supplied the proof (a sketch is re-verified)


@dataclass
class Refinement:
    """The Lean-error-to-next-sketch step (Kaveh 2026-09-07): the refuting reply is diagnosed, a counter-goal is
    derived for us to meet FIRST, its moves become the root prior, the horizon grows by the two plies the
    preparation costs, and the search runs again."""
    reply: str                       # the refuting reply, SAN, on the hardest-defence line
    diagnosis: str                   # what it did: 'takes my rook on a4' / 'my knight on f6 now falls'
    counter: str                     # the counter-goal in words: 'first secure the rook on a4'
    prior: list[str]                 # the moves injected first at the root, SAN
    result: "Result | None"          # the re-run


class GoalSearch:
    """PN search for one goal: win the enemy piece standing on `target` for `side`, within `horizon` plies."""

    def __init__(self, board: chess.Board, side: chess.Color, target: int, memo=None, engine=None, horizon: int = 6,
                 budget: int = 3000, engine_depth: int = 8, tolerance_cp: int = 40, prior_moves=(), sketch=None):
        self.root_board = board.copy(stack=False); self.root_board.turn = side
        self.side, self.target0, self.memo, self.engine = side, target, memo, engine
        self.prior_moves = list(prior_moves)       # moves a refinement (or a caller) wants tried first at the root
        self.sketch = sketch or {}                 # a stored proof's OR moves by path (uci tuple), tried first when the path recurs
        self.horizon, self.budget, self.engine_depth, self.tol = horizon, budget, engine_depth, tolerance_cp
        p = board.piece_at(target)
        self.target_piece = p
        self.target_value = value(board, target)
        self.mat0 = material(board, side) - material(board, not side)
        self.root_cp = None
        self.nodes = 0; self.vetoes = 0; self.expansions = 0; self.engine_calls = 0
        self.tried: list[str] = []
        self.trace: list[str] = []                 # a narration of the search, one line per expansion (see narrate)
        self.verbose = False
        self.root: Node | None = None

    # ---- the goal predicate
    def achieved(self, n: Node) -> bool | None:
        """At OUR turn (the opponent has replied): True when the target is gone -- captured by us -- and the engine
        says the position is not worse than the root by more than the tolerance (their counterplay elsewhere is the
        engine's to weigh; without an engine, the material must be up by the target and nothing of ours may fall).
        False at the horizon with the target still standing, or in a hopeless line. None: undecided. A leaf in the
        middle of an exchange (their capture, our recapture pending) is judged on the SETTLED count and may run a
        few plies past the horizon to close it -- the horizon must not fall between Qxd5 and cxd5."""
        b = n.board
        if b.is_game_over():
            return False
        if not n.is_or:
            return None
        net = self.net(b)
        settled = self.settled_net(n)
        if settled <= -(self.target_value + 3):                        # a hopeless line: down a piece beyond the target's worth
            return False                                               # (a heuristic floor, stated: a deeper combination could still exist)
        extend = settled > net and n.ply < self.horizon + 4            # a capture is pending: a few more plies to cash it
        if n.target is None:                                           # the target has been taken
            ok = net >= self.target_value - 0.5                        # the material must actually be in hand...
            if ok:
                if self.engine is not None:
                    ok = self.engine_ok(n)                             # ...and the engine must see no refutation
                else:
                    own_falls = sum(value(b, sq) for sq, d in all_defenses(b, memo=self.memo).items()
                                    if d.falls and b.piece_at(sq).color == self.side)
                    ok = net - own_falls >= self.target_value - 0.5
                    n.cp = 100.0 * (net - own_falls)
            if ok:
                return True
            if net >= self.target_value - 0.5:
                return False                                           # in hand, but the judge sees a refutation: final (it searched deeper than we will)
            if n.ply >= self.horizon and not extend:
                return False
            return None                                                # not in hand yet (a recapture pending): undecided before the horizon
        if n.ply >= self.horizon:
            return False
        return None

    def net(self, b: chess.Board) -> float:
        return (material(b, self.side) - material(b, not self.side)) - self.mat0

    def settled_net(self, n: Node) -> float:
        """The material count at our turn after our best capture by the ledger (pins and overloads counted): the
        quiescence stand-in. A leaf in the middle of an exchange, or with a piece of theirs still hanging, is judged
        on this count and allowed to run past the horizon to cash it."""
        b = n.board
        net = self.net(b)
        best = 0.0
        for sq, d in all_defenses(b, memo=self.memo).items():
            if d.falls and b.piece_at(sq).color != self.side:
                best = max(best, d.see)
        return net + best

    def engine_ok(self, n: Node) -> bool:
        """The engine's verdict on a claimed gain. The material is already in hand (net >= the target); the engine
        only has to say there is no tactical refutation: from our side to move the evaluation must not have fallen
        below the root's by more than the tolerance. (It must not be asked to have RISEN: a root evaluation already
        anticipates a pawn that is falling, so cashing it moves the number very little.)"""
        if self.root_cp is None:
            l = self.engine.best(self.root_board, depth=self.engine_depth + self.horizon)
            self.root_cp = l.cp if l else 0
        l = self.engine.best(n.board, depth=self.judge_depth(n))
        if l is None:
            return True
        self.engine_calls += 1
        n.cp = l.cp
        return l.cp >= self.root_cp - self.tol

    def judge_depth(self, n: Node) -> int:
        """The judge looks to the same absolute horizon from every leaf: engine_depth at the search horizon, deeper
        at shallower leaves, engine_depth + horizon at the root. Numbers from different plies are then comparable
        (a flat depth would let the same line be judged 'worse' at ply 4 and 'fine' at ply 6 on search noise)."""
        return self.engine_depth + max(0, self.horizon - n.ply)

    # ---- move ordering: the planner's prior at shallow plies, a cheap target-centred order deeper (prior only:
    # every legal move is still generated, and at AND nodes every reply must be answered for a proof)
    def ordered_moves(self, n: Node) -> list[tuple[chess.Move, float]]:
        b = n.board
        legal = list(b.legal_moves)
        prior: dict[chess.Move, float] = {}
        if n.ply == 0:
            for i, m in enumerate(self.prior_moves):
                if m in b.legal_moves:
                    prior[m] = 0.0 + 0.01 * i
        if self.sketch and n.is_or:
            u = self.sketch.get(self.path_uci(n))
            if u is not None:
                m = chess.Move.from_uci(u)
                if m in b.legal_moves:
                    prior[m] = 0.0
        if n.ply <= 3:
            if n.target is not None:
                if n.is_or:
                    for pl in derive(b, self.side, Goal("win", n.target), self.memo, depth=1)[:4]:
                        if pl.steps and pl.steps[0].move in b.legal_moves:
                            prior[pl.steps[0].move] = min(prior.get(pl.steps[0].move, 9), float(pl.hops))
                else:
                    for pl in derive(b, not self.side, Goal("defend", n.target), self.memo, depth=1)[:4]:
                        if pl.steps and pl.steps[0].move in b.legal_moves:
                            prior[pl.steps[0].move] = min(prior.get(pl.steps[0].move, 9), float(pl.hops))
            elif n.is_or:                                      # the target is taken: now bank it -- save whatever of ours falls
                for sq, d in all_defenses(b, memo=self.memo).items():
                    if d.falls and b.piece_at(sq).color == self.side and value(b, sq) >= 1:
                        for pl in derive(b, self.side, Goal("defend", sq), self.memo, depth=1)[:3]:
                            if pl.steps and pl.steps[0].move in b.legal_moves:
                                prior[pl.steps[0].move] = min(prior.get(pl.steps[0].move, 9), float(pl.hops))
        t = n.target
        def key(m):
            p = prior.get(m)
            if p is not None:
                return (0, p)
            if t is not None and m.to_square == t:
                return (1, value(b, m.from_square) if n.is_or else 0)           # capture the target (cheapest first) / recapture on it
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
        """The prior as proof/disproof numbers: the ledger's guess of how much work the goal needs from here.
        A fresh AND leaf (their move next) is priced by how many of their replies will need work: after a capture
        of the target almost none (the engine settles them at once); after a plan move or an attack on the target a
        few; after an unrelated quiet move most of them. Without this a fresh sibling always looks cheaper than an
        expanded one and the search goes breadth-first over all our moves."""
        n_moves = child.board.legal_moves.count()
        if child.is_or:                                        # our move next: how close is the target to falling?
            if child.target is None:                           # taken, but not yet banked: the material deficit is the work left
                deficit = max(0.0, self.target_value - 0.5 - self.net(child.board))
                return 1.0 + 2.0 * deficit, max(1.0, 0.5 * n_moves)
            d = defense(child.board, child.target, memo=self.memo)
            if d is not None and d.falls:
                return 1.0, max(1.0, 0.5 * n_moves)
            return (2.0 if prior < INF else 3.0), max(1.0, 0.5 * n_moves)
        if child.target is None:                               # taken; their reply may still unbank it (a counter-capture)
            deficit = max(0.0, self.target_value - 0.5 - self.net(child.board))
            factor = 0.2 if prior < INF else 0.5                           # a move that banks (saves what hangs) leaves fewer replies open
            return max(1.0, factor * n_moves) + 2.0 * deficit, 1.0
        factor = 1.0 if (prior < INF or moved_attacks) else 2.5           # expected work per reply, so a fresh sibling
        return max(1.0, factor * n_moves), 1.0                             # is priced like an expanded one

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
            if n.is_or and n.move is not None and m.to_square == n.move.to_square and n.board.is_capture(m):
                p = min(p, 1.0)                                # the recapture of what just captured: as good a prior as any plan
            child.pn, child.dn = self.init_numbers(child, p, moved_attacks)
            a = self.achieved(child)
            if a is True:
                child.pn, child.dn = 0.0, INF
            elif a is False:
                child.pn, child.dn = INF, 0.0
                child.note = self.why_failed(child)
            n.children.append(child)
            self.nodes += 1
        if not n.children:                                     # no legal moves: the side to move is stuck
            n.pn, n.dn = (INF, 0.0)
        self.expansions += 1
        self.update(n)
        if self.verbose:
            self.narrate(n)

    def why_failed(self, n: Node) -> str:
        b = n.board
        if b.is_game_over():
            return "game over"
        net = self.settled_net(n)
        if net <= -(self.target_value + 3):
            return f"down {-net:g}"
        if n.target is None and self.net(b) >= self.target_value - 0.5:
            return f"engine {n.cp:+.0f} vs root {self.root_cp:+.0f}" if n.cp is not None and self.root_cp is not None else "engine says no"
        return "horizon"

    def path_uci(self, n: Node) -> tuple:
        out = []
        while n.parent is not None:
            out.append(n.move.uci()); n = n.parent
        return tuple(reversed(out))

    # ---- the proof as a storable object: its footprint and its sketch
    def proof_footprint(self) -> int:
        """The squares the proof depends on, as a bitboard: every move inside the proof (from, to, the path between),
        the mover's pin line on its board, the target's attack reads at the root. Sound enough to be a KEY -- a
        sketch served on a footprint match is re-verified by a fresh search, never believed."""
        from .memo import attack_reads, pin_line
        mask = attack_reads(self.root_board, self.target0) | chess.BB_SQUARES[self.target0]
        def walk(n):
            nonlocal mask
            for c in n.children or []:
                if n.is_or and c.pn != 0:
                    continue
                if not n.is_or and not c.children:
                    continue                                   # a reply the judge settled at once: not part of the sketch's key
                m = c.move
                mask |= chess.BB_SQUARES[m.from_square] | chess.BB_SQUARES[m.to_square] | int(chess.SquareSet.between(m.from_square, m.to_square))
                p = n.board.piece_at(m.from_square)
                if p is not None:
                    mask |= pin_line(n.board, p.color, m.from_square)
                    k = n.board.king(p.color)
                    if k is not None:
                        mask |= chess.BB_SQUARES[k]
                walk(c)
        if self.root is not None and self.root.pn == 0:
            walk(self.root)
        return mask

    def proof_sketch(self) -> dict:
        """path (uci tuple) -> our move (uci) at every OR node inside the proof."""
        out = {}
        def walk(n, path):
            for c in n.children or []:
                if n.is_or:
                    if c.pn != 0:
                        continue
                    out[path] = c.move.uci()
                walk(c, path + (c.move.uci(),))
        if self.root is not None and self.root.pn == 0:
            walk(self.root, ())
        return out

    def path_of(self, n: Node) -> str:
        sans = []
        while n.parent is not None:
            sans.append(n.parent.board.san(n.move)); n = n.parent
        return " ".join(reversed(sans)) or "(root)"

    def narrate(self, n: Node):
        b = n.board
        who = "we" if n.is_or else "they"
        def fmt(c):
            san = b.san(c.move)
            if c.pn == 0: return f"{san}✓"
            if c.dn == 0: return f"{san}✗"
            return f"{san}({c.pn:g}/{c.dn:g})"
        kids = n.children or []
        proven = [c for c in kids if c.pn == 0]; dis = [c for c in kids if c.dn == 0]; und = [c for c in kids if 0 < c.pn < INF]
        def why(c):
            return f"{b.san(c.move)} ({c.note})" if c.note else b.san(c.move)
        line = (f"#{self.expansions:<3} {'OR ' if n.is_or else 'AND'} {self.path_of(n)}: {who} to move, {len(kids)} moves -> "
                f"{len(proven)} reach the goal, {len(dis)} fail, {len(und)} open"
                + (f"; open: {' '.join(fmt(c) for c in und[:6])}" if und else "")
                + (f"; first proven: {b.san(proven[0].move)}" if proven and n.is_or else "")
                + (f"; refuted by: {', '.join(why(c) for c in dis[:3])}" if dis and not n.is_or else "")
                + (f"; failing: {', '.join(why(c) for c in dis[:3])}" if dis and n.is_or and not und and not proven else "")
                + f" | node pn={n.pn:g} dn={n.dn:g}")
        self.trace.append(line)

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
        self.root = root
        goal_txt = f"win the {chess.piece_name(self.target_piece.piece_type)} on {chess.square_name(self.target0)}"
        while root.pn != 0 and root.dn != 0 and self.expansions < self.budget:
            leaf = self.select(root)
            if leaf.children is not None:
                break
            if leaf.pn == 0 or leaf.dn == 0:
                self.update(leaf); continue
            self.expand(leaf)
        self.tried = [root.board.san(c.move) for c in (root.children or [])          # worked on: expanded, or refuted at the leaf
                      if c.children is not None or c.dn == 0][:6]
        line = self.principal_line(root) if root.pn == 0 else []
        ref = self.hardest_defence(root) if root.pn != 0 else []
        work = self.work_line(root) if root.pn == 0 else []
        return Result(goal_txt, root.pn == 0, root.dn == 0, self.nodes, time.perf_counter() - t0, line, ref, self.tried, self.engine_calls, work)

    def principal_line(self, root: Node) -> list[str]:
        """The line inside the proof where the opponent does best: at their nodes the proven reply with the lowest
        judge's number for us, at ours the proven move with the highest (minimax over the proof's leaves)."""
        out = []; n = root
        while n.children:
            proven = [c for c in n.children if c.pn == 0]
            if not proven:
                break
            c = max(proven, key=self.proof_value) if n.is_or else min(proven, key=self.proof_value)
            out.append(n.board.san(c.move)); n = c
        return out

    def work_line(self, root: Node) -> list[str]:
        """The line inside the proof that needed the most search: at their nodes the proven reply with the deepest
        subtree, at ours the proven move."""
        out = []; n = root
        while n.children:
            proven = [c for c in n.children if c.pn == 0]
            if n.is_or:
                c = next((c for c in proven if c.children), proven[0] if proven else None)
            else:
                proven = [c for c in proven if c.children]          # a reply settled at the leaf took no work
                c = max(proven, key=self.depth_of) if proven else None
            if c is None:
                break
            out.append(n.board.san(c.move)); n = c
        return out

    def depth_of(self, n: Node) -> int:
        if not n.children:
            return 0
        return 1 + max(self.depth_of(c) for c in n.children)

    def proof_value(self, n: Node) -> float:
        """The judge's number at the end of a proven subtree, minimaxed: a leaf's cp; at our nodes the best proven
        child; at theirs the worst. A proven leaf without a number (no engine, no ledger call) counts as 0."""
        if not n.children:
            return n.cp if n.cp is not None else 0.0
        vals = [self.proof_value(c) for c in n.children if c.pn == 0]
        if not vals:
            return 0.0
        return max(vals) if n.is_or else min(vals)

    def hardest_defence(self, root: Node) -> list[str]:
        """The line where our most promising try was refuted: follow min-pn at OR nodes, min-dn at AND nodes."""
        out = []; n = root
        while n.children:
            c = min(n.children, key=lambda c: c.pn) if n.is_or else min(n.children, key=lambda c: c.dn)
            out.append(n.board.san(c.move)); n = c
        return out


# ---------------------------------------------------------------- refinement: the refutation drives the next sketch
def refutations(gs: "GoalSearch") -> list[tuple[chess.Move, chess.Board, chess.Move]]:
    """Our root moves that were refuted, each with the reply that refuted it: (our move, the board after it, the
    reply). Refuted moves first in the order they were tried, then the most promising open move with its hardest
    reply."""
    root = gs.root
    out = []
    if root is None or not root.children:
        return out
    t = gs.target0
    for c in sorted(root.children, key=lambda c: (c.move.to_square != t, root.children.index(c))):   # captures of the target first
        if c.dn == 0 and c.children:
            theirs = next((k for k in c.children if k.dn == 0), None) or min(c.children, key=lambda k: k.dn)
            out.append((c.move, c.board, theirs.move))
    open_ = [c for c in root.children if 0 < c.pn < INF and c.children]
    if open_:
        c = min(open_, key=lambda k: k.pn)
        out.append((c.move, c.board, min(c.children, key=lambda k: k.dn).move))
    return out


def diagnose(board: chess.Board, reply: chess.Move, side: chess.Color, target: int | None, memo=None) -> tuple[str, int | None, int | None]:
    """What the reply did to us, in the ledger's terms: a capture of ours off the target square, or a piece of ours
    that falls now and did not before. Returns (text, the square of ours to secure at the root, the reply's piece)."""
    b = board
    after = b.copy(stack=False); after.push(reply)
    who = lambda bb, sq: f"my {chess.piece_name(bb.piece_at(sq).piece_type)} on {chess.square_name(sq)}"
    taken = b.piece_at(reply.to_square)
    if b.is_capture(reply) and taken is not None and taken.color == side and reply.to_square != target:
        return f"takes {who(b, reply.to_square)}", reply.to_square, reply.from_square
    before = {sq for sq, d in all_defenses(b, memo=memo).items() if d.falls and b.piece_at(sq).color == side}
    now = {sq for sq, d in all_defenses(after, memo=memo).items() if d.falls and after.piece_at(sq).color == side}
    new = [sq for sq in now - before if sq != target and sq != reply.to_square]
    if new:
        sq = max(new, key=lambda q: value(after, q))
        return f"{who(after, sq)} now falls", sq, reply.to_square
    if b.gives_check(reply):
        return "gives check", None, None
    return "holds", None, None


def secure_prior(board: chess.Board, side: chess.Color, sq: int, attacker: int | None, memo=None, top: int = 6) -> list[chess.Move]:
    """Moves that meet the counter-goal 'secure the piece on sq' before the plan: the planner's defend plans when it
    already falls; otherwise step it aside, add a defender (a piece whose move attacks sq), or win the attacker."""
    b = board.copy(stack=False); b.turn = side
    out: list[chess.Move] = []
    for pl in derive(board, side, Goal("defend", sq), memo, depth=1)[:3]:
        if pl.steps and pl.steps[0].move in b.legal_moves:
            out.append(pl.steps[0].move)
    if attacker is not None and board.piece_at(attacker) is not None and board.piece_at(attacker).color != side:
        for pl in derive(board, side, Goal("win", attacker), memo, depth=1)[:2]:
            if pl.steps and pl.steps[0].move in b.legal_moves:
                out.append(pl.steps[0].move)
    for m in b.legal_moves:                                   # step aside
        if m.from_square == sq:
            out.append(m)
    for m in b.legal_moves:                                   # add a defender
        if m.from_square == sq or b.is_capture(m):
            continue
        nb = b.copy(stack=False); nb.push(m)
        if sq in nb.attacks(m.to_square):
            out.append(m)
    seen, uniq = set(), []
    for m in out:
        if m not in seen:
            seen.add(m); uniq.append(m)
    return uniq[:top]


def refine(board: chess.Board, side: chess.Color, target: int, gs: "GoalSearch", memo=None, engine=None, budget: int = 400,
           horizon: int = 6, engine_depth: int = 8) -> "Refinement | None":
    """One round of sketch revision after a failed search: diagnose the refuting reply, derive the counter-goal,
    inject its moves as the root prior, widen the horizon by the preparation's two plies, search again."""
    root_board = board.copy(stack=False); root_board.turn = side
    found = None
    for first, before, reply in refutations(gs):
        text, sq, attacker = diagnose(before, reply, side, gs.target0, memo)
        san = f"{root_board.san(first)} {before.san(reply)}"
        if found is None:
            found = (san, text)
        if sq is None or first.to_square == sq:                # no securable piece, or the plan's landing square itself is the problem
            continue
        if root_board.piece_at(sq) is None or root_board.piece_at(sq).color != side:
            continue
        found = (san, text); break
    else:
        if found is None:
            return None
        san, text = found
        return Refinement(san, text, "", [], None)
    san, text = found
    prior = secure_prior(root_board, side, sq, attacker, memo)
    counter = f"first secure the {chess.piece_name(root_board.piece_at(sq).piece_type)} on {chess.square_name(sq)}"
    if not prior:
        return Refinement(san, text, counter, [], None)
    gs2 = GoalSearch(board, side, target, memo=memo, engine=engine, horizon=horizon + 2, budget=budget,
                     engine_depth=engine_depth, prior_moves=prior)
    r2 = gs2.run()
    return Refinement(san, text, counter, [root_board.san(m) for m in prior], r2)


def search_goal(board: chess.Board, side: chess.Color, target: int, memo=None, engine=None, budget: int = 3000, horizon: int = 6,
                engine_depth: int = 8, refinements: int = 1) -> Result:
    """The goal search with the refinement loop: on failure, revise the sketch from the refutation and try again,
    up to `refinements` times. The returned Result carries the chain."""
    key = (int(side), target, horizon, engine_depth)
    if memo is not None:
        rb = board.copy(stack=False); rb.turn = side
        r = memo.lookup("proof", key, rb)                              # an exact transposition: the proof holds as it stands
        if r is not None:
            r = replace(r, served="transposition", seconds=0.0)
            return r
        sk = memo.lookup("sketch", key, rb)                            # the same pieces on the proof's footprint: re-verify the sketch
        if sk is not None:
            gs = GoalSearch(board, side, target, memo=memo, engine=engine, horizon=horizon, budget=min(budget, 150),
                            engine_depth=engine_depth, sketch=sk)
            r = gs.run()
            if r.proven:
                r.served = "sketch"
                _store_proof(memo, key, rb, gs, r)
                return r
    gs = GoalSearch(board, side, target, memo=memo, engine=engine, horizon=horizon, budget=budget, engine_depth=engine_depth)
    r = gs.run()
    if r.proven and memo is not None:
        _store_proof(memo, key, rb if memo is not None else None, gs, r)
    for _ in range(refinements):
        if r.proven:
            break
        ref = refine(board, side, target, gs, memo, engine, budget, horizon, engine_depth)
        if ref is None:
            break
        r.diagnosis = r.diagnosis or f"{ref.reply}: {ref.diagnosis}"
        r.refinements.append(ref)
        if ref.result is None:
            break
        if ref.result.proven:
            r = Result(r.goal, True, False, r.nodes + ref.result.nodes, r.seconds + ref.result.seconds, ref.result.line, [],
                       r.tried, r.engine_calls + ref.result.engine_calls, ref.result.work, r.refinements, r.diagnosis)
            break
        gs = None                                                  # a second refinement would need the new search's tree
        break
    return r


def _store_proof(memo, key, rb: chess.Board, gs: "GoalSearch", r: Result):
    """Store the proof twice: exactly (every square, castling and en-passant as predicates: a transposition serves
    it as it stands) and as a sketch on the proof's footprint (served as a prior and re-verified)."""
    memo.put("proof", key, rb, chess.BB_ALL, replace(r, served=""), preds=(("castling", ()), ("ep", ())))
    memo.put("sketch", key, rb, gs.proof_footprint(), gs.proof_sketch())


# ---------------------------------------------------------------- the goal loop
def pursue(board: chess.Board, side: chess.Color, memo=None, engine=None, budget: int = 3000, horizon: int = 6,
           max_goals: int = 4, engine_depth: int = 8, refinements: int = 1) -> list[Result]:
    """Try the progress map's targets most valuable first; each gets a budget; a proven goal ends the loop."""
    out = []
    order = {"falls": 0, "joinable": 1, "undermine": 2, "held": 3, "unreachable": 4}
    targets = [t for t in progress(board, side, top=64, memo=memo) if t.status in ("falls", "joinable", "undermine", "held")]
    targets.sort(key=lambda t: (order[t.status], -value(board, t.square)))
    for t in targets[:max_goals]:
        r = search_goal(board, side, t.square, memo=memo, engine=engine, budget=budget, horizon=horizon, engine_depth=engine_depth,
                        refinements=refinements)
        out.append(r)
        if r.proven:
            break
    return out


def result_text(r: Result) -> str:
    if r.proven:
        head = (f"{r.goal}: proven in {r.nodes} nodes, {r.engine_calls} engine checks ({r.seconds:.1f}s): {' '.join(r.line)}"
                + (f"; their longest try: {' '.join(r.work)}" if len(r.work) > len(r.line) else "")
                + (" [served from the store: a transposition]" if r.served == "transposition" else
                   " [the stored sketch, re-verified]" if r.served == "sketch" else ""))
        if r.refinements:
            ref = r.refinements[-1]
            head += f" -- after revising the sketch: {ref.reply} {ref.diagnosis}; {ref.counter} ({', '.join(ref.prior)})"
        return head
    how = f"refuted within the horizon ({r.nodes} nodes)" if r.disproven else f"given up after {r.nodes} nodes"
    out = f"{r.goal}: {how} ({r.seconds:.1f}s); tried {', '.join(r.tried) or 'nothing'}; hardest defence: {' '.join(r.refutation)}"
    for ref in r.refinements:
        out += f"; {ref.reply} {ref.diagnosis}"
        if ref.counter and ref.prior:
            out += f"; revised: {ref.counter} ({', '.join(ref.prior)}): " + (
                ("proven " + " ".join(ref.result.line)) if ref.result and ref.result.proven else
                (f"given up again ({' '.join(ref.result.refutation)})" if ref.result else "no search"))
        elif ref.counter:
            out += f"; {ref.counter}: no move does it"
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True); ap.add_argument("--side", default=""); ap.add_argument("--target", default="", help="square; empty = the goal loop")
    ap.add_argument("--budget", type=int, default=3000); ap.add_argument("--horizon", type=int, default=6); ap.add_argument("--engine", action="store_true")
    ap.add_argument("--trace", action="store_true", help="print the narration, one line per expansion (single target only)")
    a = ap.parse_args(argv)
    b = chess.Board(a.fen); side = b.turn if not a.side else a.side.lower().startswith("w")
    eng = None
    if a.engine:
        from .engine import Engine
        eng = Engine(depth=12)
    from .memo import Memo
    memo = Memo()
    if a.target:
        gs = GoalSearch(b, side, chess.parse_square(a.target), memo=memo, engine=eng, horizon=a.horizon, budget=a.budget)
        gs.verbose = a.trace
        r = gs.run()
        print("\n".join(gs.trace))
        if not r.proven:
            ref = refine(b, side, chess.parse_square(a.target), gs, memo, eng, a.budget, a.horizon)
            if ref is not None:
                r.refinements.append(ref)
        rs = [r]
    else:
        rs = pursue(b, side, memo=memo, engine=eng, budget=a.budget, horizon=a.horizon)
    for r in rs:
        print(result_text(r))
    if eng:
        eng.close()


if __name__ == "__main__":
    main()
