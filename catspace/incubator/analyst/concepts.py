"""THE CONCEPT REGISTRY (Kaveh 2026-09-20): concepts are PROGRAMS over the rules' vocabulary, versioned and
append-only. A concept has a name, an arity with typed arguments (pieces, squares, lines), and a program that,
given a board (and the graph, regions, ledger already computed), yields its INSTANCES: the bindings where it holds,
each with an optional scalar. An instance is a token for the network: the concept, its arguments, its value.
Round zero = what the rules already compute, written as concepts: the graph's node answers (unary), its pair
relations (binary), the region relations (binary), the position indicators (nullary), and the LINE predicates
(a piece and one of its lines: what else stands on it, how far it runs) -- "no other piece on this rook's file"
is an instance of one. A concept is canonical because it is code; explanations cite the registry version.
Later rounds ADD concepts proposed against the residual and admitted by the gauges; nothing is ever edited."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import chess

from .graph import NODE_QUESTIONS, EDGE_RELATIONS, POSITION_QUESTIONS, Graph, build
from .memo import Memo
from .regions import regions as _regions, PAIR_RELATIONS, UNARY, pair_relations, unary_relations

REGISTRY_VERSION = "0"
DIRECTIONS = ("file", "rank", "diag_up", "diag_down")             # the four lines through a square


@dataclass
class Instance:
    concept: str
    args: tuple                              # squares (pieces) and/or line ids, in the concept's argument order
    value: float = 1.0
    line: int = -1                           # for line concepts: the direction index

    def text(self, board: chess.Board) -> str:
        def who(sq):
            p = board.piece_at(sq)
            return f"{chess.piece_name(p.piece_type)} {chess.square_name(sq)}" if p else chess.square_name(sq)
        a = ", ".join(who(s) for s in self.args)
        ln = f" [{DIRECTIONS[self.line]}]" if self.line >= 0 else ""
        return f"{self.concept}({a}){ln}" + (f" = {self.value:g}" if self.value != 1.0 else "")


@dataclass
class Concept:
    name: str
    arity: int
    kind: str                                # "node" | "pair" | "region_pair" | "region_unary" | "position" | "line"
    version: str = REGISTRY_VERSION
    doc: str = ""
    program: Callable | None = None          # (ctx) -> list[Instance]


@dataclass
class Context:
    board: chess.Board
    graph: Graph
    regions: dict
    idx: dict = field(default_factory=dict)  # square -> node index

    @classmethod
    def of(cls, board: chess.Board, memo: Memo | None = None):
        g = build(board, memo if memo is not None else Memo())
        return cls(board, g, _regions(board), g.index())


# ---------------------------------------------------------------- round zero programs
def _node_program(q: str):
    def prog(ctx: Context) -> list[Instance]:
        out = []
        for n in ctx.graph.nodes:
            v = float(n.answers.get(q, 0.0))
            if v != 0.0:
                out.append(Instance(q, (n.square,), v))
        return out
    return prog


def _pair_program(r: str):
    def prog(ctx: Context) -> list[Instance]:
        out = []
        for (i, j), rel in ctx.graph.edges.items():
            v = float(rel.get(r, 0.0))
            if v != 0.0:
                out.append(Instance(r, (ctx.graph.nodes[i].square, ctx.graph.nodes[j].square), v))
        return out
    return prog


def _region_pair_program(k: int, name: str):
    def prog(ctx: Context) -> list[Instance]:
        out = []
        for a, ra in ctx.regions.items():
            for b, rb in ctx.regions.items():
                if a != b and pair_relations(ra, rb)[k]:
                    out.append(Instance(name, (a, b)))
        return out
    return prog


def _region_unary_program(k: int, name: str):
    def prog(ctx: Context) -> list[Instance]:
        out = []
        for s, r in ctx.regions.items():
            v = float(unary_relations(r, ctx.board)[k])
            if v != 0.0:
                out.append(Instance(name, (s,), v))
        return out
    return prog


def _position_program(q: str):
    def prog(ctx: Context) -> list[Instance]:
        v = float(ctx.graph.position.get(q, 0.0))
        return [Instance(q, (), v)] if v != 0.0 else []
    return prog


def line_squares(sq: int, d: int) -> list[int]:
    f, r = chess.square_file(sq), chess.square_rank(sq)
    if d == 0:
        return [chess.square(f, rr) for rr in range(8) if rr != r]
    if d == 1:
        return [chess.square(ff, r) for ff in range(8) if ff != f]
    out = []
    for k in range(-7, 8):
        if k == 0:
            continue
        ff, rr = f + k, r + (k if d == 2 else -k)
        if 0 <= ff < 8 and 0 <= rr < 8:
            out.append(chess.square(ff, rr))
    return out


def _line_program(name: str, fn):
    """A line concept: for each piece that moves along lines (rook, bishop, queen, and pawns on their file), and
    each of its directions, fn(board, sq, d, squares_on_line) -> value or 0."""
    def prog(ctx: Context) -> list[Instance]:
        out = []
        b = ctx.board
        for sq, p in b.piece_map().items():
            dirs = {chess.ROOK: (0, 1), chess.BISHOP: (2, 3), chess.QUEEN: (0, 1, 2, 3), chess.PAWN: (0,), chess.KING: (0, 1, 2, 3), chess.KNIGHT: ()}[p.piece_type]
            for d in dirs:
                v = fn(b, sq, d, line_squares(sq, d))
                if v:
                    out.append(Instance(name, (sq,), float(v), line=d))
        return out
    return prog


def _others_on_line(board, sq, d, squares, own: bool | None, pawns: bool | None) -> int:
    p = board.piece_at(sq); n = 0
    for t in squares:
        q = board.piece_at(t)
        if q is None:
            continue
        if own is not None and (q.color == p.color) != own:
            continue
        if pawns is not None and (q.piece_type == chess.PAWN) != pawns:
            continue
        n += 1
    return n


def _reach(board, sq, d, squares) -> int:
    """How many squares the piece's line runs before a piece blocks it, summed over the two ways."""
    p = board.piece_at(sq); f, r = chess.square_file(sq), chess.square_rank(sq)
    step = {0: (0, 1), 1: (1, 0), 2: (1, 1), 3: (1, -1)}[d]
    total = 0
    for sgn in (1, -1):
        ff, rr = f + sgn * step[0], r + sgn * step[1]
        while 0 <= ff < 8 and 0 <= rr < 8:
            q = board.piece_at(chess.square(ff, rr))
            total += 1
            if q is not None:
                break
            ff += sgn * step[0]; rr += sgn * step[1]
    return total


ROUND_ZERO: list[Concept] = (
    [Concept(q, 1, "node", doc=f"the graph's node answer {q}", program=_node_program(q)) for q in NODE_QUESTIONS]
    + [Concept(r, 2, "pair", doc=f"the graph's pair relation {r}", program=_pair_program(r)) for r in EDGE_RELATIONS]
    + [Concept(f"region_{n}", 2, "region_pair", doc=f"regions: {n}", program=_region_pair_program(k, f"region_{n}")) for k, n in enumerate(PAIR_RELATIONS)]
    + [Concept(f"region_{n}", 1, "region_unary", doc=f"regions: {n}", program=_region_unary_program(k, f"region_{n}")) for k, n in enumerate(UNARY)]
    + [Concept(f"pos_{q}", 0, "position", doc=f"the position indicator {q}", program=_position_program(f"pos_{q}")) for q in POSITION_QUESTIONS]
    + [Concept("line_own_pieces", 1, "line", doc="own non-pawn pieces on the piece's line", program=_line_program("line_own_pieces", lambda b, s, d, sqs: _others_on_line(b, s, d, sqs, True, False))),
       Concept("line_enemy_pieces", 1, "line", doc="enemy non-pawn pieces on the piece's line", program=_line_program("line_enemy_pieces", lambda b, s, d, sqs: _others_on_line(b, s, d, sqs, False, False))),
       Concept("line_own_pawns", 1, "line", doc="own pawns on the piece's line", program=_line_program("line_own_pawns", lambda b, s, d, sqs: _others_on_line(b, s, d, sqs, True, True))),
       Concept("line_enemy_pawns", 1, "line", doc="enemy pawns on the piece's line", program=_line_program("line_enemy_pawns", lambda b, s, d, sqs: _others_on_line(b, s, d, sqs, False, True))),
       Concept("line_empty", 1, "line", doc="no other piece on the piece's line (an open line for it)", program=_line_program("line_empty", lambda b, s, d, sqs: int(_others_on_line(b, s, d, sqs, None, None) == 0))),
       Concept("line_reach", 1, "line", doc="how far the line runs before a piece blocks it, both ways", program=_line_program("line_reach", _reach))]
)


# ---------------------------------------------------------------- round one (2026-09-21, from the Be6 residual): DISCOVERY
def _discovery_program(ctx: Context) -> list[Instance]:
    """discovery(A, B): A stands on the line between B (own slider) and the enemy king, alone; A's moving off the line
    gives check from B. The instance's value is the number of A's legal moves that leave the line (the family's size)."""
    out = []
    b = ctx.board
    for colour in (chess.WHITE, chess.BLACK):
        ek = b.king(not colour)
        if ek is None:
            continue
        for bsq, bp in b.piece_map().items():
            if bp.color != colour or bp.piece_type not in (chess.QUEEN, chess.ROOK, chess.BISHOP):
                continue
            between = chess.SquareSet.between(bsq, ek)
            if not between and not (bsq in b.attacks(ek) and False):
                continue
            df, dr = chess.square_file(ek) - chess.square_file(bsq), chess.square_rank(ek) - chess.square_rank(bsq)
            straight = df == 0 or dr == 0; diag = abs(df) == abs(dr)
            if not ((straight and bp.piece_type in (chess.QUEEN, chess.ROOK)) or (diag and bp.piece_type in (chess.QUEEN, chess.BISHOP))):
                continue
            blockers = [s for s in between if b.piece_at(s) is not None]
            if len(blockers) != 1 or b.piece_at(blockers[0]).color != colour:
                continue
            a = blockers[0]
            bb = b.copy(stack=False); bb.turn = colour
            n = sum(1 for m in bb.legal_moves if m.from_square == a and m.to_square not in between) if bb.is_valid() else 0
            if n:
                out.append(Instance("discovery", (a, bsq), float(n)))
    return out


ROUND_ONE: list[Concept] = ROUND_ZERO + [Concept("discovery", 2, "pair", version="1", doc="A shields the enemy king from own slider B: A's moves off the line discover check (value: how many)", program=_discovery_program)]
VERSIONS = {"0": ROUND_ZERO, "1": ROUND_ONE}


def _fix_position_names():
    # position programs read graph.position by the bare question name
    for c in ROUND_ZERO:
        if c.kind == "position":
            q = c.name[4:]
            def prog(ctx: Context, q=q, name=c.name) -> list[Instance]:
                v = float(ctx.graph.position.get(q, 0.0))
                return [Instance(name, (), v)] if v != 0.0 else []
            c.program = prog
_fix_position_names()


class Registry:
    def __init__(self, concepts: list[Concept] | None = None, version: str = REGISTRY_VERSION):
        self.concepts = list(concepts if concepts is not None else VERSIONS[version])
        self.by_name = {c.name: c for c in self.concepts}
        self.version = version

    def add(self, concept: Concept):
        assert concept.name not in self.by_name, "concepts are never edited or replaced"
        self.concepts.append(concept); self.by_name[concept.name] = concept

    def instances(self, board: chess.Board, memo: Memo | None = None, kinds: tuple | None = None) -> list[Instance]:
        ctx = Context.of(board, memo)
        out = []
        for c in self.concepts:
            if kinds and c.kind not in kinds:
                continue
            out.extend(c.program(ctx))
        return out

    def summary(self) -> str:
        from collections import Counter
        k = Counter(c.kind for c in self.concepts)
        return f"registry v{self.version}: {len(self.concepts)} concepts " + ", ".join(f"{n} {v}" for n, v in k.items())


if __name__ == "__main__":
    import sys
    reg = Registry(); print(reg.summary())
    b = chess.Board(sys.argv[1] if len(sys.argv) > 1 else "4k3/1R6/4K3/4P3/8/8/8/5r2 b - - 0 58")
    inst = reg.instances(b)
    print(len(inst), "instances")
    for i in inst:
        if i.concept.startswith("line_") or i.concept.startswith("region_safe_meets") or i.concept in ("attacks", "through", "aligned"):
            print(" ", i.text(b))
