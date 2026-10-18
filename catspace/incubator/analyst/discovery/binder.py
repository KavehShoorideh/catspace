"""THE GENERIC BINDER (ANALYST_PLAN 52.1) = the verifier. A SCHEMA is data: variables typed by legs, a premise over words, a tail, the Lean theorem's name.
BINDING: the legs GENERATE assignments (x over our piece squares, y over x's legal targets), the atoms prune under one `after`, the tail is read last; a
surviving assignment is a binding whose guarantee is the theorem's conclusion; the Lean instance is the receipt. Memory never adds or removes a binding.

The premises are NOT retyped: they come from selfmine's renderers (`mate_atoms`, `mate2_atoms`, `delayed_tail`, `tail_body`), read in Rust with
`premise_truth` / `val_at` exactly as `selfmine.bind_*` and `puzzle_apply.best_bind` read them. The delayed families' two-step reading (the leaf number with
a large C, then with C = T - ALIVE_NEED, our material alive at every leaf) is `puzzle_apply.best_bind`'s, called on one candidate.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import Callable

import chess

from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import selfmine as SM

MATE_VALUE = 100


@dataclass
class Schema:
    name: str                         # the stream's name (mate_in_one, delayed_root_d1 ...)
    vars: tuple                       # ("x", "y")
    legs: tuple                       # what generates each variable ("our piece squares", "legal targets of x")
    atoms: Callable                   # (x, y, mover) -> [(term, op, v)] : the premise, rendered by selfmine
    tail: str                         # the tail, in words (for the trace)
    need: int                         # the least value of a binding (a mate reads 100, a delayed family a gain of 1)
    theorem: str                      # the Lean theorem the binding instantiates
    kind: str                         # 'mate' (premise_truth decides, value 100) | 'gain' (best_bind's two readings, value = T - B)
    family: str                       # selfmine's family name


@dataclass
class Binding:
    schema: str
    assignment: dict
    value: int
    theorem: str
    san: str
    replies: list = field(default_factory=list)       # SAN of every reply of theirs the tail quantifies over
    tried: list = field(default_factory=list)         # (schema name, value or None, note) for every schema asked, smallest first


def _mate1_atoms(x, y, mover):
    return SM.mate_atoms(x, y, mover)


def _mate2_atoms(x, y, mover):
    return SM.mate2_atoms(x, y, mover)


def _delayed_atoms(family):
    def atoms(x, y, mover):
        with SM.using_mover(mover):
            return SM.with_legality([SM.atom_of("balance", "<=", 0), SM.atom_of(SM.delayed_tail([(x, y)], mover, family, SM.C_VAR), ">=", 1)])
    return atoms


LEGS = ("our piece squares", "legal targets of x")
SCHEMAS = [
    Schema("mate_in_one", ("x", "y"), LEGS, _mate1_atoms, "(after x y movecount) <= 0 and (after x y check) >= 1", MATE_VALUE,
           SM.schema_name(1, SM.MATE), "mate", SM.MATE),
    Schema("mate_in_two", ("x", "y"), LEGS, _mate2_atoms, "(after x y (least (most (min check (eq movecount 0))))) >= 1", MATE_VALUE,
           SM.schema_name(1, SM.MATE2), "mate", SM.MATE2),
] + [Schema(f"{fam}_d1", ("x", "y"), LEGS, _delayed_atoms(fam), "(after x y <" + fam + " tail>) >= balance + 1", 1,
            SM.schema_name(1, fam), "gain", fam) for fam in PA.ORDER]
BY_NAME = {s.name: s for s in SCHEMAS}


def schema(name: str) -> Schema:
    return BY_NAME[name]


def generate(board: chess.Board) -> list:
    """The legs: every (x, y) with x one of our pieces and y a legal target of x (castling and promotions are the schemas' leg exclusions), in the board's order."""
    seen, out = set(), []
    for m in board.legal_moves:
        if m.promotion or board.is_castling(m):
            continue
        pr = (chess.square_name(m.from_square), chess.square_name(m.to_square))
        if pr not in seen:
            seen.add(pr)
            out.append(pr)
    return out


def _read(s: Schema, fen: str, us: str, pair: tuple) -> tuple:
    """-> (value or None, note). The atoms prune, the tail is read last."""
    x, y = pair
    if s.kind == "mate":
        with SM.using_mover(us):
            prem = [SM.atom_of(("after", x, y, "movecount"), "<=", 0), SM.atom_of(("after", x, y, "check"), ">=", 1)] if s.family == SM.MATE else \
                [SM.atom_of(SM.mate2_tail(x, y), ">=", 1)]
        ok = SM.premise_truth(prem, [dict(fen=fen, mover=us)])[0]
        return (MATE_VALUE if ok else None), ""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        best = PA.best_bind(1, fen, us, [[pair]], s.family)
    note = " ".join(buf.getvalue().replace("OBS ", "").split())
    if best is None:
        return None, note or "the legs or the tail do not hold"
    _san, B, T = best[0], best[1], best[2]
    return (T - B, "") if T > B else (None, f"reads {T} against {B} now: no gain")


def bind(s: Schema, board: chess.Board, us: str, candidates: list | None = None) -> list:
    """Every assignment of the schema's variables on this board whose premise and tail hold. `candidates` = pairs to try (default: the legs generate them). The
    mate_in_two family STOPS at its first binding, as selfmine.bind_mate2 does (no mate in two beats another)."""
    fen = board.fen()
    out = []
    for pair in (generate(board) if candidates is None else candidates):
        val, _note = _read(s, fen, us, pair)
        if val is not None and val >= s.need:
            m = chess.Move.from_uci(pair[0] + pair[1])
            c = board.copy(stack=False)
            c.push(m)
            out.append(Binding(s.name, dict(x=pair[0], y=pair[1]), val, s.theorem, board.san(m), [c.san(r) for r in c.legal_moves]))
            if s.name == "mate_in_two":
                break
    return out


def verify_tried(route: chess.Move, board: chess.Board, us: str, mates_only: bool = False) -> tuple:
    """Try the schemas from the smallest up on ONE route and stop at a mate; otherwise the delayed family reading the largest guaranteed gain wins (ties in the
    ladder's order, as the stream decides). -> (Binding | None, tried) with tried = [(schema name, value or None, note)] for every schema asked."""
    pair = (chess.square_name(route.from_square), chess.square_name(route.to_square))
    fen = board.fen()
    tried, best = [], None
    for s in SCHEMAS:
        if mates_only and s.kind != "mate":
            continue
        val, note = _read(s, fen, us, pair)
        tried.append((s.name, val, note))
        if val is None or val < s.need:
            continue
        if best is None or val > best.value:
            c = board.copy(stack=False)
            c.push(route)
            best = Binding(s.name, dict(x=pair[0], y=pair[1]), val, s.theorem, board.san(route), [c.san(r) for r in c.legal_moves])
        if s.kind == "mate":
            break
    if best is not None:
        best.tried = tried
    return best, tried


def verify(route: chess.Move, board: chess.Board, us: str, mates_only: bool = False) -> Binding | None:
    return verify_tried(route, board, us, mates_only)[0]
