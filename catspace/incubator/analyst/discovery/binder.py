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
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import selfmine as SM

MATE_VALUE = 100
PIECE_NAME = {chess.PAWN: 'pawn', chess.KNIGHT: 'knight', chess.BISHOP: 'bishop', chess.ROOK: 'rook', chess.QUEEN: 'queen', chess.KING: 'king'}


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
    depth: int = 1                    # 1: one named move of ours; 2: a check, every reply, then a named capture (52.5)


@dataclass
class Binding:
    schema: str
    assignment: dict
    value: int
    theorem: str
    san: str
    replies: list = field(default_factory=list)       # SAN of every reply of theirs the tail quantifies over
    tried: list = field(default_factory=list)         # (schema name, value or None, note) for every schema asked, smallest first
    note: str = ""                                    # 52.12: "a fork: they must answer the check" (the trace line carries it; the SAN stays the move's)


def _mate1_atoms(x, y, mover):
    return SM.mate_atoms(x, y, mover)


def _mate2_atoms(x, y, mover):
    return SM.mate2_atoms(x, y, mover)


def _delayed_atoms(family):
    def atoms(x, y, mover):
        with SM.using_mover(mover):
            return SM.with_legality([SM.atom_of("balance", "<=", 0), SM.atom_of(SM.delayed_tail([(x, y)], mover, family, SM.C_VAR), ">=", 1)])
    return atoms


def _delayed_atoms2(family):
    def atoms(x, y, mover):
        raise NotImplementedError("the depth-two schemas read their pair of moves through puzzle_apply.best_bind")
    return atoms


LEGS = ("our piece squares", "legal targets of x")
SCHEMAS = [
    Schema("mate_in_one", ("x", "y"), LEGS, _mate1_atoms, "(after x y movecount) <= 0 and (after x y check) >= 1", MATE_VALUE,
           SM.schema_name(1, SM.MATE), "mate", SM.MATE),
    Schema("mate_in_two", ("x", "y"), LEGS, _mate2_atoms, "(after x y (least (most (min check (eq movecount 0))))) >= 1", MATE_VALUE,
           SM.schema_name(1, SM.MATE2), "mate", SM.MATE2),
] + [Schema(f"{fam}_d1", ("x", "y"), LEGS, _delayed_atoms(fam), "(after x y <" + fam + " tail>) >= balance + 1", 1,
            SM.schema_name(1, fam), "gain", fam) for fam in PA.ORDER]
# The depth-two delayed schemas (52.5, proved in Schemas.lean) are asked only on routes that are CHECKS: our named first move, every reply quantified, our named
# second move (a capture legal after every reply), the exchange reading. They replace the deep tail (the Lean theorem delayed_balance_deep stays).
LEGS2 = ("our piece squares", "legal targets of x that check", "our captures legal after every reply", "their square")
SCHEMAS2 = [Schema(f"{fam}_d2", ("x", "y", "x2", "y2"), LEGS2, _delayed_atoms2(fam), "(after x y <" + fam + " tail over the reply> x2 y2) >= balance + 1", 1,
                   SM.schema_name(2, fam), "gain", fam, 2) for fam in PA.ORDER]
# 52.15 (3): THE KEEP FAMILY. The same Lean theorem as the gain family (`delayed_balance_alive_d1`, general in the bound B and the gain v: hTB : T >= B + v, hCT : C + 4 = T,
# no floor on the balance -- the floor 68 = 64 + 4 is the non-alive `delayed_balance_d1`'s, which the binder does not use) read with B = the balance now and v = 0:
# the atoms are `balance <= B` and the tail >= B (C = B - 4, our material alive at every leaf): after the named move and every reply the balance can be held. It is not tried by
# `verify_tried` (a gain schema asks for >= 1); `verify_keep` asks it of ONE parry.
KEEP_FAMILY = "delayed_balance"
KEEP = Schema("keep_d1", ("x", "y"), LEGS, lambda x, y, mover: None, "(after x y <" + KEEP_FAMILY + " tail>) >= balance + 0", 0, SM.schema_name(1, KEEP_FAMILY), "keep", KEEP_FAMILY)
BY_NAME = {s.name: s for s in SCHEMAS + SCHEMAS2 + [KEEP]}


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


def second_moves(board: chess.Board, route: chess.Move, trap: int | None = None) -> list:
    """The second moves of the depth-two schemas after a CHECK `route`: our captures legal after EVERY reply (intersected by from/to). [] when the route is not a
    check or has no reply. A capture illegal under some reply is left out; the legs would fail on it anyway (best_bind says so).
    52.10: a QUIET route that leaves the piece of theirs on `trap` with no safe square (the regress question `take its squares` names it) has the same legs, the
    second move being our capture of that piece; [] when the piece has a safe square after all (the word `mobility`, read here again)."""
    quiet = not board.gives_check(route)
    if quiet:
        if trap is None:
            return []
        after = board.copy(stack=False)
        after.push(route)
        if after.piece_at(trap) is None or G.mobility_of(after, trap) != 0:
            return []
    c = board.copy(stack=False)
    c.push(route)
    common = None
    for r in list(c.legal_moves):
        d = c.copy(stack=False)
        d.push(r)
        mine = {(chess.square_name(m.from_square), chess.square_name(m.to_square)) for m in d.legal_moves
                if d.is_capture(m) and not m.promotion and not d.is_castling(m) and (not quiet or m.to_square == trap)}
        common = mine if common is None else common & mine
        if not common:
            return []
    return sorted(common or [])


def _read2(s: Schema, fen: str, us: str, pair: tuple, seconds: list) -> tuple:
    """-> (value or None, note, the second move or None) for the depth-two schema on the route `pair`."""
    if not seconds:
        return None, "not a check (or a trap), or no capture of ours is legal after every reply", None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        best = PA.best_bind(2, fen, us, [[pair, sec] for sec in seconds], s.family)
    note = " ".join(buf.getvalue().replace("OBS ", "").split())
    if best is None:
        return None, note or f"the legs do not hold for any of {len(seconds)} capture(s) after {pair[0]}{pair[1]}", None
    B, T, sec = best[1], best[2], best[4][1]
    return ((T - B, "", sec) if T > B else (None, f"reads {T} against {B} now: no gain", sec))


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
    if s.depth == 2:
        for pair in (generate(board) if candidates is None else candidates):
            b2 = verify_tried(chess.Move.from_uci(pair[0] + pair[1]), board, us, schemas=[s])[0]
            if b2 is not None:
                out.append(b2)
        return out
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


def verify_tried(route: chess.Move, board: chess.Board, us: str, mates_only: bool = False, schemas: list | None = None, trap: int | None = None, fork: int | None = None) -> tuple:
    """Try the schemas from the smallest up on ONE route and stop at a mate; otherwise the delayed family reading the largest guaranteed gain wins (ties in the
    ladder's order, as the stream decides; the depth-two schemas come last and only on a check, or on a quiet route that traps the piece on `trap`, 52.10). -> (Binding | None, tried) with tried = [(schema name, value or
    None, note)] for every schema asked."""
    pair = (chess.square_name(route.from_square), chess.square_name(route.to_square))
    fen = board.fen()
    tried, best = [], None
    seconds = None
    for s in (SCHEMAS + SCHEMAS2 if schemas is None else schemas):
        if mates_only and s.kind != "mate":
            continue
        sec = None
        if s.depth == 2:
            if seconds is None:
                seconds = second_moves(board, route, trap)
            val, note, sec = _read2(s, fen, us, pair, seconds)
        else:
            val, note = _read(s, fen, us, pair)
        tried.append((s.name, val, note))
        if val is None or val < s.need:
            continue
        if best is None or val > best.value:
            c = board.copy(stack=False)
            c.push(route)
            asg, san = dict(x=pair[0], y=pair[1]), board.san(route)
            if sec is not None:
                asg.update(x2=sec[0], y2=sec[1])
                san += " then " + PA._san_pairs(fen, [pair, sec])[1].rstrip("+#")      # the check mark depends on the reply
                if not board.gives_check(route) and trap is not None:
                    san += f" (the {PIECE_NAME[board.piece_at(trap).piece_type]} has no safe square)"
            best = Binding(s.name, asg, val, s.theorem, san, [c.san(r) for r in c.legal_moves])
            if sec is not None and fork is not None and board.gives_check(route):
                best.note = "a fork: they must answer the check"
        if s.kind == "mate":
            break
    if best is not None:
        best.tried = tried
    return best, tried


def verify(route: chess.Move, board: chess.Board, us: str, mates_only: bool = False) -> Binding | None:
    return verify_tried(route, board, us, mates_only)[0]


def verify_keep(route: chess.Move, board: chess.Board, us: str, v: int = 0) -> tuple:
    """52.15 (3): does the named move HOLD what we have? The keep family (see KEEP): B = the balance now, T = B + v (v = 0), C = T - 4; the legality legs, `balance <= B`
    and the tail >= T are read in Rust exactly as `puzzle_apply.best_bind` reads a gain. -> (Binding | None, note). The binding's value is v, its theorem the Lean name."""
    s = KEEP
    if route.promotion or board.is_castling(route):                       # the schema's leg exclusions
        return None, "a promotion or castling is not a leg of the schema"
    pair = (chess.square_name(route.from_square), chess.square_name(route.to_square))
    fen = board.fen()
    B = SM.val_at(fen, "balance", us)
    T = B + v
    need = SM.ALIVE_NEED[s.family]
    if T < need:
        return None, f"B + v = {T} is below the alive material {need}: the schema cannot apply"
    with SM.using_mover(us):
        prem = [SM.atom_of("balance", "<=", B), SM.atom_of(SM.delayed_tail([pair], us, s.family, SM.alive_const(T, s.family)), ">=", T)]
    legs = SM.with_legality(prem)[:-len(prem)]
    if legs:
        lterms = list({BV.sexpr(t): t for t, _, _ in legs}.values())
        lvals = SM.evaluate(lterms, [dict(fen=fen, mover=us)])
        if not all((lvals[BV.sexpr(t)][0] >= x) if op == ">=" else (lvals[BV.sexpr(t)][0] <= x) for t, op, x in legs):
            return None, "the legs do not hold"
    if not SM.premise_truth(prem, [dict(fen=fen, mover=us)])[0]:
        return None, f"the balance {B} is not held (the tail reads below {T})"
    c = board.copy(stack=False)
    c.push(route)
    return Binding(s.name, dict(x=pair[0], y=pair[1]), v, s.theorem, board.san(route), [c.san(r) for r in c.legal_moves]), ""
