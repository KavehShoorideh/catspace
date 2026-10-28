"""plan 76.0 (b) THE CHESS BINDING: the fast kernel's state as the spec's facts, the spec's moves as the kernel's notation -- the one place where the GDL interpreter
(`gdl.py`, which knows no game) meets python-chess. Everything here is a reading of rules/chess.gdl's state vocabulary:

  (cell ?f ?r ?c ?k)          python-chess `Board.piece_map()` (files a..h, ranks 1..8, colours white / black, kinds by `chess.piece_name`)
  (control ?c)                `Board.turn`
  (castling_right ?c ?side)   `Board.clean_castling_rights()` (the rook squares h1 / a1 / h8 / a8 -> kingside / queenside)
  (ep_square ?f ?r)           `Board.ep_square` -- the RAW square (set after every double push, legal capture or not: what the spec's `next` sets)
  (halfmove ?n)               min(`Board.halfmove_clock`, 100) -- the spec's clock saturates at its limit (the rule reads only "100 or more")

and the moves: (move f1 r1 f2 r2) / (promote f1 r1 f2 r2 k) are the UCI text (castling the king's two-file move, en passant the pawn's diagonal move). The fullmove
number is no rule: it is not in the state.
"""
from __future__ import annotations

from functools import lru_cache

import chess

from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.io import paths

SPEC = paths.REPO_ROOT / "rules" / "chess.gdl"          # the spec (a tracked file of the repo, not data)
CLOCK_LIMIT = 100
COLOUR = {chess.WHITE: "white", chess.BLACK: "black"}
RIGHT = {chess.H1: ("white", "kingside"), chess.A1: ("white", "queenside"), chess.H8: ("black", "kingside"), chess.A8: ("black", "queenside")}
PROMO = {"q": "queen", "r": "rook", "b": "bishop", "n": "knight"}
PROMO_LETTER = {v: k for k, v in PROMO.items()}


@lru_cache(maxsize=1)
def program() -> GDL.Program:
    """rules/chess.gdl, loaded once (its static relations evaluated)."""
    return GDL.load(SPEC)


def sq(s: int) -> tuple:
    return (chess.FILE_NAMES[chess.square_file(s)], str(chess.square_rank(s) + 1))


def state_of(board: chess.Board, ep: "int | None | str" = "raw") -> frozenset:
    """The spec's state for a python-chess board. `ep`: 'raw' -- `board.ep_square`; or an explicit square / None (another kernel's raw square)."""
    facts = set()
    for s, p in board.piece_map().items():
        f, r = sq(s)
        facts.add(("true.cell", (f, r, COLOUR[p.color], chess.piece_name(p.piece_type))))
    facts.add(("true.control", (COLOUR[board.turn],)))
    for s in chess.SquareSet(board.clean_castling_rights()):
        c, side = RIGHT[s]
        facts.add(("true.castling_right", (c, side)))
    e = board.ep_square if ep == "raw" else ep
    if e is not None:
        facts.add(("true.ep_square", sq(e)))
    facts.add(("true.halfmove", (str(min(board.halfmove_clock, CLOCK_LIMIT)),)))
    return frozenset(facts)


def move_term(uci: str) -> tuple:
    f1, r1, f2, r2 = uci[0], uci[1], uci[2], uci[3]
    if len(uci) == 5:
        return ("promote", f1, r1, f2, r2, PROMO[uci[4]])
    return ("move", f1, r1, f2, r2)


def uci_of(term) -> str:
    if term[0] == "promote":
        return "".join(term[1:5]) + PROMO_LETTER[term[5]]
    if term[0] == "move":
        return "".join(term[1:5])
    raise ValueError(f"not a move term: {term}")


def fact_text(f) -> str:
    rel, args = f
    name = rel.split(".", 1)[1] if rel.startswith("true.") else rel
    return "(" + " ".join((name,) + tuple(GDL.term_text(a) for a in args)) + ")"


class Reading:
    """The spec's answers at one state: the mover's legal moves (UCI), the terminal verdict, the terminal reasons that hold, the goal values."""

    REASONS = ("mate", "stalemate", "draw", "draw_by_fifty_moves")

    def __init__(self, state: frozenset, prog: "GDL.Program | None" = None):
        prog = prog or program()
        self.state = state
        self.db = prog.run(state)
        mover = next(a[0] for rel, a in state if rel == "true.control")
        self.mover = mover
        self.legal = sorted(uci_of(m) for role, m in self.db.facts("legal") if role == mover and m != "noop")
        self.terminal = prog.holds(self.db, "terminal")
        self.reasons = tuple(r for r in self.REASONS if prog.holds(self.db, r))
        self.goals = {role: sorted(vs) for role, vs in prog.goals(self.db).items()}
        self.check = (mover,) in self.db.facts("check")


def next_state(state: frozenset, uci: str, mover: str, prog: "GDL.Program | None" = None) -> frozenset:
    prog = prog or program()
    other = "black" if mover == "white" else "white"
    return prog.next_state(state, [(mover, move_term(uci)), (other, "noop")])


# the kernel's terminal kinds (Game.terminal) and the spec's reasons, by name
KERNEL_KIND = {"mate": "mate", "stalemate": "stalemate", "draw": "draw", "fifty-move": "draw_by_fifty_moves"}
KERNEL_ORDER = ("mate", "stalemate", "draw", "draw_by_fifty_moves")      # Game.terminal's order (Lean `Chess.gameStatus`: a mate on the move first)


def kernel_reasons(board: chess.Board) -> tuple:
    """python-chess's terminal predicates, each read on its own (not only the first): the reasons that hold."""
    out = []
    if board.is_checkmate():
        out.append("mate")
    if board.is_stalemate():
        out.append("stalemate")
    if board.is_insufficient_material():
        out.append("draw")
    if board.is_fifty_moves():
        out.append("draw_by_fifty_moves")
    return tuple(out)


def goals_of(kind: "str | None", mover: str) -> dict:
    """The goal values of a kernel verdict (the spec's: the mated side 0, the other 100; a draw 50 each; not terminal: none)."""
    other = "black" if mover == "white" else "white"
    if kind is None:
        return {}
    if kind == "mate":
        return {mover: ["0"], other: ["100"]}
    return {mover: ["50"], other: ["50"]}
