"""L3 -- STATIC RELATIONS, indexed by ARITY. A relation of arity k is a fact over k pieces or
squares that holds on the CURRENT board and is defined ONLY through L1 probes (attack = "could
capture there by the rules"). No objective, no valence: a pin is a pin in antichess. Arity is
also the fact's locality/complexity -- what a learner must see at once.

Pins, skewers, discovered attacks and batteries are found by the REMOVAL TEST rather than by
walking chess-specific rays: piece S attacks P1; remove P1; the squares S now newly attacks are
what P1 was shielding. This works for any sliding rule the variant defines and yields nothing
for non-sliders, automatically."""
from __future__ import annotations

from dataclasses import dataclass

import chess

from . import rules as R
from .facts import value

# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Relation:
    name: str
    arity: int
    squares: tuple[int, ...]      # pieces/squares in role order (actor first)
    owner: bool | None            # color whose piece performs the relation (None = symmetric)
    tags: tuple[str, ...] = ()
    pieces: tuple[str, ...] = ()  # piece symbols at `squares` when the relation was observed

    @property
    def key(self) -> tuple:
        return (self.name, self.squares, self.owner, self.tags)

    def names(self) -> tuple[str, ...]:
        return tuple(chess.square_name(s) for s in self.squares)


def _rel(board: chess.Board, name: str, arity: int, squares: tuple, owner, tags: tuple = ()) -> Relation:
    return Relation(name, arity, tuple(squares), owner, tuple(tags),
                    tuple(board.piece_at(q).symbol() if board.piece_at(q) else "-" for q in squares))


def _enemy(board: chess.Board, sq: int, color: chess.Color) -> bool:
    p = board.piece_at(sq)
    return p is not None and p.color != color


def _friend(board: chess.Board, sq: int, color: chess.Color) -> bool:
    p = board.piece_at(sq)
    return p is not None and p.color == color


# ---- arity 2: attacks / defends ---------------------------------------------------------------
def attacks_and_defends(board: chess.Board) -> tuple[list[Relation], list[Relation]]:
    atk, dfd = [], []
    for sq, piece in board.piece_map().items():
        for a in board.attackers(not piece.color, sq):
            atk.append(_rel(board, "attacks", 2, (a, sq), not piece.color))
        for d in board.attackers(piece.color, sq):
            dfd.append(_rel(board, "defends", 2, (d, sq), piece.color))
    return atk, dfd


# ---- arity 1/2: hanging, loose, attacked by a lower-valued piece -------------------------------
def hanging_and_loose(board: chess.Board) -> list[Relation]:
    out = []
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING and R.is_royal(board, sq):
            continue
        attackers = board.attackers(not piece.color, sq)
        defenders = board.attackers(piece.color, sq)
        if attackers and not defenders:
            out.append(_rel(board, "hanging", 1, (sq,), piece.color))
        elif not attackers and not defenders and piece.piece_type != chess.PAWN:
            out.append(_rel(board, "loose", 1, (sq,), piece.color))
        for a in attackers:
            if value(board, a) < value(board, sq):
                out.append(_rel(board, "attacked_by_lower_value", 2, (a, sq), not piece.color))
    return out


# ---- arity 3: pins, skewers, discovered attacks, batteries (removal test) --------------------
def line_relations(board: chess.Board) -> list[Relation]:
    out = []
    for s, piece in board.piece_map().items():
        c = piece.color
        before = board.attacks(s)
        for p1 in list(before):
            if board.piece_at(p1) is None:
                continue
            b2 = board.copy(stack=False)
            b2.remove_piece_at(p1)
            gained = b2.attacks(s) - before
            for p2 in gained:
                q = board.piece_at(p2)
                if q is None:
                    continue
                p1_enemy, p2_enemy = _enemy(board, p1, c), _enemy(board, p2, c)
                if p1_enemy and p2_enemy:
                    v1, v2 = value(board, p1), value(board, p2)
                    if R.is_royal(board, p2):
                        out.append(_rel(board, "pin", 3, (s, p1, p2), c, ("absolute",)))
                    elif v2 > v1:
                        out.append(_rel(board, "pin", 3, (s, p1, p2), c, ("relative",)))
                    elif v1 > v2:
                        out.append(_rel(board, "skewer", 3, (s, p1, p2), c, ("check",) if R.is_royal(board, p1) else ()))
                    else:
                        out.append(_rel(board, "xray", 3, (s, p1, p2), c))
                elif _friend(board, p1, c) and p2_enemy:
                    if p2 in board.attacks(p1):
                        out.append(_rel(board, "battery", 3, (s, p1, p2), c))
                    else:
                        out.append(_rel(board, "discovered_attack", 3, (s, p1, p2), c,
                                        ("check",) if R.is_royal(board, p2) else ()))
    return out


# ---- arity 1+k: forks -----------------------------------------------------------------------
def forks(board: chess.Board) -> list[Relation]:
    out = []
    for s, piece in board.piece_map().items():
        c, va = piece.color, value(board, s)
        targets = []
        for t in board.attacks(s):
            if not _enemy(board, t, c):
                continue
            vt = value(board, t)
            undefended = not board.attackers(not c, t)
            if R.is_royal(board, t) or vt > va or undefended:
                targets.append(t)
        if len(targets) >= 2:
            out.append(_rel(board, "fork", 1 + len(targets), (s, *sorted(targets)), c))
    return out


# ---- arity 1+n: overloaded defenders ----------------------------------------------------------
def overloaded(board: chess.Board) -> list[Relation]:
    out = []
    for d, piece in board.piece_map().items():
        c = piece.color
        duties = []
        for t in board.attacks(d):
            if not _friend(board, t, c):
                continue
            if board.attackers(not c, t) and board.attackers(c, t) == chess.SquareSet([d]):
                duties.append(t)
        if len(duties) >= 2:
            out.append(_rel(board, "overloaded", 1 + len(duties), (d, *sorted(duties)), c))
    return out


# ---- arity 2: pressure on the royal king's zone (zone = the king's own move set, via L1) -----
def king_zone_pressure(board: chess.Board) -> list[Relation]:
    out = []
    for c in (chess.WHITE, chess.BLACK):
        k = R.royal_square(board, c)
        if k is None:
            continue
        zone = board.attacks(k) | chess.SquareSet([k])
        for e, piece in board.piece_map().items():
            if piece.color == c:
                continue
            n = len(board.attacks(e) & zone)
            if n:
                out.append(_rel(board, "attacks_king_zone", 2, (e, k), not c, (str(n),)))
    return out


# ---- arity 4: COMPOSED relations -- a pin plus one more piece (Kaveh 2026-09-04) -------------
# A functional pin is a FOUR-piece interaction: pinner S, pinned P1, shielded P2, and the fourth
# piece T that P1 could act on and now cannot. The STRUCTURE lives here, static and rules-only;
# the EFFECT of the disabling (illegal vs merely costly, good vs bad) lives in L4/L5 because it
# differs by rules and objective. Composition = an arity-3 relation joined with one more L1 probe.
def composed_pin_relations(board: chess.Board, pins: list[Relation] | None = None) -> list[Relation]:
    out = []
    pins = [r for r in (pins if pins is not None else line_relations(board)) if r.name == "pin"]
    for r in pins:
        s, p1, p2 = r.squares
        c = board.piece_at(p1).color
        line = chess.SquareSet.between(s, p2) | chess.SquareSet([s])   # moves along the line keep the shield
        for t in board.attacks(p1):
            if t in line:
                continue
            q = board.piece_at(t)
            if q is None:
                continue
            if q.color != c:
                out.append(_rel(board, "disabled_attack", 4, (s, p1, p2, t), r.owner, r.tags))
            elif board.attackers(not c, t) and not R.is_royal(board, t):   # kings are not "defended"
                out.append(_rel(board, "disabled_defense", 4, (s, p1, p2, t), r.owner, r.tags))
        for a in board.attackers(not c, p1):
            if a != s:
                out.append(_rel(board, "pinned_and_attacked", 4, (s, p1, p2, a), r.owner, r.tags))
    return out


def hanging_through_pin(board: chess.Board, composed: list[Relation]) -> list[Relation]:
    """A piece whose EVERY defender is a pinned piece that cannot recapture off its line."""
    out = []
    dead = {}                                                   # (defender, defended) pairs
    for r in composed:
        if r.name == "disabled_defense":
            dead.setdefault(r.squares[3], set()).add(r.squares[1])
    for t, piece in board.piece_map().items():
        if not board.attackers(not piece.color, t):
            continue
        defenders = set(board.attackers(piece.color, t))
        if defenders and defenders <= dead.get(t, set()):
            out.append(_rel(board, "hanging_through_pin", 1 + len(defenders), (t, *sorted(defenders)), piece.color))
    return out


def relations(board: chess.Board) -> list[Relation]:
    atk, dfd = attacks_and_defends(board)
    lines = line_relations(board)
    composed = composed_pin_relations(board, lines)
    return (atk + dfd + hanging_and_loose(board) + lines + composed + hanging_through_pin(board, composed)
            + forks(board) + overloaded(board) + king_zone_pressure(board))


def relation_keys(board: chess.Board) -> set[tuple]:
    return {r.key for r in relations(board)}
