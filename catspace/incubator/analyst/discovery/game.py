"""plan 71.2 THE GAME INTERFACE (layer 2 of the three: the AXIOMS of a domain, stated, not proved). The machinery (layer 1: claims, read sets, threats, forcing,
parries, defects, the AND-OR proof graph, the frame lemma, tempo arithmetic, the judge, the status rule, the policy, the fact store) speaks to a domain ONLY
through this protocol; chess is the instance (`ChessGame`, python-chess underneath, the Rust core beside it). A second domain -- or a chess variant whose laws
differ -- is a second instance; a variant that keeps the laws (Chess960) is this instance with one parameter.

The vocabulary of the protocol is the machinery's, not chess's: a STATE, the mover's MOVES, APPLY, the PASS (the same state with the other side to move: the
counterfactual every threat word is defined through), TERMINAL (and its outcome for the side), the MEASURE (material here: the status and every claim's value
are differences of it), UNITS (the pieces: location, owner, worth) and LOCATIONS (the squares: what read sets are made of), ATTACKS (which units of a side bear
on a location), IN_CHECK (the forced-reply condition), and NAMES for the trace.

Cheap by construction: states are python-chess boards (no conversion); every method is a one-line delegation; the engine's calls are routed here module by
module (71.2), each step trace-identical (scripts/trace_identity.py)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

import chess

from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import lemma_store as LS

WHITE, BLACK = chess.WHITE, chess.BLACK
MATE_VALUE = LS.goal_kinds().terminal.stake       # plan 75.2 (a): the stake of the goal that ends the game, read from its GOAL entry (lemmas/mate.json: ONE definition)


@dataclass(frozen=True)
class Unit:
    """A piece of a game: where it stands, whose it is, what it is worth, and its kind (a domain token: for chess the python-chess piece type)."""
    loc: int
    owner: bool
    worth: int
    kind: int


@runtime_checkable
class Game(Protocol):
    """The axioms a domain supplies. `State` and `Move` are the domain's own types; the machinery never looks inside them."""

    def moves(self, s) -> list: ...
    def apply(self, s, m): ...
    def turn(self, s) -> bool: ...
    def pass_(self, s): ...
    def terminal(self, s) -> str | None: ...                      # None, or 'mate' (the mover is mated), 'stalemate', 'draw'
    def measure(self, s, side: bool) -> int: ...
    def in_check(self, s) -> bool: ...
    def name(self, s, m) -> str: ...
    def key(self, s) -> str: ...
    def from_key(self, key: str): ...
    def units(self, s, side: bool | None = None) -> list: ...
    def locations(self) -> Iterable[int]: ...
    def loc_name(self, loc: int) -> str: ...
    def attackers(self, s, side: bool, loc: int) -> list: ...
    def is_capture(self, s, m) -> bool: ...
    def gives_check(self, s, m) -> bool: ...
    def unit_at(self, s, loc: int) -> Unit | None: ...
    def coords(self, loc: int) -> tuple: ...                       # a location's coordinates in [0, 1], for features
    def adjacent(self, loc: int) -> Iterable[int]: ...             # the locality graph (what a king's step is in chess)
    def attacks_from(self, s, loc: int) -> Iterable[int]: ...      # the locations the unit on `loc` bears on
    def royal(self, s, side: bool) -> int | None: ...              # the unit whose loss is terminal (the king)
    def kinds(self) -> int: ...                                    # how many unit kinds the domain has (one-hot width)
    def kind_name(self, kind: int) -> str: ...
    def move_key(self, m) -> str: ...                              # a move's name independent of the state (UCI)
    def move_from_key(self, s, key: str): ...
    def move_ends(self, m) -> tuple: ...                           # (from location, to location)
    def promotes(self, m) -> bool: ...


class ChessGame:
    """Chess as the instance: states are `chess.Board`s, moves `chess.Move`s. `chess960` is the one parameter of the laws this instance admits."""

    def __init__(self, chess960: bool = False):
        self.chess960 = chess960

    # the state and its moves
    def moves(self, s: chess.Board) -> list:
        return list(s.legal_moves)

    def apply(self, s: chess.Board, m: chess.Move) -> chess.Board:
        c = s.copy(stack=False)
        c.push(m)
        return c

    def turn(self, s: chess.Board) -> bool:
        return s.turn

    def pass_(self, s: chess.Board) -> chess.Board:
        return G.with_turn(s, not s.turn)

    def terminal(self, s: chess.Board) -> str | None:
        if s.is_checkmate():
            return "mate"
        if s.is_stalemate():
            return "stalemate"
        if s.is_insufficient_material():
            return "draw"
        return None

    def measure(self, s: chess.Board, side: bool) -> int:
        """The material balance for `side` (the sum of the worths of its units less the other side's)."""
        return sum(G.worth(p) * (1 if p.color == side else -1) for p in s.piece_map().values() if p.piece_type != chess.KING)

    def in_check(self, s: chess.Board) -> bool:
        return s.is_check()

    # names and keys
    def name(self, s: chess.Board, m: chess.Move) -> str:
        return s.san(m)

    def key(self, s: chess.Board) -> str:
        return s.fen()

    def from_key(self, key: str) -> chess.Board:
        return chess.Board(key, chess960=self.chess960)

    def loc_name(self, loc: int) -> str:
        return chess.square_name(loc)

    # units and locations
    def units(self, s: chess.Board, side: bool | None = None) -> list:
        return [Unit(sq, p.color, G.worth(p), p.piece_type) for sq, p in s.piece_map().items() if side is None or p.color == side]

    def unit_at(self, s: chess.Board, loc: int) -> Unit | None:
        p = s.piece_at(loc)
        return None if p is None else Unit(loc, p.color, G.worth(p), p.piece_type)

    def locations(self) -> Iterable[int]:
        return chess.SQUARES

    def attackers(self, s: chess.Board, side: bool, loc: int) -> list:
        return list(s.attackers(side, loc))

    def is_capture(self, s: chess.Board, m: chess.Move) -> bool:
        return s.is_capture(m)

    def gives_check(self, s: chess.Board, m: chess.Move) -> bool:
        return s.gives_check(m)

    # geometry and kinds
    def coords(self, loc: int) -> tuple:
        return (chess.square_file(loc) / 7, chess.square_rank(loc) / 7)

    def adjacent(self, loc: int) -> Iterable[int]:
        return chess.SquareSet(chess.BB_KING_ATTACKS[loc])

    def attacks_from(self, s: chess.Board, loc: int) -> Iterable[int]:
        return s.attacks(loc)

    def royal(self, s: chess.Board, side: bool) -> int | None:
        return s.king(side)

    def kinds(self) -> int:
        return 6

    def kind_name(self, kind: int) -> str:
        return chess.piece_name(kind)

    def move_key(self, m: chess.Move) -> str:
        return m.uci()

    def move_from_key(self, s: chess.Board, key: str) -> chess.Move:
        return chess.Move.from_uci(key)

    def move_ends(self, m: chess.Move) -> tuple:
        return (m.from_square, m.to_square)

    def promotes(self, m: chess.Move) -> bool:
        return m.promotion is not None


class KingOfTheHillGame(ChessGame):
    """plan 71.6: a VARIANT BY ITS GOAL (Lean `Ideas.Game.kothGoal`): the laws are chess's; a side also wins the moment its king stands on one of the four centre squares.
    The Game is the same instance with one more terminal; the machinery changes nothing (a goal is a predicate)."""

    CENTRE = (chess.D4, chess.E4, chess.D5, chess.E5)

    def terminal(self, s: chess.Board) -> str | None:
        for side in (chess.WHITE, chess.BLACK):
            k = s.king(side)
            if k is not None and k in self.CENTRE:
                return "loss" if side != s.turn else "win"       # their king stands there with us to move: we have lost; ours: we have won (the mover reads the terminal)
        return super().terminal(s)


CHESS = ChessGame()
CHESS960 = ChessGame(chess960=True)                              # plan 71.6: the same laws, another start (and castling by the kernel's 960 rule): one parameter, not a second instance
KING_OF_THE_HILL = KingOfTheHillGame()
assert isinstance(CHESS, Game) and isinstance(CHESS960, Game) and isinstance(KING_OF_THE_HILL, Game)
