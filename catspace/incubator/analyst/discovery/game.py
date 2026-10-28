"""plan 71.2 THE GAME INTERFACE (layer 2 of the three: the AXIOMS of a domain, stated, not proved). The machinery (layer 1: claims, read sets, threats, forcing,
parries, defects, the AND-OR proof graph, the frame lemma, tempo arithmetic, the judge, the status rule, the policy, the fact store) speaks to a domain ONLY
through this protocol; chess is the instance (`ChessGame`, python-chess underneath, the Rust core beside it). A second domain -- or a chess variant whose laws
differ -- is a second instance; a variant that keeps the laws (Chess960) is this instance with one parameter.

The vocabulary of the protocol is the machinery's, not chess's: a STATE, the mover's MOVES, APPLY, the PASS (the same state with the other side to move: the
counterfactual every threat word is defined through), TERMINAL (and its outcome for the side), the MEASURE (material here: the status and every claim's value
are differences of it), UNITS (the pieces: location, owner, worth) and LOCATIONS (the squares: what read sets are made of), ATTACKS (which units of a side bear
on a location), IN_CHECK (the forced-reply condition), and NAMES for the trace.

Cheap by construction: states are python-chess boards (no conversion); every method is a one-line delegation; the engine's calls are routed here module by
module (71.2), each step trace-identical (scripts/trace_identity.py).

plan 75.4 (1) THE GAME CONTEXT IS A FIRST-CLASS ARGUMENT (Kaveh 2026-10-12: "every frame carries the 50-move clock and the triple position"). A state's IDENTITY
(`position_key`: what a repetition repeats -- for chess the board, the side to move, the castling rights and an en-passant square only when the capture is LEGAL,
Lean `Chess.repetitionKey`) and the GAME RECORD (`Record`: how often each identity occurred before the state, Lean `Ideas.Game.Record`) are the domain's; TERMINAL
reads the record and the state's clock and adds the draws by rule (the third occurrence, Lean `Chess.threefoldClaimable` / `Ideas.Game.chess_threefold_iff`; the
clock at its limit with no mate on the move, Lean `Chess.fiftyMoveClaimable` / `Ideas.Game.chess_fifty_iff`); CLOCK_EVENTS names what resets the clock (Lean
`Chess.applyMove_halfmove` / `Ideas.Game.chess_clock`). The play server builds the record from the game (`play_server.game_history`), the chase reads it beside the
line it is reading (agenda.py `ChaseCtx.history`, `_LineRecord`)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

import chess

from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import names as NM

WHITE, BLACK = chess.WHITE, chess.BLACK
MATE_VALUE = LS.goal_kinds().terminal.stake       # plan 75.2 (a): the stake of the goal that ends the game, read from its GOAL entry (lemmas/mate.json: ONE definition)
DRAWS_BY_RULE = ("repetition", "fifty-move")      # plan 75.4 (1): the terminals TERMINAL reads off the game record and the clock (each a draw; Lean `Ideas.Game.Record.drawByRule`)


class Record:
    """plan 75.4 (1) THE GAME RECORD (Lean `Ideas.Game.Record`): how often each position occurred BEFORE the current one, by the domain's identity
    (`Game.position_key`). `add` is the Lean `Record.push` (the count invariant `Record.Sound` is kept by every push, `Record.push_sound`, so along every step of
    the game, `Record.step_sound`). Empty: no game before the position (a puzzle). `key in record` -- the identity occurred before."""

    __slots__ = ("counts",)

    def __init__(self, counts=None):
        self.counts = Counter(counts or {})

    @classmethod
    def of_line(cls, game: "Game", states: Iterable) -> "Record":
        """The record of a line of states (oldest first) played past: each pushed once."""
        r = cls()
        for s in states:
            r.push(game, s)
        return r

    def count(self, key) -> int:
        """How often the identity `key` occurred before."""
        return self.counts.get(key, 0)

    def add(self, key, n: int = 1) -> "Record":
        """`key`'s position played past (Lean `Record.push`)."""
        self.counts[key] += n
        return self

    def push(self, game: "Game", s) -> "Record":
        """The state `s` played past: its identity counted once more."""
        return self.add(game.position_key(s))

    def copy(self) -> "Record":
        return Record(self.counts)

    def total(self) -> int:
        """The positions before (with their repeats)."""
        return sum(self.counts.values())

    def __contains__(self, key) -> bool:
        return self.counts.get(key, 0) > 0

    def __len__(self) -> int:                     # the distinct identities (an empty record is falsy)
        return len(self.counts)

    def __iter__(self):
        return iter(self.counts)

    def __repr__(self) -> str:
        return f"Record({dict(self.counts)!r})"


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
    def terminal(self, s, before: "Record | None" = None) -> str | None: ...      # None, or 'mate' (the mover is mated), 'stalemate', 'draw', or a draw by rule (DRAWS_BY_RULE) read off the record and the clock
    def position_key(self, s) -> str: ...                         # plan 75.4: the state's identity (what a repetition repeats; the clocks omitted)
    def clock(self, s) -> int: ...                                # plan 75.4: the state's clock (the moves since the last clock event)
    def key_clock(self, key: str) -> int: ...                     # plan 75.4: the clock a state's key carries (without building the state)
    def clock_events(self, s, m) -> frozenset: ...                # plan 75.4 (3): what of the move resets the clock (empty: the clock goes on)
    def moves_to(self, s, frm: int, to: int, legal: bool = True) -> list: ...      # plan 75.4 (5): the moves from one location to another (a promotion: one per unit it may place)
    def placing_most(self, moves: list, valuation): ...           # plan 75.4 (5): among them, the one placing the unit a valuation values most
    def push_to(self, s, frm: int, to: int, valuation, legal: bool = True): ...   # ... the two together (None: no such move)
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

    REPETITIONS = 3                                   # plan 75.4 (1): the occurrence that draws (the kernel's: Lean `Chess.threefoldClaimable`, 3 <= the identity's occurrences)
    CLOCK_LIMIT = 100                                 # ... and the clock's limit in half-moves (the kernel's: Lean `Chess.fiftyMoveClaimable`, 100 <= halfmove; python-chess `Board.is_fifty_moves`)

    def terminal(self, s: chess.Board, before: "Record | None" = None) -> str | None:
        """The rule's terminals in the kernel's order (Lean `Chess.gameStatus`: a mate on the move comes first): 'mate', 'stalemate', 'draw' (insufficient material),
        then the DRAWS BY RULE (plan 75.4 (1)): 'repetition' -- the state's identity occurred at least twice BEFORE it in the game record `before` (the third occurrence,
        Lean `Ideas.Game.Record.threefold` = `Chess.threefoldClaimable` of the history, `Ideas.Game.chess_threefold_iff`); 'fifty-move' -- the clock at its limit
        (python-chess `Board.is_fifty_moves`: a hundred half-moves and a legal move, Lean `Chess.fiftyMoveClaimable`, `Ideas.Game.chess_fifty_iff`). Without a record no
        repetition is read (the clock rides on the state). Both are the CLAIMABLE draws: either side may claim, so no claim to win survives one."""
        if s.is_checkmate():
            return "mate"
        if s.is_stalemate():
            return "stalemate"
        if s.is_insufficient_material():
            return "draw"
        if before is not None and self.repetition(s, before):
            return "repetition"
        if s.is_fifty_moves():
            return "fifty-move"
        return None

    def repetition(self, s: chess.Board, before: "Record") -> bool:
        """plan 75.4 (1): the third occurrence of `s` -- its identity occurred at least twice before it (Lean `Ideas.Game.Record.threefold`)."""
        return before.count(self.position_key(s)) + 1 >= self.REPETITIONS

    def position_key(self, s: chess.Board) -> str:
        """plan 75.4 (1) / (4): THE KERNEL'S IDENTITY of a position (Lean `Chess.repetitionKey`): the board, the side to move, the castling rights and the en-passant
        square ONLY when an en-passant capture is legal (`Chess.effectiveEnPassant`), the clocks omitted -- python-chess's EPD with its default `en_passant='legal'`
        (and its cleaned castling rights), never the raw FEN's fields."""
        return s.epd()

    def clock(self, s: chess.Board) -> int:
        """plan 75.4 (1): the halfmove clock (Lean `Position.halfmove`)."""
        return s.halfmove_clock

    def key_clock(self, key: str) -> int:
        """plan 75.4 (1): the clock a state's key carries without building the state -- the halfmove field of the kernel's notation (the FEN the kernel rendered; 0 when the
        key has none). Equal to `clock(from_key(key))`."""
        f = key.split()
        return int(f[4]) if len(f) > 4 else 0

    def clock_events(self, s: chess.Board, m: chess.Move) -> frozenset:
        """plan 75.4 (3): what of the move `m` resets the clock -- the fifty-move rule's events (Lean `Chess.applyMove_halfmove`: `moveIsPawn || moveIsCapture`,
        `Ideas.Game.clockResets`, `Ideas.Game.chess_clock`), by the names table's names: 'capture' (python-chess `Board.is_capture`, en passant included) and 'pawn' (the unit
        moved is a pawn, a promotion included). Non-empty exactly when python-chess's `Board.is_zeroing(m)`: the clock after the move is 0."""
        ev = []
        if s.is_capture(m):
            ev.append(NM.EVENT_CAPTURE)
        if NM.pawn(s.piece_at(m.from_square)):
            ev.append(NM.EVENT_PAWN)
        return frozenset(ev)

    def moves_to(self, s: chess.Board, frm: int, to: int, legal: bool = True) -> list:
        """plan 75.4 (5): the kernel's moves from `frm` to `to` -- python-chess's generator, LEGAL ones (or, with `legal` False, the pseudo-legal ones a static exchange reads):
        a pawn reaching its last rank gives one move per unit the promotion may place (queen, rook, bishop, knight; Lean `Chess.Move.promotion`); empty when none goes there."""
        gen = s.generate_legal_moves if legal else s.generate_pseudo_legal_moves
        return list(gen(chess.BB_SQUARES[frm], chess.BB_SQUARES[to]))

    def placing_most(self, moves: list, valuation) -> "chess.Move | None":
        """plan 75.4 (5): among moves between two locations, the one placing the unit `valuation` (the store's VALUATION entry: `material`) values most -- the promotion's
        choice by the valuation, not by a unit written here (a move that places nothing is the only one between its squares); None for no move."""
        return max(moves, key=lambda m: valuation.of(chess.piece_symbol(m.promotion)) if m.promotion else 0, default=None)

    def push_to(self, s: chess.Board, frm: int, to: int, valuation, legal: bool = True) -> "chess.Move | None":
        """plan 75.4 (5): THE PUSH (or the capture) from `frm` to `to` the words build: the kernel's moves there (`moves_to`), the unit placed chosen by `valuation`
        (`placing_most`); None when the kernel has no such move."""
        return self.placing_most(self.moves_to(s, frm, to, legal), valuation)

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

    def terminal(self, s: chess.Board, before: "Record | None" = None) -> str | None:
        for side in (chess.WHITE, chess.BLACK):
            k = s.king(side)
            if k is not None and k in self.CENTRE:
                return "loss" if side != s.turn else "win"       # their king stands there with us to move: we have lost; ours: we have won (the mover reads the terminal)
        return super().terminal(s, before)


CHESS = ChessGame()
CHESS960 = ChessGame(chess960=True)                              # plan 71.6: the same laws, another start (and castling by the kernel's 960 rule): one parameter, not a second instance
KING_OF_THE_HILL = KingOfTheHillGame()
assert isinstance(CHESS, Game) and isinstance(CHESS960, Game) and isinstance(KING_OF_THE_HILL, Game)
