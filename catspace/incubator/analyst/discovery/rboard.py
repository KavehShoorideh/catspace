"""plan 72.7 step 2a THE RUST BOARD (docs/RUST_BOUNDARY.md 3.3, the retire layer): `RBoard`, a thin Python face over the Rust position handle `analyst_core.Pos`,
exposing exactly the python-chess `Board` subset the chase's reply loop and the static leaf read on boards of their own (grep-verified 2026-10-24, see below):
`turn`, `legal_moves` (python-chess's ORDER: `Pos.legal_ucis`, RUST_BOUNDARY risk 4), `push` (a NEW board: the handle is immutable, so `copy` is free), `piece_at`,
`attackers`, `attacks` (as `chess.SquareSet` from the u64 masks), `is_check`, `gives_check`, `is_capture`, `san` (from Rust: the step-0 cross-check found no SAN
mismatch on 45,607 moves), `fen`, `king`, `ep_square`, `board_fen`, `is_checkmate`, `is_stalemate`; beyond the RUST_BOUNDARY subset, found by the grep of the sites
it replaces: `is_castling` (the leaf drops castling), `is_attacked_by` (the principal captures), `attacks_mask` (the refuter fix), `with_turn` (the pass:
graph.with_turn on the handle). `chess.Move` stays the move currency; ucis cross the border.

It is NOT a python-chess board: a Python word that takes a board (regress, race, binder, the agenda's own words) still gets a python-chess `Board`, built where it
was built before. `RBoard` is for the reads the loop and the leaf make themselves.

THE BRIDGE (RUST_BOUNDARY risk 9; graph.BRIDGE_CHECK, the one debug flag, CATSPACE_BRIDGE_CHECK=1): every RBoard then carries a python-chess SHADOW of itself
(the parent's shadow pushed, or the handle's FEN parsed) and every read is asserted equal to the shadow's -- the legal moves in order, the SAN, the attackers,
the checks -- so a guard run with the flag on compares the Rust board with python-chess on every position the loop and the leaf read.
"""
from __future__ import annotations

import chess

import analyst_core as _A
from catspace.incubator.analyst.discovery import graph as G

_PIECES = {(pt, c): chess.Piece(pt, c) for pt in chess.PIECE_TYPES for c in chess.COLORS}       # python-chess Piece values, one per (type, colour)
BRIDGE_READS = [0]                                                                              # under BRIDGE_CHECK: how many RBoard reads were compared with the shadow


class _Moves:
    """`board.legal_moves` of an RBoard: python-chess's order, as `chess.Move` values (iterable, `in`, `count()`, truth, `len`)."""
    __slots__ = ("_rb", "_list", "_set")

    def __init__(self, rb: "RBoard"):
        self._rb = rb
        self._list = None
        self._set = None

    def _moves(self) -> list:
        if self._list is None:
            us = self._rb.pos.legal_ucis()
            self._list = [chess.Move.from_uci(u) for u in us]
            self._set = frozenset(us)
            if G.BRIDGE_CHECK:
                _check(self._rb, [m.uci() for m in self._rb._shadow.legal_moves] == us, "legal moves")
        return self._list

    def __iter__(self):
        return iter(self._moves())

    def __len__(self) -> int:
        return len(self._moves())

    def count(self) -> int:
        return len(self._moves())

    def __bool__(self) -> bool:
        return self._rb.pos.legal_count() > 0

    def __contains__(self, move: chess.Move) -> bool:
        """python-chess's `board.is_legal(move)` (standard chess): the move's uci is a legal one, or it is the king's step from e1 / e8 onto the corner h / a, which
        python-chess reads as the castling to g / c (`Board._from_chess960`) and accepts when that castling is legal."""
        self._moves()
        if not move:
            return False
        u = move.uci()
        out = u in self._set
        if not out and move.promotion is None and move.drop is None:
            alias = {(chess.E1, chess.H1): "e1g1", (chess.E1, chess.A1): "e1c1", (chess.E8, chess.H8): "e8g8", (chess.E8, chess.A8): "e8c8"}.get((move.from_square, move.to_square))
            if alias is not None:
                k = self._rb.pos.piece_at(move.from_square)
                out = k is not None and k[0] == chess.KING and alias in self._set
        if G.BRIDGE_CHECK:
            _check(self._rb, (move in self._rb._shadow.legal_moves) == out, f"legal {u}")
        return out


def _check(rb: "RBoard", ok: bool, what: str) -> None:
    BRIDGE_READS[0] += 1
    assert ok, f"the RBoard bridge split on {what}: {rb.pos.fen()}"


class RBoard:
    """A board read from its Rust handle `pos` (see the module docstring). Immutable: `push` returns a new RBoard, `copy` returns itself."""
    __slots__ = ("pos", "_moves", "_shadow")

    def __init__(self, pos: "_A.Pos", shadow: "chess.Board | None" = None):
        self.pos = pos
        self._moves = None
        self._shadow = None
        if G.BRIDGE_CHECK:
            self._shadow = shadow if shadow is not None else chess.Board(pos.fen())
            _check(self, self._shadow.fen() == pos.fen(), "fen")

    @classmethod
    def of(cls, board: chess.Board) -> "RBoard":
        """The RBoard of a python-chess board (its handle from its parts, graph.handle; the board itself is the shadow under the bridge)."""
        return cls(G.handle(board), board.copy(stack=False) if G.BRIDGE_CHECK else None)

    @classmethod
    def on(cls, board: chess.Board, pos: "_A.Pos") -> "RBoard":
        """The RBoard of a NODE (graph.Node): its handle `pos`, its python-chess `board` the shadow under the bridge (the node's two must be one position)."""
        if G.BRIDGE_CHECK:
            G._bridge(board, pos)
            return cls(pos, board.copy(stack=False))
        return cls(pos)

    # ------------------------------------------------------------------------------------------------ the position
    @property
    def turn(self) -> bool:
        return self.pos.turn_white()

    @property
    def legal_moves(self) -> _Moves:
        if self._moves is None:
            self._moves = _Moves(self)
        return self._moves

    @property
    def ep_square(self) -> "int | None":
        return self.pos.ep_square()

    def fen(self) -> str:
        return self.pos.fen()

    def board_fen(self) -> str:
        return self.pos.board_fen()

    def key(self) -> int:
        return self.pos.key()

    def copy(self, stack: bool = False) -> "RBoard":
        return self

    def push(self, move: chess.Move) -> "RBoard":
        """The board after the LEGAL move (ValueError on an illegal one: the handle never plays an unchecked move, risk 7). A new RBoard; this one is unchanged."""
        u = move.uci() if isinstance(move, chess.Move) else move
        shadow = None
        if G.BRIDGE_CHECK:
            shadow = self._shadow.copy(stack=False)
            shadow.push(chess.Move.from_uci(u))
        return RBoard(self.pos.push(u), shadow)

    def pushed(self, move: chess.Move, pos: "_A.Pos") -> "RBoard":
        """The board after the legal `move` whose handle the caller already holds (`pos`: the route table's); under the bridge its shadow is this board's pushed."""
        shadow = None
        if G.BRIDGE_CHECK:
            shadow = self._shadow.copy(stack=False)
            shadow.push(move)
        return RBoard(pos, shadow)

    def with_turn(self, colour: bool) -> "RBoard | None":
        """graph.with_turn on the handle (that side to move, en passant cleared, castling and clocks kept); None where shakmaty does not hold the board so (the
        side not to move would stand in check)."""
        p = self.pos.with_turn(colour)
        return None if p is None else RBoard(p, G.with_turn(self._shadow, colour) if G.BRIDGE_CHECK else None)

    # ------------------------------------------------------------------------------------------------ the squares
    def piece_at(self, sq: int) -> "chess.Piece | None":
        p = self.pos.piece_at(sq)
        out = None if p is None else _PIECES[p]
        if G.BRIDGE_CHECK:
            _check(self, self._shadow.piece_at(sq) == out, f"piece_at {sq}")
        return out

    def king(self, colour: bool) -> "int | None":
        out = self.pos.king(colour)
        if G.BRIDGE_CHECK:
            _check(self, self._shadow.king(colour) == out, "king")
        return out

    def attackers_mask(self, colour: bool, sq: int) -> int:
        out = self.pos.attackers(sq, colour)
        if G.BRIDGE_CHECK:
            _check(self, self._shadow.attackers_mask(colour, sq) == out, f"attackers {sq}")
        return out

    def attackers(self, colour: bool, sq: int) -> chess.SquareSet:
        return chess.SquareSet(self.attackers_mask(colour, sq))

    def is_attacked_by(self, colour: bool, sq: int) -> bool:
        return bool(self.attackers_mask(colour, sq))

    def attacks_mask(self, sq: int) -> int:
        out = self.pos.attacks(sq)
        if G.BRIDGE_CHECK:
            _check(self, self._shadow.attacks_mask(sq) == out, f"attacks {sq}")
        return out

    def attacks(self, sq: int) -> chess.SquareSet:
        return chess.SquareSet(self.attacks_mask(sq))

    # ------------------------------------------------------------------------------------------------ the moves (legal moves only)
    def san(self, move: chess.Move) -> str:
        out = self.pos.san(move.uci())
        if G.BRIDGE_CHECK:
            _check(self, self._shadow.san(move) == out, f"san {move.uci()}")
        return out

    def is_capture(self, move: chess.Move) -> bool:
        out = self.pos.is_capture(move.uci())
        if G.BRIDGE_CHECK:
            _check(self, self._shadow.is_capture(move) == out, f"is_capture {move.uci()}")
        return out

    def gives_check(self, move: chess.Move) -> bool:
        out = self.pos.gives_check(move.uci())
        if G.BRIDGE_CHECK:
            _check(self, self._shadow.gives_check(move) == out, f"gives_check {move.uci()}")
        return out

    def is_castling(self, move: chess.Move) -> bool:
        """python-chess's `is_castling` (a king move of more than one file, or onto its own rook), read off the handle's pieces."""
        k = self.pos.piece_at(move.from_square)
        out = False
        if k is not None and k[0] == chess.KING:
            to = self.pos.piece_at(move.to_square)
            out = abs(chess.square_file(move.from_square) - chess.square_file(move.to_square)) > 1 or (to is not None and to[0] == chess.ROOK and to[1] == self.turn)
        if G.BRIDGE_CHECK:
            _check(self, self._shadow.is_castling(move) == out, f"is_castling {move.uci()}")
        return out

    def is_check(self) -> bool:
        out = self.pos.is_check()
        if G.BRIDGE_CHECK:
            _check(self, self._shadow.is_check() == out, "is_check")
        return out

    def is_checkmate(self) -> bool:
        out = self.pos.is_checkmate()
        if G.BRIDGE_CHECK:
            _check(self, self._shadow.is_checkmate() == out, "is_checkmate")
        return out

    def is_stalemate(self) -> bool:
        out = self.pos.is_stalemate()
        if G.BRIDGE_CHECK:
            _check(self, self._shadow.is_stalemate() == out, "is_stalemate")
        return out

    def __repr__(self) -> str:
        return f"RBoard({self.pos.fen()!r})"
