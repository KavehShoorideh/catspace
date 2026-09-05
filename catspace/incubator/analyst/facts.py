"""L2 -- STATE FACTS. Global, square-free facts of a position, derived generically from L1,
plus the CONVENTIONAL EXCHANGE SCALE (pawn 1 ... queen 9) that L3 uses to compare pieces.
The scale is a fact about the piece set, not a valuation: whether losing material is bad is
L5's business (it is good in antichess)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import chess

from . import rules as R

def value(board: chess.Board, sq: int, scale: "R.Scale | None" = None) -> float:
    """Exchange value of the piece on sq under the board's variant scale (or an explicit one)."""
    p = board.piece_at(sq)
    if p is None:
        return 0.0
    sc = scale or R.scale_of(board)
    return sc.of_type(p.piece_type, royal=R.is_royal(board, sq))


def material(board: chess.Board, color: chess.Color, scale: "R.Scale | None" = None) -> float:
    sc = scale or R.scale_of(board)
    return sum(sc.of_type(p.piece_type) for p in board.piece_map().values() if p.color == color)


@dataclass(frozen=True)
class StateFacts:
    variant: str
    turn: str
    in_check: bool
    castling: dict
    ep_square: str | None
    material_white: int
    material_black: int
    material_diff: int          # white minus black, exchange scale
    phase_nonpawn: int          # non-pawn material of both sides (0 = pawn ending)
    halfmove_clock: int
    fullmove: int
    repetition: bool

    def as_dict(self) -> dict:
        return asdict(self)


def state_facts(board: chess.Board) -> StateFacts:
    mw, mb = material(board, chess.WHITE), material(board, chess.BLACK)
    sc = R.scale_of(board)
    nonpawn = sum(sc.of_type(p.piece_type) for p in board.piece_map().values() if p.piece_type not in (chess.PAWN, chess.KING))
    return StateFacts(
        variant=R.rules_of(board).name,
        turn="white" if board.turn else "black",
        in_check=board.is_check(),
        castling={"K": board.has_kingside_castling_rights(chess.WHITE), "Q": board.has_queenside_castling_rights(chess.WHITE),
                  "k": board.has_kingside_castling_rights(chess.BLACK), "q": board.has_queenside_castling_rights(chess.BLACK)},
        ep_square=chess.square_name(board.ep_square) if board.ep_square is not None else None,
        material_white=mw, material_black=mb, material_diff=mw - mb, phase_nonpawn=nonpawn,
        halfmove_clock=board.halfmove_clock, fullmove=board.fullmove_number,
        repetition=board.is_repetition(2),
    )
