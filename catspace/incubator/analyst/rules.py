"""L1 -- RULES. Piece set, movement, special moves, legality, terminal conditions, and the
OBJECTIVE, implemented by python-chess Board classes (standard + variants).

Contract for every layer above: use ONLY this module and the Board probe API (legal_moves,
attacks, attackers, is_capture, gives_check, is_game_over, king, ...). Never a chess constant
("bishops move diagonally"). That discipline is what makes a variant a one-layer change.
The objective is QUARANTINED: L2-L4 must not read Rules.objective; only L5 may."""
from __future__ import annotations

from dataclasses import dataclass

import chess
import chess.variant


@dataclass(frozen=True)
class Rules:
    name: str
    board_cls: type
    royal: bool                  # a royal king exists: check / checkmate are meaningful
    objective: str               # L5-ONLY
    captures_forced: bool = False

    def board(self, fen: str | None = None) -> chess.Board:
        return self.board_cls(fen) if fen else self.board_cls()


VARIANTS: dict[str, Rules] = {
    "chess": Rules("chess", chess.Board, True, "checkmate"),
    "antichess": Rules("antichess", chess.variant.AntichessBoard, False, "lose_all_pieces", True),
    "atomic": Rules("atomic", chess.variant.AtomicBoard, True, "checkmate_or_explode_king"),
    "kingofthehill": Rules("kingofthehill", chess.variant.KingOfTheHillBoard, True, "checkmate_or_king_to_center"),
    "racingkings": Rules("racingkings", chess.variant.RacingKingsBoard, True, "king_to_rank8"),
    "threecheck": Rules("threecheck", chess.variant.ThreeCheckBoard, True, "checkmate_or_three_checks"),
    "crazyhouse": Rules("crazyhouse", chess.variant.CrazyhouseBoard, True, "checkmate"),
    "horde": Rules("horde", chess.variant.HordeBoard, True, "checkmate_or_capture_horde"),
}


def rules_for(name: str) -> Rules:
    return VARIANTS[name]


def rules_of(board: chess.Board) -> Rules:
    for r in VARIANTS.values():
        if type(board) is r.board_cls:
            return r
    return VARIANTS["chess"]


def royal_square(board: chess.Board, color: chess.Color) -> int | None:
    """The square of color's royal king, or None when the rules have no royal piece."""
    return board.king(color) if rules_of(board).royal else None


def is_royal(board: chess.Board, sq: int) -> bool:
    p = board.piece_at(sq)
    return p is not None and p.piece_type == chess.KING and rules_of(board).royal
