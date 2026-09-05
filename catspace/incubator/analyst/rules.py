"""L1 -- RULES. Piece set, movement, special moves, legality, terminal conditions, and the
OBJECTIVE, implemented by python-chess Board classes (standard + variants).

Contract for every layer above: use ONLY this module and the Board probe API (legal_moves,
attacks, attackers, is_capture, gives_check, is_game_over, king, ...). Never a chess constant
("bishops move diagonally"). That discipline is what makes a variant a one-layer change.
The objective is QUARANTINED: L2-L4 must not read Rules.objective; only L5 may."""
from __future__ import annotations

from dataclasses import dataclass, field

import chess
import chess.variant


@dataclass(frozen=True)
class Scale:
    """The EXCHANGE SCALE: what a piece is worth in a trade. Not a rule (rules say how pieces move) and
    not objective-free (in antichess material is a liability; a knight in hand in crazyhouse is worth
    more) -- it is a DERIVATIVE OF THE OBJECTIVE, declared with it here, and consumed explicitly by the
    relations that classify by value (pin vs skewer vs x-ray, a fork's teeth, attacked by a lesser
    piece) and by every layer that prices a capture. One object, one place to change per variant."""
    values: dict = field(default_factory=lambda: {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0, chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0})
    royal_value: float = 100.0   # a royal piece outranks everything in exchange comparisons
    material_sign: int = 1       # +1: material is an asset; -1: a liability (antichess)
    minor_value: float = 3.0     # the "real piece" threshold used by significance filters

    def of_type(self, piece_type: int, royal: bool = False) -> float:
        return self.royal_value if royal and piece_type == chess.KING else self.values.get(piece_type, 0.0)

    def of_symbol(self, sym: str) -> float:
        """'n' / 'N' -> value; kings count as royal (the symbol carries no royalty flag)."""
        if not sym or sym == "-":
            return 0.0
        pt = chess.Piece.from_symbol(sym).piece_type
        return self.royal_value if pt == chess.KING else self.values.get(pt, 0.0)


STANDARD = Scale()
ANTICHESS = Scale(material_sign=-1, royal_value=0.0)          # no royal piece; material is a liability
CRAZYHOUSE = Scale(values={chess.PAWN: 1.0, chess.KNIGHT: 3.5, chess.BISHOP: 3.5, chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0})


@dataclass(frozen=True)
class Rules:
    name: str
    board_cls: type
    royal: bool                  # a royal king exists: check / checkmate are meaningful
    objective: str               # L5-ONLY
    captures_forced: bool = False
    scale: Scale = STANDARD      # the objective's exchange scale (see Scale)

    def board(self, fen: str | None = None) -> chess.Board:
        return self.board_cls(fen) if fen else self.board_cls()


VARIANTS: dict[str, Rules] = {
    "chess": Rules("chess", chess.Board, True, "checkmate"),
    "antichess": Rules("antichess", chess.variant.AntichessBoard, False, "lose_all_pieces", True, ANTICHESS),
    "atomic": Rules("atomic", chess.variant.AtomicBoard, True, "checkmate_or_explode_king"),
    "kingofthehill": Rules("kingofthehill", chess.variant.KingOfTheHillBoard, True, "checkmate_or_king_to_center"),
    "racingkings": Rules("racingkings", chess.variant.RacingKingsBoard, True, "king_to_rank8"),
    "threecheck": Rules("threecheck", chess.variant.ThreeCheckBoard, True, "checkmate_or_three_checks"),
    "crazyhouse": Rules("crazyhouse", chess.variant.CrazyhouseBoard, True, "checkmate", False, CRAZYHOUSE),
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


def scale_of(board: chess.Board) -> Scale:
    return rules_of(board).scale


def set_scale(variant: str, scale: Scale) -> None:
    """Swap a variant's exchange scale (a user's own values, a phase-dependent scale, a fairy set).
    Every valued relation, capture price and significance threshold follows from the next call on."""
    from dataclasses import replace
    VARIANTS[variant] = replace(VARIANTS[variant], scale=scale)


def is_royal(board: chess.Board, sq: int) -> bool:
    p = board.piece_at(sq)
    return p is not None and p.piece_type == chess.KING and rules_of(board).royal
