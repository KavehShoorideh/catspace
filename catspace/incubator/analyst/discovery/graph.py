"""LAYER 1 OF THE CORE (ANALYST_PLAN 52.1): the board as a graph that answers attack / reach / COUNTERFACTUAL questions. Pure functions over python-chess
boards (the rules) and, where the vocabulary has the word, the Rust `analyst_core` reading (`see`). A counterfactual question is a board with a piece placed
or removed, re-read; it is never a claim: the guarantee of anything built on it comes from a proved schema's binding (binder.py).

Colours are python-chess colours (True = white). Squares are python-chess ints.
KNOWN TRAP: the Rust word `see` reads 0 for the side NOT to move (see reasoned.py's docstring): `see` here always reads with the mover = the side to move
at the board, so for the exchange WE would win on their piece, put us on move first (`with_turn`).
"""
from __future__ import annotations

import chess

from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import selfmine as SM

WORTH = SM.PIECE_WORTH                       # p 1, n 3, b 3, r 5, q 9, k 0
SLIDING = {chess.BISHOP: (-9, -7, 7, 9), chess.ROOK: (-8, -1, 1, 8), chess.QUEEN: (-9, -8, -7, -1, 1, 7, 8, 9)}


def worth(piece: chess.Piece | None) -> int:
    return 0 if piece is None else WORTH[piece.symbol().lower()]


def name(sq: int) -> str:
    return chess.square_name(sq)


def colour_of(us: str | bool) -> bool:
    return us if isinstance(us, bool) else us == "w"


def with_turn(board: chess.Board, colour: bool) -> chess.Board:
    """A copy with `colour` to move (en passant square and the move stack dropped: a counterfactual board has no history)."""
    c = board.copy(stack=False)
    c.turn = colour
    c.ep_square = None
    return c


# ---------------------------------------------------------------------------------------------------------------- attack questions

def attackers_of(board: chess.Board, sq: int, colour: bool) -> list:
    """Squares of `colour`'s pieces that attack `sq` now (x-rays are `xrays_to`)."""
    return sorted(board.attackers(colour, sq))


def defenders_of(board: chess.Board, sq: int) -> list:
    """Squares of the pieces that guard the piece on `sq` (its own colour's attackers of the square)."""
    pc = board.piece_at(sq)
    return [] if pc is None else attackers_of(board, sq, pc.color)


def flights(board: chess.Board, king_sq: int) -> list:
    """Squares the king on `king_sq` can step to that are not attacked by the other side (the king itself lifted off the board, so a slider's shadow behind
    it still attacks), not occupied by its own side, and not the square of a piece giving check (taking the checker is the defect `checker takeable`, not a
    flight)."""
    king = board.piece_at(king_sq)
    colour = king.color
    lifted = board.copy(stack=False)
    lifted.remove_piece_at(king_sq)
    checkers = set(board.attackers(not colour, king_sq))
    out = []
    for to in board.attacks(king_sq):
        pc = board.piece_at(to)
        if (pc is not None and pc.color == colour) or to in checkers:
            continue
        if lifted.is_attacked_by(not colour, to):
            continue
        out.append(to)
    return sorted(out)


def xrays_to(board: chess.Board, sq: int, colour: bool, max_blockers: int = 2) -> list:
    """The sliders of `colour` that would attack `sq` if the pieces between them and it were gone: [(piece_sq, [blocker squares])] with 1..max_blockers
    blockers on the line (any colour), the nearest slider per ray only counts (a rook behind a rook is a battery, found through its blocker)."""
    out = []
    for ps in chess.SQUARES:
        pc = board.piece_at(ps)
        if pc is None or pc.color != colour or pc.piece_type not in SLIDING:
            continue
        between = chess.SquareSet(chess.between(ps, sq))
        # the piece must move along that line: rook-like on a file / rank, bishop-like on a diagonal
        same_file_rank = chess.square_file(ps) == chess.square_file(sq) or chess.square_rank(ps) == chess.square_rank(sq)
        diag = abs(chess.square_file(ps) - chess.square_file(sq)) == abs(chess.square_rank(ps) - chess.square_rank(sq))
        if ps == sq or not (same_file_rank or diag):
            continue
        if pc.piece_type == chess.ROOK and not same_file_rank:
            continue
        if pc.piece_type == chess.BISHOP and not diag:
            continue
        blockers = [b for b in between if board.piece_at(b) is not None]
        if 1 <= len(blockers) <= max_blockers:
            blockers.sort(key=lambda b: chess.square_distance(b, ps))
            out.append((ps, blockers))
    return sorted(out)


# ---------------------------------------------------------------------------------------------------------------- counterfactuals

def without(board: chess.Board, sq: int) -> chess.Board:
    """The board with the piece on `sq` removed."""
    c = board.copy(stack=False)
    c.remove_piece_at(sq)
    return c


def with_piece(board: chess.Board, piece: chess.Piece, at: int, removed_from: int | None = None) -> chess.Board:
    """The counterfactual placement: `piece` on `at` (whatever stood there is gone), lifted from `removed_from` when given."""
    c = board.copy(stack=False)
    if removed_from is not None:
        c.remove_piece_at(removed_from)
    c.set_piece_at(at, piece)
    return c


def mobility(board: chess.Board, colour: bool) -> int:
    """Legal moves `colour` would have on move."""
    return with_turn(board, colour).legal_moves.count()


def see(board: chess.Board, sq: int) -> int:
    """The Rust word `see` on `sq`, read with the mover = the side to move at `board`: what the SIDE TO MOVE wins by the exchange started by a capture on `sq`
    (it reads 0 for the side not to move, and nothing for a piece of the mover's own side). `see_for` puts the capturer on move first."""
    BV.install_terms()
    return SM.val_at(board.fen(), ("see", name(sq)), "w" if board.turn else "b") or 0


def see_for(board: chess.Board, sq: int, capturer: bool) -> int:
    """What `capturer` wins by the exchange on the piece on `sq` (a piece of the other side): `see` read with `capturer` on move."""
    return see(with_turn(board, capturer), sq)
