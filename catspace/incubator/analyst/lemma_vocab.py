"""THE VOCABULARY the lemma miner reasons in (Kaveh 2026-10-10: "the LLM is going to have to be you and me reviewing
something together and agreeing ... not the lemma, but the next piece of VOCABULARY, whatever the highest level of
abstraction is").

A CONCEPT is a named predicate over a CONTEXT -- the board after a move, the defender (the side to move there), the
move that was just played and the board before it. Each concept carries its plain-language definition, its level and
its SOURCE: built in, or agreed in a review (who, when, from which open cases). The miner (lemma_miner.py) turns concepts
into lemmas soundly; inventing a concept is the step a review (or, later, a model behind an API) takes. To add one: write
it below with @concept(...), source = "review YYYY-MM-DD: <who> -- <the open cases it explains>".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import chess


@dataclass
class Context:
    before: chess.Board            # the board before the move (the mover to move)
    move: chess.Move | None        # the move just played (None: a position on its own)
    board: chess.Board             # the board after it -- the defender to move

    @property
    def defender(self) -> bool:
        return self.board.turn

    @property
    def mover(self) -> bool:
        return not self.board.turn


@dataclass
class Concept:
    name: str
    text: str
    level: str                     # "position" (the defender's position), "move" (a class of the move just played), "material"
    source: str
    fn: Callable[[Context], bool] = field(repr=False)


VOCAB: dict[str, Concept] = {}


def concept(name: str, text: str, level: str, source: str):
    def wrap(fn):
        VOCAB[name] = Concept(name, text, level, source, fn)
        return fn
    return wrap


BUILTIN = "built in 2026-10-10 (Claude, from Kaveh's stalemate lemmas)"


def _moves_of(c: Context, kinds) -> bool:
    b = c.board
    return any(b.piece_type_at(m.from_square) in kinds for m in b.legal_moves)


# ---- the defender's position
@concept("defender in check", "the side to move is in check", "position", BUILTIN)
def _in_check(c): return c.board.is_check()

@concept("defender has a free pawn", "a pawn of the side to move has a legal move", "position", BUILTIN)
def _free_pawn(c): return _moves_of(c, {chess.PAWN})

@concept("defender has a free piece", "a knight, bishop, rook or queen of the side to move has a legal move", "position", BUILTIN)
def _free_piece(c): return _moves_of(c, {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN})

@concept("defender's king has a free square", "the king of the side to move has a legal move", "position", BUILTIN)
def _king_free(c): return _moves_of(c, {chess.KING})

@concept("defender has only the king", "the side to move has no piece or pawn besides its king", "material", BUILTIN)
def _bare(c): return len(c.board.piece_map()) - len([1 for p in c.board.piece_map().values() if p.color != c.defender]) == 1

@concept("defender has pawns", "the side to move has at least one pawn", "material", BUILTIN)
def _d_pawns(c): return bool(c.board.pieces(chess.PAWN, c.defender))

@concept("defender has a blocked pawn", "a pawn of the side to move has no legal move", "position", BUILTIN)
def _blocked(c):
    sqs = c.board.pieces(chess.PAWN, c.defender)
    movers = {m.from_square for m in c.board.legal_moves}
    return any(s not in movers for s in sqs)

@concept("defender has a pinned piece", "a piece of the side to move is pinned to its king", "position", BUILTIN)
def _pinned(c): return any(c.board.is_pinned(c.defender, s) for s in c.board.piece_map() if c.board.color_at(s) == c.defender and c.board.piece_type_at(s) != chess.KING)

@concept("mover has a queen", "the side that just moved has a queen", "material", BUILTIN)
def _m_queen(c): return bool(c.board.pieces(chess.QUEEN, c.mover))

@concept("mover has a rook", "the side that just moved has a rook", "material", BUILTIN)
def _m_rook(c): return bool(c.board.pieces(chess.ROOK, c.mover))

# ---- the move just played
@concept("the move gave check", "the move just played gives check", "move", BUILTIN)
def _mv_check(c): return c.move is not None and c.board.is_check()

@concept("the move was a capture", "the move just played captured something", "move", BUILTIN)
def _mv_cap(c): return c.move is not None and c.before.is_capture(c.move)

@concept("the move was a pawn move", "the move just played moved a pawn", "move", BUILTIN)
def _mv_pawn(c): return c.move is not None and c.before.piece_type_at(c.move.from_square) == chess.PAWN

@concept("the move was a king move", "the move just played moved the king", "move", BUILTIN)
def _mv_king(c): return c.move is not None and c.before.piece_type_at(c.move.from_square) == chess.KING

@concept("the move was a promotion", "the move just played promoted a pawn", "move", BUILTIN)
def _mv_promo(c): return c.move is not None and c.move.promotion is not None

@concept("the move was quiet", "the move just played neither captured nor gave check", "move", BUILTIN)
def _mv_quiet(c): return c.move is not None and not c.before.is_capture(c.move) and not c.board.is_check()


# ---- the questions (targets): what the miner looks for sufficient conditions of
TARGETS: dict[str, Callable[[Context], bool]] = {
    "stalemate": lambda c: c.board.is_stalemate(),
    "checkmate": lambda c: c.board.is_checkmate(),
}
