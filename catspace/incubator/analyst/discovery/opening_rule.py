"""(Restored from commit 1e2bc782, deleted later: the `opening` tie-break rule, used by the event loop's opening default.) OPENING SUBGOALS AS TIE-BREAKS (Kaveh 2026-10-06: "opening subgoals as tie-breaks", after the dig of the games vs
Stockfish-2000: the zero-search player loses them in the opening -- after our 5th move the median evaluation is -163 cp,
in 36 of 37 lost games it first falls below -150 by about ply 8 in a small step, and in 32 of those a move tied with the
played one stayed above -150. Over our first five moves the one-ply ledger ties a median of 19 moves: nothing is at
stake, so a random pick plays the opening).

Five classical opening principles, each a function of (board, move) -> bool, combined LEXICOGRAPHICALLY in this
priority order (a more important principle always outranks any number of less important ones):
    castle            the move castles (king safety first)
    develop           a knight or bishop that has not moved leaves its own starting square
    develop_central   ... onto files c-f, ranks 3-6
    centre_pawn       our pawns occupy or attack more of d4, e4, d5, e5 after the move than before
    steady            the move does not loosen the position: no king walk, no early queen (two minor pieces still
                      undeveloped), no rook move with the king still on its home square, no flank pawn (a, b, f, g, h
                      files, unless it captures), no second move of a piece that already moved this game
THE OPENING: fullmove <= OPENING_MOVES and our development is unfinished -- a minor piece still undeveloped, or the king
on its home square with a castling right (review 2026-10-06: EPD positions carry no move counters and load as move 1;
the move number alone opened the gate on every WAC position, endings included).
WHAT HAS MOVED: the game is replayed from its root, each piece followed by identity (a castling rook, en passant,
promotions included; review 2026-10-06: a scan of the move list missed the rook that moved by castling). A board built
from a FEN has no history: a minor piece then counts as undeveloped on its own starting square, and "a second move of a
piece" is not counted.
A conjecture-tier rule in frame_player: its claims only order the moves the ledger already ties, never a value (AGENTS
rule 7: subgoals enter through the prior)."""
from __future__ import annotations

from functools import lru_cache

import chess

OPENING_MOVES = 15

CENTRE = (chess.D4, chess.E4, chess.D5, chess.E5)
MINOR_HOME = {chess.WHITE: {chess.B1: chess.KNIGHT, chess.G1: chess.KNIGHT, chess.C1: chess.BISHOP, chess.F1: chess.BISHOP},
              chess.BLACK: {chess.B8: chess.KNIGHT, chess.G8: chess.KNIGHT, chess.C8: chess.BISHOP, chess.F8: chess.BISHOP}}
KING_HOME = {chess.WHITE: chess.E1, chess.BLACK: chess.E8}
FLANK_FILES = (0, 1, 5, 6, 7)                                   # a, b, f, g, h

# (name, weight): distinct powers of two, so the sum orders lexicographically by this priority
PRINCIPLES = (("castle", 16), ("develop", 8), ("develop_central", 4), ("centre_pawn", 2), ("steady", 1))


@lru_cache(maxsize=256)
def _moved_from_history(root_fen: str, moves: tuple) -> frozenset:
    """The squares whose current piece has moved at least once since the root position, by replaying the game."""
    b = chess.Board(root_fen)
    moved: set = set()
    for m in moves:
        if b.is_castling(m):                                        # the king and the rook both move
            side = b.turn
            kingside = b.is_kingside_castling(m)
            rank = 0 if side == chess.WHITE else 7
            p = b.piece_at(m.to_square)                             # king-takes-rook encoding (Chess960): the rook's square
            rook_from = m.to_square if p is not None and p.color == side and p.piece_type == chess.ROOK \
                else chess.square(7 if kingside else 0, rank)
            moved.discard(m.from_square); moved.discard(rook_from)
            b.push(m)
            moved.add(chess.square(6 if kingside else 2, rank)); moved.add(chess.square(5 if kingside else 3, rank))
            continue
        if b.is_en_passant(m):
            moved.discard(m.to_square + (-8 if b.turn == chess.WHITE else 8))  # the captured pawn
        moved.discard(m.from_square)
        b.push(m)
        moved.add(m.to_square)                                      # the arriving piece (a capture replaces the victim)
    return frozenset(moved)


def moved_squares(b: chess.Board) -> frozenset | None:
    """The squares whose current piece has moved this game; None for a board without history (built from a FEN)."""
    if not b.move_stack:
        return None
    return _moved_from_history(b.root().fen(), tuple(b.move_stack))


def _undeveloped(b: chess.Board, sq: int, side: chess.Color) -> bool:
    """A knight or bishop of `side` on its OWN starting square (by type) that has not moved (when the history is known)."""
    kind = MINOR_HOME[side].get(sq)
    p = b.piece_at(sq)
    if kind is None or p is None or p.color != side or p.piece_type != kind:
        return False
    moved = moved_squares(b)
    return moved is None or sq not in moved


def undeveloped_count(b: chess.Board, side: chess.Color) -> int:
    return sum(_undeveloped(b, sq, side) for sq in MINOR_HOME[side])


def in_opening(b: chess.Board) -> bool:
    """fullmove <= OPENING_MOVES, and our development is unfinished (an undeveloped minor piece, or the king at home with
    a castling right)."""
    side = b.turn
    if b.fullmove_number > OPENING_MOVES:
        return False
    if undeveloped_count(b, side) > 0:
        return True
    return b.king(side) == KING_HOME[side] and b.has_castling_rights(side)


def castle(b: chess.Board, m: chess.Move) -> bool:
    return b.is_castling(m)


def develop(b: chess.Board, m: chess.Move) -> bool:
    return _undeveloped(b, m.from_square, b.turn)


def develop_central(b: chess.Board, m: chess.Move) -> bool:
    return develop(b, m) and 2 <= chess.square_file(m.to_square) <= 5 and 2 <= chess.square_rank(m.to_square) <= 5


def _centre_count(b: chess.Board, side: chess.Color) -> int:
    """Centre squares our pawns stand on or attack."""
    pawns = b.pieces(chess.PAWN, side)
    n = 0
    for sq in CENTRE:
        if sq in pawns or any(sq in b.attacks(p) for p in pawns):
            n += 1
    return n


def centre_pawn(b: chess.Board, m: chess.Move) -> bool:
    p = b.piece_at(m.from_square)
    if p is None or p.piece_type != chess.PAWN:
        return False
    nb = b.copy(stack=False)
    nb.push(m)
    return _centre_count(nb, b.turn) > _centre_count(b, b.turn)


def steady(b: chess.Board, m: chess.Move) -> bool:
    p = b.piece_at(m.from_square)
    if p is None:
        return False
    side = b.turn
    if p.piece_type == chess.KING and not b.is_castling(m):
        return False
    if p.piece_type == chess.QUEEN and undeveloped_count(b, side) >= 2:
        return False
    if p.piece_type == chess.ROOK and b.king(side) == KING_HOME[side]:
        return False
    if p.piece_type == chess.PAWN and chess.square_file(m.from_square) in FLANK_FILES and not b.is_capture(m):
        return False
    moved = moved_squares(b)
    if moved is not None and not b.is_capture(m) and not b.is_castling(m) and m.from_square in moved:
        return False
    return True


FUNCS = {"castle": castle, "develop": develop, "develop_central": develop_central, "centre_pawn": centre_pawn, "steady": steady}


def opening_claims(b: chess.Board, moves=None) -> dict:
    """-> {uci: (score, why)} for the moves (default: every legal move) that serve at least one principle, in the opening
    only; score = the sum of the weights of the principles the move serves (lexicographic by PRINCIPLES)."""
    if not in_opening(b):
        return {}
    out = {}
    for m in (moves if moves is not None else b.legal_moves):
        served = [name for name, _ in PRINCIPLES if FUNCS[name](b, m)]
        if served:
            score = float(sum(w for name, w in PRINCIPLES if name in served))
            out[m.uci()] = (score, "opening subgoals: " + ", ".join(served))
    return out
