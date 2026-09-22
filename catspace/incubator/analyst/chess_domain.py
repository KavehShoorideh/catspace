"""The CHESS ADAPTER for lemma_gen: the only file that knows what a piece is. It supplies the rules and nothing
else -- entity types, relations as programs over a board, actions and the transition, a scalar (material), and
placements (every board holding exactly the frame's pieces plus both kings, canonicalized by the board's symmetry).
No pawns in this first version: the eightfold symmetry then holds and the frame is orientation-free."""
from __future__ import annotations

import itertools
import random

import chess

from .lemma_gen import Domain, Relation

VAL = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0, chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 100.0}
KIND = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT}
TYPES = ["wK", "wQ", "wR", "wB", "wN", "bK", "bQ", "bR", "bB", "bN"]
CANON = [chess.A1, chess.B1, chess.C1, chess.D1, chess.B2, chess.C2, chess.D2, chess.C3, chess.D3, chess.D4]


class State:
    __slots__ = ("board", "ids")

    def __init__(self, board: chess.Board, ids: list):
        self.board = board; self.ids = ids                                # ids = the slot entities' squares, in frame order

    def fen(self):
        return self.board.fen()


def _piece(t: str) -> chess.Piece:
    return chess.Piece(KIND[t[1]], t[0] == "w")


def placements(types: tuple, cap: int, seed: int):
    """Boards holding exactly the frame's entities plus both kings (added if the frame does not name them). The
    first slot's square is fixed to the ten canonical squares (symmetry). Exhaustive when the raw count fits the
    cap, else a uniform sample with the flag False. White to move; actions flip the turn to the actor's side."""
    pieces = [_piece(t) for t in types]
    extra = []
    if not any(p.piece_type == chess.KING and p.color for p in pieces):
        extra.append(chess.Piece(chess.KING, True))
    if not any(p.piece_type == chess.KING and not p.color for p in pieces):
        extra.append(chess.Piece(chess.KING, False))
    allp = pieces + extra
    n = len(allp)
    raw = len(CANON) * 1
    for i in range(1, n):
        raw *= (64 - i)
    exhaustive = raw <= cap * 2
    rng = random.Random(seed)
    yielded = 0
    if exhaustive:
        it = ((c,) + rest for c in CANON for rest in itertools.permutations([s for s in range(64)], n - 1) if c not in rest)
    else:
        def sampler():
            while True:
                c = rng.choice(CANON)
                rest = rng.sample([s for s in range(64) if s != c], n - 1)
                yield (c,) + tuple(rest)
        it = sampler()
    for squares in it:
        if not exhaustive and yielded >= cap:
            return
        if exhaustive and yielded >= cap * 2:
            return
        b = chess.Board(None)
        for sq, p in zip(squares, allp):
            b.set_piece_at(sq, p)
        b.turn = chess.WHITE
        if not b.is_valid():
            continue
        if b.is_check():
            continue
        # the black king must not be attacked either (a frame is a quiet configuration for both sides)
        b2 = b.copy(stack=False); b2.turn = chess.BLACK
        if not b2.is_valid() or b2.is_check():
            continue
        yielded += 1
        yield State(b, list(squares[:len(types)])), exhaustive


def slots_of(state: State):
    return state.ids


def actions(state: State, e: int):
    b = state.board.copy(stack=False); b.turn = state.board.piece_at(e).color
    if not b.is_valid():
        return []
    return [(b, m) for m in b.legal_moves if m.from_square == e]


def transition(state: State, action):
    b, m = action
    nb = b.copy(stack=False); nb.push(m)
    ids = [m.to_square if s == m.from_square else s for s in state.ids]
    return State(nb, ids)


def _material(board: chess.Board, side: chess.Color) -> float:
    return sum(VAL[p.piece_type] for p in board.piece_map().values() if p.color == side and p.piece_type != chess.KING) - \
           sum(VAL[p.piece_type] for p in board.piece_map().values() if p.color != side and p.piece_type != chess.KING)


def scalar(state: State, e: int) -> float:
    p = state.board.piece_at(e)
    return _material(state.board, p.color)


def reply_scalar(after: State, e: int) -> float:
    """The entity's side's material after the opponent's best immediate reply (the reply that minimizes it)."""
    b = after.board
    side = not b.turn                                                 # the actor's side: it just moved
    if b.is_checkmate():                                              # NOT is_game_over(): insufficient material would end K+B vs K before the king captures (2026-09-22)
        return _material(b, side) + (-1000.0 if b.turn == side else 1000.0)
    if b.is_stalemate():
        return _material(b, side)
    worst = None
    for r in b.legal_moves:
        nb = b.copy(stack=False); nb.push(r)
        v = _material(nb, side) + (-1000.0 if nb.is_checkmate() else 0.0)
        worst = v if worst is None else min(worst, v)
    return worst if worst is not None else _material(b, side)


# ---------------------------------------------------------------- relations: programs over the board, nothing named
def same_side(state, i, j):
    return state.board.piece_at(i).color == state.board.piece_at(j).color


def attacks(state, i, j):
    return j in state.board.attacks(i)


def defends(state, i, j):
    return same_side(state, i, j) and j in state.board.attacks(i)


def aligned(state, i, j):
    return bool(chess.SquareSet.ray(i, j)) and not any(state.board.piece_at(s) for s in chess.SquareSet.between(i, j))


def between(state, i, j, k):
    """j on the segment i..k, nothing else between."""
    if not chess.SquareSet.ray(i, k) or j not in chess.SquareSet.between(i, k):
        return False
    return not any(state.board.piece_at(s) for s in chess.SquareSet.between(i, k) if s != j)


def worth_ge(state, i, j):
    return VAL[state.board.piece_at(i).piece_type] >= VAL[state.board.piece_at(j).piece_type]


def attacked(state, i):
    p = state.board.piece_at(i)
    return bool(state.board.attackers(not p.color, i))


def defended(state, i):
    p = state.board.piece_at(i)
    return bool(state.board.attackers(p.color, i))


def slides(state, i):
    return state.board.piece_at(i).piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP)


# action-level: fn(state, action, *slots); the action is (board, move)
def gives_check(state, action):
    b, m = action; nb = b.copy(stack=False); nb.push(m); return nb.is_check()


def captures(state, action):
    b, m = action; return b.is_capture(m)


def captures_worth_ge(state, action, j):
    b, m = action
    if not b.is_capture(m):
        return False
    return VAL[b.piece_at(m.to_square).piece_type] >= VAL[b.piece_at(j).piece_type]


def stays_aligned(state, action, j):
    """After the action the mover is still on a line with slot j (nothing between)."""
    b, m = action
    if m.to_square == j:
        return False
    nb = b.copy(stack=False); nb.push(m)
    return bool(chess.SquareSet.ray(m.to_square, j)) and not any(nb.piece_at(s) for s in chess.SquareSet.between(m.to_square, j))


def lands_attacked(state, action):
    b, m = action; nb = b.copy(stack=False); nb.push(m)
    return bool(nb.attackers(nb.turn, m.to_square))


def domain() -> Domain:
    rels = [Relation("same_side", 2, fn=same_side), Relation("attacks", 2, fn=attacks), Relation("defends", 2, fn=defends),
            Relation("aligned", 2, fn=aligned), Relation("between", 3, fn=between), Relation("worth_ge", 2, fn=worth_ge),
            Relation("attacked", 1, fn=attacked), Relation("defended", 1, fn=defended), Relation("slides", 1, fn=slides),
            Relation("gives_check", 0, True, gives_check), Relation("captures", 0, True, captures),
            Relation("captures_worth_ge", 1, True, captures_worth_ge), Relation("stays_aligned", 1, True, stays_aligned),
            Relation("lands_attacked", 0, True, lands_attacked)]
    return Domain("chess", TYPES, rels, placements, slots_of, actions, transition, scalar, reply_scalar)
