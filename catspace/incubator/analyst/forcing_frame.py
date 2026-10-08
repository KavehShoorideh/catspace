"""THE FORCING FRAME: the exchange joined with every capture elsewhere, and (optionally) checks -- a frame on the board
whose moves are the forcing ones (Kaveh 2026-10-08: "the exchange should include intermezzos ... a higher value threat
somewhere else"; the replay's surprises: captures elsewhere carry 59% of the frame's error, checks in between 39%).

A state is (board, plies left, checks left). The side to move may EXIT (stand pat) unless it is in check; its moves are
its captures (any square, promotions included) and, while `checks left` > 0, its checking moves; in check, every legal
evasion. Mate is -MATE for the mated side; at the horizon the side to move stands pat (0) unless mated. Values are
material in pawns, negamax, searched by frame_search.FrameSearch (lemmas, priors, supports) like any frame. This is the
classical quiescence search with checks, seen as one frame among others: its moves are the board's forcing moves.
"""
from __future__ import annotations

import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
MATE = 20.0


class ForcingFrame:
    def __init__(self, checks: bool = True):
        self.checks = checks

    def moves(self, st):
        b, plies, checks = st
        if plies <= 0:
            return []
        if b.is_check():
            return list(b.legal_moves)
        out = [m for m in b.legal_moves if b.is_capture(m) or m.promotion]
        if self.checks and checks > 0:
            out += [m for m in b.legal_moves if not b.is_capture(m) and not m.promotion and b.gives_check(m)]
        victim = lambda m: VAL[(b.piece_at(m.to_square) or chess.Piece(chess.PAWN, not b.turn)).piece_type] if b.is_capture(m) else 0
        return sorted(out, key=lambda m: (-victim(m), VAL[b.piece_at(m.from_square).piece_type]))   # most valuable victim first

    def gain(self, st, m):
        b = st[0]
        g = 0.0
        if b.is_capture(m):
            g += VAL[(b.piece_at(m.to_square) or chess.Piece(chess.PAWN, not b.turn)).piece_type]
        if m.promotion:
            g += VAL[m.promotion] - 1
        return g

    def next(self, st, m):
        b, plies, checks = st
        nb = b.copy(stack=False); nb.push(m)
        quiet_check = not b.is_capture(m) and not m.promotion and not b.is_check()
        return (nb, plies - 1, checks - (1 if quiet_check else 0))

    def exit(self, st):
        b, plies, _ = st
        if b.is_check():
            return -MATE if (plies <= 0 and b.is_checkmate()) or plies > 0 else 0.0   # in check: no standing pat
        return 0.0

    def key(self, st):
        b, plies, checks = st
        return (b.board_fen(), b.turn, b.castling_rights, b.ep_square, plies, checks)


def forcing_lemmas():
    from .frame_search import Lemma

    def stand(st):                                       # not in check, the side to move can always stand pat
        return None if st[0].is_check() else (0.0, float("inf"))
    return [Lemma("stand pat", "proved: the exit is a move of the frame (FrameSearch.exit)", state=stand)]


def capture_value(b: chess.Board, s: int, plies: int = 8, checks: int = 2, use_checks: bool = True) -> tuple:
    """The value for the side to move of capturing on s (its best legal capture there), in the forcing frame.
    -> (value, nodes)."""
    from .frame_search import FrameSearch
    fs = FrameSearch(ForcingFrame(use_checks), forcing_lemmas())
    fr = fs.frame; st = (b, plies, checks)
    best = None
    for m in b.legal_moves:
        if m.to_square == s and b.is_capture(m):
            v = fr.gain(st, m) - fs.value(fr.next(st, m))
            best = v if best is None else max(best, v)
    return (best if best is not None else 0.0), fs.nodes
