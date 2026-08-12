#!/usr/bin/env python
"""fastboard.py -- Rust movegen for the search hot path (Kaveh 2026-08-12: "let's do the eng
fix with a rust chess framework"). Wraps cozy-chess (analog-hors' Rust movegen, PyO3-bound via
cozy-chess-py): measured 13.9x on movegen+apply and 3.1x on tokenization vs python-chess, with
token-exact agreement with the jepa tokenizer.

Semantics notes, stated:
  * cozy castling moves are king-takes-own-rook (Chess960 convention); uci() converts to
    standard e1g1/e1c1 form so every consumer keeps speaking python-chess uci.
  * terminal_value covers mate, stalemate, the 50-move rule (halfmove_clock >= 100) and simple
    insufficient material (<=1 minor total). In-search REPETITION draws are not detected
    (copies carry no history) -- same practical behavior as fresh-board expansion, and the
    root game loop still uses python-chess with full claim-draw rules.
  * key() is cozy's zobrist-style hash incl. side to move -- a faster eval-cache key than fen.
"""
from __future__ import annotations

import copy

import numpy as np

import cozy_chess as cc

_PIECES = (cc.Piece.Pawn, cc.Piece.Knight, cc.Piece.Bishop, cc.Piece.Rook,
           cc.Piece.Queen, cc.Piece.King)
_SQ = {s.lower(): getattr(cc.Square, s) for s in dir(cc.Square)
       if len(s) == 2 and s[0] in "ABCDEFGH"}
_FILES = "abcdefgh"


class FB:
    """thin cozy-chess board with exactly the surface the coherent search needs."""
    __slots__ = ("b",)

    def __init__(self, b):
        self.b = b

    @classmethod
    def from_chess(cls, board):
        return cls(cc.Board.from_fen(board.fen()))

    @property
    def turn(self):
        return self.b.side_to_move() == cc.Color.White \
            if callable(getattr(self.b, "side_to_move", None)) else \
            self.b.side_to_move == cc.Color.White

    def key(self):
        return self.b.hash()

    def fen(self):
        return self.b.fen()

    def legal_count(self):
        return len(self.b.generate_moves())

    def uci(self, mv):
        u = str(mv)
        f, t = u[:2], u[2:4]
        if (self.b.piece_on(_SQ[f]) == cc.Piece.King
                and self.b.piece_on(_SQ[t]) == cc.Piece.Rook
                and self.b.color_on(_SQ[t]) == (cc.Color.White if self.turn
                                                else cc.Color.Black)):
            return f + (("g" + f[1]) if t[0] > f[0] else ("c" + f[1]))
        return u

    def children(self):
        """[(standard_uci, child FB)] for every legal move."""
        out = []
        for mv in self.b.generate_moves():
            b2 = copy.copy(self.b)
            b2.play(mv)
            out.append((self.uci(mv), FB(b2)))
        return out

    def tok_glob(self):
        """(tok (64,) uint8, glob (6,) uint8) -- byte-identical to jepa.tokenize()."""
        tok = np.zeros(64, np.uint8)
        for base, col in ((1, cc.Color.White), (7, cc.Color.Black)):
            for k, pc in enumerate(_PIECES):
                m = int(self.b.colored_pieces(col, pc))
                while m:
                    tok[(m & -m).bit_length() - 1] = base + k
                    m &= m - 1
        crw = self.b.castle_rights(cc.Color.White)
        crb = self.b.castle_rights(cc.Color.Black)
        ep = self.b.en_passant()
        glob = np.array([1 if self.turn else 0,
                         1 if crw.short is not None else 0,
                         1 if crw.long is not None else 0,
                         1 if crb.short is not None else 0,
                         1 if crb.long is not None else 0,
                         0 if ep is None else _FILES.index(str(ep)) + 1], np.uint8)
        return tok, glob

    def n_pieces(self):
        return bin(int(self.b.occupied())).count("1")

    def terminal_value(self, mate, tb_probe=None):
        """mover-POV exact value at game end, or None. Mirrors KittyChess._terminal_value;
        tb_probe(fen) -> value handles the <=5-piece tablebase region (rare -> fen convert)."""
        try:
            st = self.b.status()
        except BaseException:                          # pyo3 PanicException on ILLEGAL boards
            return 0.0                                 # (kings adjacent etc); score as dead draw
        if st == cc.GameStatus.Won:                    # cozy: the side to move is mated
            return -mate
        if st == cc.GameStatus.Drawn:                  # stalemate
            return 0.0
        if self.b.halfmove_clock >= 100:
            return 0.0
        np_ = self.n_pieces()
        if np_ <= 3:
            heavy = any(int(self.b.colored_pieces(c, p))
                        for c in (cc.Color.White, cc.Color.Black)
                        for p in (cc.Piece.Pawn, cc.Piece.Rook, cc.Piece.Queen))
            if not heavy:
                return 0.0                             # K vs K (+ single minor)
        if tb_probe is not None and np_ <= 5:
            return tb_probe(self.fen())
        return None
