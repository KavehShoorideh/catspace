"""
data/encode.py — packed-bitboard board codec: 12 uint64 planes (96 bytes) per
position instead of 768-byte/float unpacked planes, decoded to (N,12,8,8)
uint8 with ONE vectorized np.unpackbits call per batch -- the lc0-chunkparser
pattern for keeping full-board shards laptop-sized (store packed, unpack on
the fly).
"""
from __future__ import annotations

import chess
import numpy as np

# 12 planes: (piece type, color), white pieces first then black -- fixed order.
PLANES = [(pt, color) for color in (chess.WHITE, chess.BLACK)
          for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)]

# meta layout: [side-to-move(0=W/1=B), K, Q, k, q castling rights, ep_file+1
# (0=none), min(halfmove_clock,255), reserved]
META_LEN = 8


def encode_packed(board: chess.Board) -> np.ndarray:
    """(12,) uint64 -- one bitboard per (piece type, color) plane, bit s = square s."""
    return np.array([board.pieces_mask(pt, color) for pt, color in PLANES], dtype=np.uint64)


def encode_meta(board: chess.Board) -> np.ndarray:
    """(8,) uint8 side/castling/en-passant/halfmove-clock summary."""
    ep_file = (chess.square_file(board.ep_square) + 1) if board.ep_square is not None else 0
    return np.array([
        int(board.turn == chess.BLACK),
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
        ep_file,
        min(board.halfmove_clock, 255),
        0,
    ], dtype=np.uint8)


def decode_planes(packed: np.ndarray) -> np.ndarray:
    """(...,12) uint64 -> (...,12,8,8) uint8. bit index s (0=a1..63=h8) lands
    at [..., p, s // 8, s % 8] -- i.e. row = rank, col = file."""
    packed = np.asarray(packed, dtype=np.uint64)
    bits = np.unpackbits(packed[..., None].view(np.uint8), axis=-1, bitorder="little")
    return bits.reshape(*packed.shape, 8, 8)


def board_from_packed(packed: np.ndarray, meta: np.ndarray) -> chess.Board:
    """Reconstruct a Board from one packed-plane row + its meta row. Used only
    by round-trip tests -- the training path never needs to go back to a
    python-chess Board."""
    board = chess.Board(None)
    board.turn = chess.BLACK if int(meta[0]) else chess.WHITE
    for (pt, color), mask in zip(PLANES, packed.tolist()):
        for sq in chess.SquareSet(int(mask)):
            board.set_piece_at(sq, chess.Piece(pt, color))

    fen = "".join(c for c, bit in zip("KQkq", meta[1:5]) if bit)
    board.set_castling_fen(fen or "-")

    if meta[5]:
        file = int(meta[5]) - 1
        rank = 5 if board.turn == chess.WHITE else 2
        board.ep_square = chess.square(file, rank)
    board.halfmove_clock = int(meta[6])
    return board
