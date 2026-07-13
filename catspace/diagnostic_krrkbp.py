"""
diagnostic_krrkbp.py — the K+R+R (white) vs K+B+P (black) endgame diagnostic
(2026-07-12, Kaveh's design): can the plan hold/convert an overwhelming but
technically simple material edge against a bishop's color-bound geometry?
The concept under test -- keep the rooks on squares the bishop can never
reach -- is a crisp, nameable technique, not a vague "is it strong" signal.

random_krrkbp() generates a random LEGAL position with this exact material
signature: White K+R+R, Black K + a bishop constrained to ONE fixed color
complex (so "the bishop's color" is a constant across the whole sampled set,
not a per-position roll) + a pawn on the e-file. White to move, no castling
rights (doesn't apply to this material).

Kaveh's explicit methodology requirement: any comparison between algorithms
must use the SAME distribution of starting placements, or the comparison is
biased by chance differences in starting difficulty -- matches the
matched-seed pairing already used in catspace/abtest.py. build_fixed_set()
generates a set ONCE with a fixed seed and is the thing every future
algorithm/config comparison reuses, never regenerated per-run.
"""
from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np


def _square_is_light(sq: int) -> bool:
    return (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1


def random_krrkbp(rng: np.random.Generator, bishop_light_squared: bool = True,
                  pawn_rank_range: tuple = (3, 6), max_tries: int = 500) -> chess.Board:
    """One random legal KRR(white) vs KBP(black) position. Raises RuntimeError
    if max_tries is exhausted (should not happen in practice -- the
    constraints are loose relative to the 64-square board)."""
    e_file = chess.FILE_NAMES.index("e")
    pawn_candidates = [s for s in chess.SQUARES if chess.square_file(s) == e_file
                       and pawn_rank_range[0] <= chess.square_rank(s) <= pawn_rank_range[1]]

    for _ in range(max_tries):
        order = [int(s) for s in rng.permutation(64)]
        pawn_sq = pawn_candidates[int(rng.integers(len(pawn_candidates)))]
        used = {pawn_sq}

        bishop_sq = next((s for s in order if s not in used
                          and _square_is_light(s) == bishop_light_squared), None)
        if bishop_sq is None:
            continue
        used.add(bishop_sq)

        remaining = [s for s in order if s not in used]
        if len(remaining) < 4:
            continue
        wk_sq, bk_sq, wr1_sq, wr2_sq = remaining[:4]
        if chess.square_distance(wk_sq, bk_sq) <= 1:
            continue

        board = chess.Board(None)
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(wr1_sq, chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(wr2_sq, chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(bishop_sq, chess.Piece(chess.BISHOP, chess.BLACK))
        board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, chess.BLACK))
        board.turn = chess.WHITE
        board.castling_rights = chess.BB_EMPTY
        board.ep_square = None
        board.halfmove_clock = 0
        board.fullmove_number = 1

        if not board.is_valid():
            continue
        if board.is_check():          # skip starts where White is already in check
            continue
        if board.is_game_over(claim_draw=True):
            continue
        return board
    raise RuntimeError(f"could not generate a valid KRRvKBP position in {max_tries} tries")


def build_fixed_set(n: int, seed: int, bishop_light_squared: bool = True,
                    pawn_rank_range: tuple = (3, 6)) -> list:
    rng = np.random.default_rng(seed)
    return [random_krrkbp(rng, bishop_light_squared, pawn_rank_range) for _ in range(n)]


def save_fixed_set(boards: list, path) -> None:
    Path(path).write_text(json.dumps({"fens": [b.fen() for b in boards]}, indent=2))


def load_fixed_set(path) -> list:
    data = json.loads(Path(path).read_text())
    return [chess.Board(fen) for fen in data["fens"]]
