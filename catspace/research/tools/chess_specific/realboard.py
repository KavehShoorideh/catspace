"""
realboard.py — the real-board (python-chess) analogue of the toy game layer.
The chain Policy/Opponent protocols are index-based and don't apply to 8x8
boards, so real play gets its own minimal seam:

    BoardPolicy.move(board, rng) -> chess.Move

play_board_game handles opening randomization (seeded random plies for start
diversification -- the paired-arena analogue of toy start sampling), the ply
cap, and result extraction. No torch here; FBBoardPolicy lives in nn/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import chess
import numpy as np


class BoardPolicy(Protocol):
    def move(self, board: chess.Board, rng: np.random.Generator) -> chess.Move: ...


class RandomBoardPolicy:
    def move(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        legal = list(board.legal_moves)
        return legal[int(rng.integers(0, len(legal)))]


@dataclass
class BoardGameRecord:
    moves: list = field(default_factory=list)     # UCI strings, opening included
    opening_plies: int = 0
    result: str = "*"                             # "1-0" | "0-1" | "1/2-1/2" | "*" (cap)
    termination: str = ""
    n_plies: int = 0
    final_fen: str = ""


def play_board_game(white: BoardPolicy, black: BoardPolicy,
                    start: chess.Board | None = None, opening_plies: int = 0,
                    max_plies: int = 300, rng: np.random.Generator | None = None) -> BoardGameRecord:
    """One game from `start` (default: initial position), after `opening_plies`
    seeded-random plies by BOTH sides. claim_draw semantics: threefold/50-move
    end the game as a draw (like online play with auto-claim)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    board = start.copy() if start is not None else chess.Board()
    rec = BoardGameRecord(opening_plies=opening_plies)
    rand = RandomBoardPolicy()

    def push(move: chess.Move):
        rec.moves.append(move.uci())
        board.push(move)

    for _ in range(opening_plies):
        if board.is_game_over(claim_draw=True):
            break
        push(rand.move(board, rng))

    while len(rec.moves) < max_plies:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            break
        player = white if board.turn == chess.WHITE else black
        push(player.move(board, rng))

    outcome = board.outcome(claim_draw=True)
    rec.result = outcome.result() if outcome is not None else "*"
    rec.termination = outcome.termination.name if outcome is not None else "PLY_CAP"
    rec.n_plies = len(rec.moves)
    rec.final_fen = board.fen()
    return rec


def record_to_pgn(rec: BoardGameRecord, white_name: str, black_name: str) -> "chess.pgn.Game":
    import chess.pgn
    game = chess.pgn.Game()
    game.headers["White"], game.headers["Black"] = white_name, black_name
    game.headers["Result"] = rec.result
    node = game
    board = chess.Board()
    for uci in rec.moves:
        move = chess.Move.from_uci(uci)
        node = node.add_variation(move)
        board.push(move)
    return game
