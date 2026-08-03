"""catspace/controlfield/wdl_decay.py -- Kaveh's pivot (2026-08-02): don't use
the hand-coded control field C to judge whether a move is good -- C (and SEE
underneath it) is structurally blind to follow-up tactics (a sacrifice that only
pays off via a fork two moves later is invisible to a static, single-square
exchange evaluation). Use the DECAY of Stockfish's own WDL/committor instead:
play the candidate move, then continue with Stockfish's own best play for both
sides for a few more of the mover's turns, and ask whether the mover's winning
probability holds up or collapses. This is real search-based ground truth, not
a hand-coded heuristic -- it can see whatever Stockfish can see, arbitrarily far
ahead, unlike SEE/C which see nothing past the first move.

This generalizes what catspace/research/components/encoder/approaches/control_field_wdl/experiments/controlfield_gate3_decay.py did for 3
hand-picked gambits into a reusable per-move test.
"""
from __future__ import annotations

import chess
import chess.engine


def committor(eng: chess.engine.SimpleEngine, board: chess.Board, depth: int, pov: chess.Color) -> float:
    info = eng.analyse(board, chess.engine.Limit(depth=depth))
    w = info["wdl"].pov(pov)
    tot = max(1, w.wins + w.draws + w.losses)
    return w.wins / tot


def best_move(eng: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> chess.Move:
    info = eng.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
    return info[0]["pv"][0]


def _committor_or_terminal(eng, b, depth, mover):
    """committor(), but handles terminal positions (checkmate/stalemate/etc)
    where SF has nothing to analyse and `info['wdl']` isn't present at all --
    this is the exact bug that silently ate 18/20 mateIn2 puzzles on first run
    (mate-in-2 solutions routinely END the game on the mover's own 2nd move)."""
    if b.is_checkmate():
        return 1.0 if b.turn != mover else 0.0   # side NOT to move just got mated
    if b.is_game_over():
        return 0.5   # stalemate/insufficient material/repetition -> treat as a draw
    return committor(eng, b, depth, mover)


def wdl_decay_trend(eng: chess.engine.SimpleEngine, board: chess.Board, move: chess.Move,
                     k_moves: int = 4, depth: int = 12):
    """Play `move`, then continue with Stockfish's own best play for BOTH sides
    for up to `k_moves` more turns of the ORIGINAL mover. -> list of committor
    values (mover POV) at each of the mover's turns, starting right after
    `move`. First value is the baseline (no further play needed to observe);
    trend = committors[-1] - committors[0]. Stops early on game-over (a mate-in-2
    solution routinely ends the game on the mover's own 2nd move -- must check
    terminal state after EVERY push, not just the opponent's)."""
    mover = board.turn
    b = board.copy()
    b.push(move)
    committors = [_committor_or_terminal(eng, b, depth, mover)]
    for _ in range(k_moves - 1):
        if b.is_game_over():
            break
        # opponent's best reply
        b.push(best_move(eng, b, depth))
        if b.is_game_over():
            committors.append(_committor_or_terminal(eng, b, depth, mover))
            break
        # mover's own best follow-up
        b.push(best_move(eng, b, depth))
        committors.append(_committor_or_terminal(eng, b, depth, mover))
        if b.is_game_over():
            break
    return committors


def is_good_by_decay(eng: chess.engine.SimpleEngine, board: chess.Board, move: chess.Move,
                      k_moves: int = 4, depth: int = 12, decay_tol: float = 0.0) -> tuple[bool, list]:
    """-> (is_good, committor_trajectory). is_good = committor does not decay
    below `decay_tol` (trend >= -decay_tol) over the tracked continuation --
    i.e. Stockfish's own best play keeps the winning chances from collapsing.
    No control-field computation anywhere in this function."""
    traj = wdl_decay_trend(eng, board, move, k_moves, depth)
    if len(traj) < 2:
        return True, traj   # nothing to decay yet (game ended immediately, e.g. mate)
    trend = traj[-1] - traj[0]
    return trend >= -decay_tol, traj
