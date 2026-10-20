"""PLAN 53 (1): THE STOCKFISH VERDICT on a decision -- the training signal of the question prior (Kaveh 2026-10-20: "see if the deciding line was in
agreement with Stockfish"). Stockfish is the GRADER here, never a premise: nothing of the agenda reads it.

verdict(fen, uci, depth) -> (verdict, loss_cp, best_uci): RIGHT when the decision is Stockfish's best move or loses at most `TOL` centipawns against
it (from the mover's view), WRONG otherwise; UNKNOWN when there is no decision or the engine is missing. One engine per process."""
from __future__ import annotations

import os

import chess
import chess.engine

STOCKFISH = "/opt/homebrew/bin/stockfish"
TOL = 50
_engine = [None]


def _get():
    if _engine[0] is None and os.path.exists(STOCKFISH):
        _engine[0] = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    return _engine[0]


def close() -> None:
    if _engine[0] is not None:
        _engine[0].quit()
        _engine[0] = None


def verdict(fen: str, uci: str | None, depth: int = 14) -> tuple:
    e = _get()
    if e is None or not uci:
        return "UNKNOWN", None, None
    b = chess.Board(fen)
    best = e.analyse(b, chess.engine.Limit(depth=depth))
    best_mv = best["pv"][0]
    best_cp = best["score"].pov(b.turn).score(mate_score=10000)
    if best_mv.uci()[:4] == uci[:4]:
        return "RIGHT", 0, best_mv.uci()
    c = b.copy()
    c.push_uci(uci)
    after = e.analyse(c, chess.engine.Limit(depth=depth))
    ours_cp = after["score"].pov(b.turn).score(mate_score=10000)
    loss = best_cp - ours_cp
    return ("RIGHT" if loss <= TOL else "WRONG"), loss, best_mv.uci()
