"""PLAN 54: THE PLAY SERVER -- play against the agenda, watch it think.

Item 1 (this block): the mover. `AgendaPlayer.move(board)` builds an Agenda on the board's FEN, asks, and plays the decision's first move; when the
agenda has no decision it falls back to a NAMED player (one_ply = the rules-based frame_player.choose, stockfish, or random). The server owns the full
board with its move stack (50-move clock, repetition); the agenda sees the FEN only (a known gap, [[game_context_in_frames]]). One lock around every ask():
`R.reset_ids()` is a module global, so one Agenda at a time.

  .venv/bin/python -m catspace.incubator.analyst.web --play [--fallback one_ply|stockfish|random]
"""
from __future__ import annotations

import random
import threading
import time

import chess
import chess.engine

from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import state_of

AGENDA_LOCK = threading.Lock()                       # R.reset_ids() is a module global: one agenda at a time
STOCKFISH = "/opt/homebrew/bin/stockfish"
FALLBACKS = ("one_ply", "stockfish", "random")
CLOCK_NOTE_AT = 70                                   # half-move clock (plies) from which the 50-move rule is within 30 plies


def context_notes(board: chess.Board) -> list[str]:
    """What the agenda cannot see (it reads the FEN only): the 50-move clock and a repetition that is one move away."""
    notes = []
    if board.halfmove_clock >= CLOCK_NOTE_AT:
        notes.append(f"game context the agenda does not see: the 50-move clock is {board.halfmove_clock} plies (draw claimable at 100)")
    for mv in board.legal_moves:
        board.push(mv)
        try:
            if board.is_repetition(3):
                notes.append(f"game context the agenda does not see: {board.peek().uci()} would repeat the position a third time")
                break
        finally:
            board.pop()
    return notes


class AgendaPlayer:
    def __init__(self, fallback: str = "one_ply", seed: int = 0, stockfish_movetime: float = 0.1):
        if fallback not in FALLBACKS:
            raise ValueError(f"fallback must be one of {FALLBACKS}")
        self.fallback = fallback
        self.rng = random.Random(seed)
        self.sf_movetime = stockfish_movetime
        self._sf = None
        BV.install_terms()

    # ---- the fallbacks (each returns a legal Move)
    def _fallback_move(self, board: chess.Board) -> chess.Move:
        if self.fallback == "random":
            return self.rng.choice(list(board.legal_moves))
        if self.fallback == "stockfish":
            if self._sf is None:
                self._sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
            return self._sf.play(board, chess.engine.Limit(time=self.sf_movetime)).move
        from catspace.incubator.analyst.frame_player import choose
        best = choose(board.copy())["best"]
        return chess.Move.from_uci(self.rng.choice(best))

    def close(self):
        if self._sf is not None:
            try:
                self._sf.quit()
            except Exception:
                pass
            self._sf = None

    def move(self, board: chess.Board, lines=None, wrap_lines=None):
        """-> (move, trace_lines, source). source is 'agenda' or 'fallback:<name>'. `wrap_lines(ag)` may replace ag.lines (item 2)."""
        out: list[str] = []
        for n in context_notes(board):
            out.append(n)
        fen = board.fen()
        us = "w" if board.turn else "b"
        decision = None
        ag = None
        with AGENDA_LOCK:
            ag = Agenda(state_of(fen, us))                    # chase_log stays unset in a server
            if wrap_lines is not None:
                wrap_lines(ag)
            ag.ask()
            decision = ag.decision()
            agenda_lines = list(ag.lines)
        move = None
        if decision is not None and decision.move:
            try:
                cand = chess.Move.from_uci(decision.move)
                if cand in board.legal_moves:
                    move = cand
            except ValueError:
                move = None
            if move is None:
                out.append(f"agenda decision {decision.what} carries move {decision.move!r} which is not legal here; falling back")
        if move is not None:
            out.append(f"decision: agenda chose {board.san(move)} ({decision.kind} {decision.what}, value {decision.value}, lemma {decision.lemma})")
            return move, out + agenda_lines, "agenda"
        move = self._fallback_move(board)
        head = f"no decision: fallback {self.fallback} chose {board.san(move)}" if decision is None else f"fallback {self.fallback} chose {board.san(move)}"
        return move, out + [head], f"fallback:{self.fallback}"
