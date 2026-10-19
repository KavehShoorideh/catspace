"""PLAN 54: THE PLAY SERVER -- play against the agenda, watch it think.

Item 1 (this block): the mover. `AgendaPlayer.move(board)` builds an Agenda on the board's FEN, asks, and plays the decision's first move; when the
agenda has no decision it falls back to a NAMED player (one_ply = the rules-based frame_player.choose, stockfish, or random). The server owns the full
board with its move stack (50-move clock, repetition); the agenda sees the FEN only (a known gap, [[game_context_in_frames]]). One lock around every ask():
`R.reset_ids()` is a module global, so one Agenda at a time.

  .venv/bin/python -m catspace.incubator.analyst.web --play [--fallback one_ply|stockfish|random]
"""
from __future__ import annotations

import random
import re
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


class TraceLines(list):
    """Drop-in for Agenda.lines: every line appended (append, extend, +=, insert) also calls `cb(text)`; a text with newlines is one call per line."""

    def __init__(self, cb):
        super().__init__()
        self.cb = cb

    def _emit(self, x):
        for part in str(x).split("\n"):
            self.cb(part)

    def append(self, x):
        super().append(x); self._emit(x)

    def extend(self, xs):
        for x in xs:
            self.append(x)

    def __iadd__(self, xs):
        self.extend(xs)
        return self

    def insert(self, i, x):
        super().insert(i, x); self._emit(x)


# ---------------------------------------------------------------- trace records (item 2)
_SAN = r"(?:O-O-O|O-O|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h]x[a-h][1-8](?:=[QRBN])?|[a-h][1-8](?:=[QRBN])?)[+#]?"
SAN_RE = re.compile(r"(?<![A-Za-z0-9])" + _SAN + r"(?![A-Za-z0-9])")
SQUARE_RE = re.compile(r"(?<![A-Za-z0-9])[a-h][1-8](?![A-Za-z0-9])")
KINDS = ("claim", "kill", "gap", "stop", "fuse", "chase", "note", "question", "route", "their", "decision", "fallback", "context", "state", "other")


def classify(text: str) -> str:
    t = text.strip()
    if t.startswith("decision:"):
        return "decision"
    if t.startswith("no decision:") or t.startswith("fallback "):
        return "fallback"
    if t.startswith("game context"):
        return "context"
    if "CLAIM" in t:
        return "claim"
    if "KILLED" in t or "killed" in t:
        return "kill"
    if "GAP" in t:
        return "gap"
    if t.startswith("STOP"):
        return "stop"
    if "fuse" in t:
        return "fuse"
    if t.startswith("chase:") or "chase:" in t:
        return "chase"
    if t.startswith(("note:", "memory:", "recall:")):
        return "note"
    if re.match(r"^(->\s*)?Q\d+", t):
        return "question"
    if t.startswith("EV"):
        return "state"
    if re.match(r"^\s+\S+:", text):
        return "route"
    if "their" in t or "parry" in t or "parried" in t:
        return "their"
    return "other"


def parse_moves(text: str, root: chess.Board) -> list[dict]:
    """Every SAN token of the line, parsed against the board the line is about: the root board, advanced by each move it parses (so "X then Y"
    reads Y after X); a token that does not parse there is tried on the root; a bare square token (a pawn push like 'a7') counts as a move only when
    it opens the line or follows 'then' / 'after' / 'verify' (else 'their P on a7' would draw an arrow). Best effort: never raises."""
    out: list[dict] = []
    try:
        b = root.copy(stack=False)
        for m in SAN_RE.finditer(text):
            tok = m.group(0)
            bare = re.fullmatch(r"[a-h][1-8]", tok) is not None
            if bare:
                before = text[:m.start()].rstrip().split()
                if before and before[-1] not in ("then", "after", "verify", "->") and not (text[:m.start()].strip() == ""):
                    continue
            mv = None
            for board in (b, root):
                try:
                    mv = board.parse_san(tok)
                except ValueError:
                    continue
                if board is root and b is not root:
                    b = root.copy(stack=False)
                break
            if mv is None:
                continue
            rec = dict(san=tok, **{"from": chess.square_name(mv.from_square), "to": chess.square_name(mv.to_square)})
            if not any(o["from"] == rec["from"] and o["to"] == rec["to"] for o in out):
                out.append(rec)
            if mv in b.legal_moves:
                b.push(mv)
    except Exception:
        pass
    return out


def make_record(n: int, text: str, root: chess.Board, ply: int = 0) -> dict:
    return dict(n=n, ply=ply, text=text, kind=classify(text), indent=len(text) - len(text.lstrip(" ")),
                moves=parse_moves(text, root), squares=sorted(set(SQUARE_RE.findall(text))))


class TraceBuffer:
    """A per-game ring buffer of trace records; n counts from 1 and never repeats; records(since) returns those with n > since."""

    def __init__(self, cap: int = 6000):
        self.cap, self.n, self.items, self.lock = cap, 0, [], threading.Lock()

    def add(self, text: str, root: chess.Board, ply: int = 0) -> dict:
        with self.lock:
            self.n += 1
            rec = make_record(self.n, text, root, ply)
            self.items.append(rec)
            if len(self.items) > self.cap:
                del self.items[: len(self.items) - self.cap]
            return rec

    def since(self, n: int) -> list[dict]:
        with self.lock:
            return [r for r in self.items if r["n"] > n]


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

    def move(self, board: chess.Board, emit=None):
        """-> (move, trace_lines, source). source is 'agenda' or 'fallback:<name>'. `emit(text)` is called for every trace line AS IT IS
        APPENDED (item 2), so a UI can stream while the agenda thinks; trace_lines is the full list in order."""
        out: list[str] = []

        def say(t: str):
            out.append(t)
            if emit:
                emit(t)
        for n in context_notes(board):
            say(n)
        fen = board.fen()
        us = "w" if board.turn else "b"
        with AGENDA_LOCK:
            ag = Agenda(state_of(fen, us))                    # chase_log stays unset in a server
            ag.lines = TraceLines(say)                        # the agenda only ever appends (verified: append / += in agenda.py); no edit to agenda.py
            ag.ask()
            decision = ag.decision()
        move = None
        if decision is not None and decision.move:
            try:
                cand = chess.Move.from_uci(decision.move)
                if cand in board.legal_moves:
                    move = cand
            except ValueError:
                move = None
            if move is None:
                say(f"agenda decision {decision.what} carries move {decision.move!r} which is not legal here; falling back")
        if move is not None:
            say(f"decision: agenda chose {board.san(move)} ({decision.kind} {decision.what}, value {decision.value}, lemma {decision.lemma})")
            return move, out, "agenda"
        move = self._fallback_move(board)
        say(f"no decision: fallback {self.fallback} chose {board.san(move)}" if decision is None else f"fallback {self.fallback} chose {board.san(move)}")
        return move, out, f"fallback:{self.fallback}"
