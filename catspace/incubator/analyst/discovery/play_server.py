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
KINDS = ("claim", "kill", "gap", "stop", "fuse", "chase", "note", "question", "route", "their", "decision", "fallback", "context", "state", "background", "estimated", "other")


def classify(text: str) -> str:
    t = text.strip()
    if t.startswith("background:"):                      # 52.19 (3): thinking on the opponent's clock
        return "background"
    if t.startswith("decision:") or t.startswith("DECISION ESTIMATED"):
        return "decision"
    if "ESTIMATED" in t:                                 # plan 61 B1: an estimated (not proved) claim or leaf is labelled in the trace
        return "estimated"
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
    if t.startswith("budget:") or "out of time" in t:
        return "fuse"
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
    return dict(n=n, ply=ply, fen=root.fen(), text=text, kind=classify(text), indent=len(text) - len(text.lstrip(" ")),
                moves=parse_moves(text, root), squares=sorted(set(SQUARE_RE.findall(text))))


class TraceBuffer:
    """The per-game store of trace records (plan 54 (6): EVERY record of every ply is kept, each with its ply and the FEN it was parsed against -- the store is the whole
    game); n counts from 1 and never repeats; since(n) returns those with n > since, at_ply(k) those of one ply."""

    def __init__(self, cap: "int | None" = None):
        self.cap, self.n, self.items, self.lock = cap, 0, [], threading.Lock()

    def add(self, text: str, root: chess.Board, ply: int = 0) -> dict:
        with self.lock:
            self.n += 1
            rec = make_record(self.n, text, root, ply)
            self.items.append(rec)
            if self.cap is not None and len(self.items) > self.cap:
                del self.items[: len(self.items) - self.cap]
            return rec

    def since(self, n: int) -> list[dict]:
        with self.lock:
            return [r for r in self.items if r["n"] > n]

    def at_ply(self, k: int) -> list[dict]:
        with self.lock:
            return [r for r in self.items if r["ply"] == k]


class AgendaPlayer:
    def __init__(self, fallback: str = "one_ply", seed: int = 0, stockfish_movetime: float = 0.1):
        if fallback not in FALLBACKS:
            raise ValueError(f"fallback must be one of {FALLBACKS}")
        self.fallback = fallback
        self.rng = random.Random(seed)
        self.sf_movetime = stockfish_movetime
        self._sf = None
        self.carry = None                                # 52.19 (4): the chase context of the last move (fact store, memos) -- the facts of move 30 serve move 31
        self.ponderer = None
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

    def ponder(self, board: chess.Board, emit=None, seconds: "float | None" = None) -> None:
        """52.19 (3): the BACKGROUND tier. `board`: the position after our move, the human to move. Think on his clock: resume what the live move left behind (the out-of-time GAPs first, by chain stake),
        then ask our agenda on the position after each of his legal replies in the order of what it disturbs; lines go to the trace as "background: ...". Stops when `cancel_ponder` is called."""
        from catspace.incubator.analyst.discovery.ponder import Ponderer
        if self.carry is None:
            return
        with AGENDA_LOCK:
            pd = self.ponderer = Ponderer(self.carry)
            if getattr(self, "_cancelled", False):
                pd.cancel()
            try:
                pd.run(board, seconds, emit)
            finally:
                self.carry.deadline = None                     # the next real move sets its own clock

    def cancel_ponder(self) -> None:
        self._cancelled = True
        if self.ponderer is not None:
            self.ponderer.cancel()

    def close(self):
        if self._sf is not None:
            try:
                self._sf.quit()
            except Exception:
                pass
            self._sf = None

    def move(self, board: chess.Board, emit=None, seconds: "float | None" = None):
        """-> (move, trace_lines, source). source is 'agenda' or 'fallback:<name>'. `emit(text)` is called for every trace line AS IT IS
        APPENDED (item 2), so a UI can stream while the agenda thinks; trace_lines is the full list in order. `seconds` (52.17 (5)): the time budget of
        the agenda (None: the question-count fuse as before); the fallback runs after the agenda returns, so a move can take a little longer."""
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
            if self.carry is not None:
                stale = self.carry.facts.advance(board)        # 52.19 (2): the listener -- what the last two plies changed flags the facts that read it (nothing is recomputed here)
                if stale:
                    say(f"facts: {len(stale)} flagged stale by the last two plies (recomputed lazily, when a live question needs them)")
            ag = Agenda(state_of(fen, us), seconds=seconds, carry=self.carry)  # chase_log stays unset in a server
            ag.lines = TraceLines(say)                        # the agenda only ever appends (verified: append / += in agenda.py); no edit to agenda.py
            ag.ctx.history = game_history(board)              # plan 64.5: the positions of the game so far with us to move: CYCLE (61 C4) closes a chase position that repeats one of them
            ag.ask()
            decision = ag.decision()
            if self.carry is None:
                ag.ctx.facts.advance(board)
            self.carry = ag.ctx
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
            lean = "" if decision.proved else f" [ESTIMATED, not proved: confidence {decision.confidence:.2f} by {decision.estimator}]"
            say(f"decision: agenda chose {board.san(move)} ({decision.kind} {decision.what}, value {decision.value}, lemma {decision.lemma}){lean}")
            return move, out, "agenda"
        move = self._fallback_move(board)
        say(f"no decision: fallback {self.fallback} chose {board.san(move)}" if decision is None else f"fallback {self.fallback} chose {board.san(move)}")
        return move, out, f"fallback:{self.fallback}"


def game_history(board: chess.Board) -> set:
    """plan 64.5: the position keys (`agenda._pos_key`) of the earlier positions of this game in which it was OUR move (the side to move on `board`), the root itself excluded -- what the agenda's
    CYCLE terminal reads as the game's repetitions (a chase position that repeats one is a repetition). Empty when the board carries no move stack."""
    from catspace.incubator.analyst.discovery.agenda import _pos_key
    b = board.root()
    keys = set()
    for mv in board.move_stack:
        if b.turn == board.turn:
            keys.add(_pos_key(b.fen()))
        b.push(mv)
    return keys


# ---------------------------------------------------------------- the game (item 3)
PGN_KEY_KINDS = ("context", "claim", "decision", "fallback", "stop")


def pgn_path() -> str:
    from catspace.io import paths
    return paths.experiment("play_games.pgn")


class Game:
    """One game against the engine. The server owns the board (full move stack); the engine's reply runs on a worker thread so the trace streams."""

    def __init__(self, human_white: bool, player: AgendaPlayer, gid: str, path: str | None = None, seconds: "float | None" = None, background: bool = True):
        self.id, self.player, self.human_white = gid, player, human_white
        self.background = background                    # 52.19 (3): think on the human's clock (the only budget: his move, or background_seconds)
        self.background_seconds = None
        self.seconds = seconds                          # 52.17 (5): the agenda's time budget per move (None: the fuse); changeable between moves
        self.board = chess.Board()
        self.trace = TraceBuffer()
        self.lock = threading.RLock()
        self.thinking = False
        self.engine_move: dict | None = None            # {uci, san, source, ply, secs, trace_from, trace_to} of the last engine move
        self.history: list[dict] = []                   # one row per played ply
        self.secs: list[float] = []
        self.error: str | None = None
        self.path = path
        self.started = time.strftime("%Y.%m.%d")
        self.worker: threading.Thread | None = None
        self.bg: threading.Thread | None = None          # 52.19 (3): the background thinker (the human's clock)
        self.live = True
        if not self.human_turn():
            self._start_engine()

    # ---- state
    def human_turn(self) -> bool:
        return self.board.turn == self.human_white

    def over(self):
        o = self.board.outcome(claim_draw=True)
        return o

    def state(self) -> dict:
        with self.lock:
            b = self.board
            o = self.over()
            dests: dict[str, list[str]] = {}
            if o is None and self.human_turn() and not self.thinking:
                for mv in b.legal_moves:
                    dests.setdefault(chess.square_name(mv.from_square), []).append(chess.square_name(mv.to_square))
            last = b.peek() if b.move_stack else None
            return dict(game=self.id, fen=b.fen(), turn="white" if b.turn else "black", human="white" if self.human_white else "black",
                        dests=dests, last_move=[chess.square_name(last.from_square), chess.square_name(last.to_square)] if last else None,
                        ply=len(b.move_stack), thinking=self.thinking, fallback=self.player.fallback,
                        over=o is not None, result=o.result() if o else None, reason=o.termination.name.lower().replace("_", " ") if o else None,
                        seconds=self.seconds, in_check=b.is_check(), notes=context_notes(b) if o is None else [], engine_move=self.engine_move,
                        san=[h["san"] for h in self.history],
                        moves=[dict(ply=h["ply"], san=h["san"], uci=h["uci"], who=h["who"], fen=h["fen"], source=h.get("source")) for h in self.history], error=self.error, trace_n=self.trace.n)

    # ---- moves
    def human_move(self, uci: str) -> dict:
        with self.lock:
            if self.over() is not None:
                raise ValueError("the game is over")
            if not self.human_turn() or self.thinking:
                raise ValueError("not your turn")
            try:
                mv = chess.Move.from_uci(uci)
            except ValueError:
                raise ValueError(f"bad move {uci!r}")
            if mv not in self.board.legal_moves:
                promo = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)
                if promo in self.board.legal_moves:
                    mv = promo
                else:
                    raise ValueError(f"illegal move {uci}")
            san = self.board.san(mv)
            self.board.push(mv)
            self.history.append(dict(ply=len(self.board.move_stack), san=san, uci=mv.uci(), who="human", comment="", fen=self.board.fen()))
            self.engine_move = None                      # the engine's arrow stays until the human moves
            self._stop_background()
            self._save()
            if self.over() is None:
                self._start_engine()
            return self.state()

    def _stop_background(self):
        """The human moved: the background thinker stops (its agendas test the deadline) and lets the lock go before the engine's move asks for it."""
        self.player.cancel_ponder()
        if self.bg is not None and self.bg is not threading.current_thread():
            self.bg.join(5)
        self.bg = None

    def _background(self, root: chess.Board, ply: int):
        try:
            self.player.ponder(root, emit=lambda t: self.trace.add(t, root, ply) if self.live else None, seconds=self.background_seconds)
        except Exception as e:
            import traceback
            self.error = f"background: {e}\n{traceback.format_exc()[-800:]}"

    def _start_engine(self):
        self.thinking = True
        self.worker = threading.Thread(target=self._engine, daemon=True)
        self.worker.start()

    def _engine(self):
        t0 = time.time()
        try:
            with self.lock:
                root = self.board.copy()
                ply = len(root.move_stack) + 1
            first = self.trace.n + 1
            mv, lines, source = self.player.move(root, emit=lambda t: self.trace.add(t, root, ply) if self.live else None, seconds=self.seconds)
            secs = time.time() - t0
            with self.lock:
                if not self.live:
                    return
                san = self.board.san(mv)
                self.board.push(mv)
                recs = [r for r in self.trace.since(first - 1)]
                comment = " | ".join(r["text"].strip() for r in recs if r["kind"] in PGN_KEY_KINDS)[:1500].replace("{", "(").replace("}", ")")
                self.engine_move = dict(uci=mv.uci(), san=san, source=source, ply=ply, secs=round(secs, 2), trace_from=first, trace_to=self.trace.n,
                                        **{"from": chess.square_name(mv.from_square), "to": chess.square_name(mv.to_square)})
                self.history.append(dict(ply=ply, san=san, uci=mv.uci(), who="engine", source=source, secs=round(secs, 2), comment=f"[{source}] {comment}", fen=self.board.fen()))
                self.secs.append(secs)
                self._save()
                if self.over() is None and self.background and self.live:       # 52.19 (3): the human is to move: think on his clock
                    self.player._cancelled = False
                    self.bg = threading.Thread(target=self._background, args=(self.board.copy(), len(self.board.move_stack) + 1), daemon=True)
                    self.bg.start()
        except Exception as e:                           # a failed engine move must be visible, not a hung page
            import traceback
            self.error = f"{e}\n{traceback.format_exc()[-1200:]}"
        finally:
            self.thinking = False

    # ---- the PGN (rewritten per move, keyed by the GameId header: one entry per game)
    def pgn(self) -> str:
        import chess.pgn
        g = chess.pgn.Game()
        g.headers.update({"Event": "catspace play server", "Date": self.started, "GameId": self.id,
                          "White": "human" if self.human_white else "catspace agenda", "Black": "catspace agenda" if self.human_white else "human",
                          "Fallback": self.player.fallback, "Result": (self.over().result() if self.over() else "*")})
        node = g
        b = chess.Board()
        for h in self.history:
            node = node.add_variation(b.parse_san(h["san"]))
            b.push(node.move)
            if h.get("comment"):
                node.comment = h["comment"]
        return str(g)

    def _save(self):
        import chess.pgn
        import io
        path = self.path or pgn_path()
        try:
            existing = []
            try:
                with open(path) as f:
                    while True:
                        g = chess.pgn.read_game(f)
                        if g is None:
                            break
                        existing.append(g)
            except FileNotFoundError:
                pass
            mine = chess.pgn.read_game(io.StringIO(self.pgn()))
            existing = [g for g in existing if g.headers.get("GameId") != self.id] + [mine]
            with open(path, "w") as f:
                for g in existing:
                    f.write(str(g) + "\n\n")
        except Exception as e:                           # the record must never stop the game
            self.error = f"pgn: {e}"


class PlayManager:
    """The single current game of this local server."""

    def __init__(self, fallback: str = "one_ply", seed: int = 0, pgn: str | None = None, seconds: "float | None" = None, background: bool = True):
        self.fallback, self.seed, self.pgn = fallback, seed, pgn
        self.background = background
        self.seconds = seconds
        self.game: Game | None = None
        self.count = 0
        self.lock = threading.Lock()

    def settings(self, seconds=None, set_seconds: bool = False) -> "Game | None":
        """52.17 (5): change the time budget (seconds > 0, or None for the fuse) for the next engine move of the current game and for new games."""
        sec = self.check_seconds(seconds)
        with self.lock:
            self.seconds = sec
            if self.game is not None:
                self.game.seconds = sec
            return self.game

    @staticmethod
    def check_seconds(seconds) -> "float | None":
        if seconds in (None, "", 0):
            return None
        try:
            v = float(seconds)
        except (TypeError, ValueError):
            raise ValueError(f"seconds must be a number, got {seconds!r}")
        if not v > 0:
            raise ValueError("seconds must be positive")
        return v

    def new(self, colour: str = "white", fallback: str | None = None, seconds=None) -> Game:
        sec = self.check_seconds(seconds) if seconds not in (None, "") else self.seconds
        with self.lock:
            if self.game is not None:
                self.game.live = False
                self.game.player.close()
            self.count += 1
            player = AgendaPlayer(fallback or self.fallback, seed=self.seed + self.count)
            gid = time.strftime("%Y%m%d-%H%M%S") + f"-{self.count}"
            self.seconds = sec
            self.game = Game(colour != "black", player, gid, self.pgn, sec, self.background)
            return self.game
