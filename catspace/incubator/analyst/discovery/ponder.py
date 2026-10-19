"""52.19 (3) THE BACKGROUND TIER: what was left behind is analysed on the opponent's clock ("after our move was made, we can continue analysing whatever was left behind" -- Kaveh 2026-10-19).

Two tiers of thinking. FOREGROUND: the agenda's heap (stake-ordered) reads a fact on demand; a stale fact the live question needs is recomputed now, and nothing else. BACKGROUND: this module drains
the queue of `FactStore.queue` -- the open GAP facts (routes the fuse or the clock cut: the out-of-time routes of the live move first, they are the freshest) by CHAIN STAKE, then the facts the
listener flagged stale on the game's line (re-asked on the current board when it is our move) -- and, when the human is to move (the play server), PONDERS BY LEMMA: for each legal reply of the
human, in the order of the chain stake of the facts that reply would disturb (the listener's flag set, `touched_stake`), our agenda is asked on the position after it, so the facts are
fresh when the next move is asked. The only budget is the clock: `seconds`, or "the human moved" (`Ponderer.cancel`, which makes every open agenda stop at its next test of the deadline).
"""
from __future__ import annotations

import threading
import time

import chess

from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.agenda import Agenda, FUSE, _pos_key
from catspace.incubator.analyst.discovery.state import State


def _scoped(ctx, end: float, stop=None):
    """Run the agendas of one item under a deadline: the fuse is new, the clock is `end` (now, when `stop()` already says the human moved)."""
    ctx.hit = False
    ctx.deadline = ctx.final = time.monotonic() if stop is not None and stop() else end
    if ctx.t0 is None:
        ctx.t0 = time.monotonic()


def resume_gap(ctx, us: str, fact, end: float, stop=None) -> bool:
    """Resume a cut route exactly where it was cut: the position, the route (its replies already read are memoised in the inner agendas and the store), with a new fuse and the clock `end`.
    -> True when the route now ran to its end (the gap fact is resolved)."""
    r = fact.resume
    saved = R._ids[0]
    try:
        ag = Agenda(State(fen=r["fen"], us=us, pieces=[]), side_depth=0, chase_depth=fact.depth, ctx=ctx)
        ag._cur = R.new_question("win", fact.stake, {}, text=r.get("question", ""))
        ctx.left = ag.budget * FUSE
        ctx.routes.pop((fact.pk, r["uci"] + (":promote" if r["promote"] else "")), None)      # the memo says "cut, and no more budget has come": it has now
        _scoped(ctx, end, stop)
        res = ag._chase_route(chess.Move.from_uci(r["uci"]), promote=bool(r["promote"]))
    finally:
        R._ids[0] = saved
    return res.get("gap") is None


def recompute_stale(ctx, us: str, end: float, stop=None) -> int:
    """The facts the listener flagged on the game's line are re-asked on the current board when it is our move: one agenda over every root recomputes them (a recomputation replaces the stale
    fact of the same question). -> how many stale facts were replaced."""
    fs = ctx.facts
    b = fs.board
    if b is None or b.turn != (us == "w"):
        return 0
    before = {f.id for f in fs.queue() if f.kind != "gap"}
    if not before:
        return 0
    saved = R._ids[0]
    try:
        ctx.left = 1 << 30
        _scoped(ctx, end, stop)
        ag = Agenda(State(fen=b.fen(), us=us, pieces=[]), side_depth=0, chase_depth=1, ctx=ctx, min_stake=1)
        ag.ask()
    finally:
        R._ids[0] = saved
    return len(before - {f.id for f in fs.queue() if f.kind != "gap"})


def drain(ctx, us: str, seconds: float, log=None, stop=None) -> dict:
    """Drain the queue for `seconds` (or until `stop()`): the open gap routes by chain stake, then the stale facts. -> {queue, done, left, secs}."""
    fs = ctx.facts
    t0 = time.monotonic()
    end = t0 + seconds
    queue0 = len(fs.queue())
    done = 0
    for f in [x for x in fs.queue() if x.kind == "gap"]:
        if time.monotonic() >= end or (stop is not None and stop()):
            break
        if f.resolved or f.superseded:
            continue
        ok = resume_gap(ctx, us, f, end, stop)
        done += 1 if ok else 0
        if log:
            log(f"{'resumed' if ok else 'cut again'}: {f.text} (chain stake {f.chain_stake}, {f.under})")
    if time.monotonic() < end and not (stop is not None and stop()):
        n = recompute_stale(ctx, us, end, stop)
        done += n
        if n and log:
            log(f"recomputed {n} stale fact(s) on the current board")
    return dict(queue=queue0, done=done, left=len(fs.queue()), secs=round(time.monotonic() - t0, 1))


class Ponderer:
    """The play server's background worker: after our move, with the human to move. `run` drains the gap queue, then asks our agenda on the position after each legal reply of the human, in the order
    of the chain stake of what the reply disturbs; `cancel` (the human moved) stops it within one test of the deadline."""

    def __init__(self, ctx):
        self.ctx, self.stopped, self.lock = ctx, False, threading.Lock()

    def cancel(self) -> None:
        self.stopped = True
        self.ctx.deadline = time.monotonic()                   # every open agenda stops at its next test (STOP (t))

    def run(self, board: chess.Board, seconds: "float | None", emit=None) -> dict:
        """`board`: the position after our move (the human to move). -> {replies, gaps, secs}."""
        ctx, fs = self.ctx, self.ctx.facts
        us = "b" if board.turn else "w"
        t0 = time.monotonic()
        end = t0 + (seconds if seconds else 3600.0)
        say = emit or (lambda t: None)
        g = drain(ctx, us, max(end - t0, 0), log=lambda t: say("background: " + t), stop=lambda: self.stopped)
        if g["queue"]:
            say(f"background: gap routes resumed {g['done']} of {g['queue']} queued, {g['left']} left ({g['secs']} s)")
        order = []
        for mv in board.legal_moves:
            c = board.copy(stack=False)
            san = board.san(mv)
            c.push(mv)
            order.append((-fs.touched_stake(R.diff_squares(board, c)), not board.is_capture(mv), not board.is_check() and not c.is_check(), mv.uci(), san, c))
        order.sort(key=lambda t: t[:4])
        replies = 0
        for stake, _cap, _chk, uci, san, c in order:
            if self.stopped or time.monotonic() >= end:
                break
            if c.is_game_over() or _pos_key(c.fen()) in ctx.memo:
                continue
            saved = R._ids[0]
            try:
                ctx.left = 1 << 30
                _scoped(ctx, end, lambda: self.stopped)
                h = Agenda(State(fen=c.fen(), us=us, pieces=[]), side_depth=0, chase_depth=0, ctx=ctx)
                a0 = fs.snapshot()
                ans = h._inner(c.fen(), 1)
            finally:
                R._ids[0] = saved
            if self.stopped:
                break
            replies += 1
            a1 = fs.snapshot()
            say(f"background: if {san}: " + (f"our best claim {ans['claim'].what} wins {ans['value']}" if ans and ans.get("claim") is not None else "no claim of ours") + f" ({a1['derived'] - a0['derived']} facts derived, {a1['served'] - a0['served']} served; it disturbs facts of stake {-stake})")
        return dict(replies=replies, gaps=g, secs=round(time.monotonic() - t0, 1))
