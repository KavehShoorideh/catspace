"""catspace/tb.py -- canonical tablebase utilities (moved from experiments/value_fixed_point.py
and experiments/gen_dtm_data.py, which now re-export from here; ~10 scripts imported these
cross-experiment, which is the smell this refactor removes).

TB              cached Syzygy probes (wdl/dtz separately -- dtz raises where wdl is fine)
tb_best_move    optimal move: best outcome, then fastest/zeroing/no-repeat if winning,
                longest resistance if losing. NOTE minimizes DTZ, not DTM -- it can hang a
                rook (zeroing reads cheap) and still win; for CLEAN attacking lines prefer
                a UCI engine (JOURNAL 2026-07-22).
white_pov_value V* under the 50-move rule: 1.0/0.5/0.0 or None.
rollout         eps-greedy-White vs optimal-Black playout -> White score.
rollout_dtm     plies-to-mate under tb-optimal play (both sides) -- the DTM proxy used for
                labels (Syzygy stores DTZ; we count the optimal line).
"""
from __future__ import annotations

from functools import lru_cache

import chess
import chess.syzygy
import numpy as np

DEFAULT_SYZYGY = "data/syzygy"


DEFAULT_PROBE_CACHE = "data/derived/tb_probe_cache.sqlite"


class TB:
    """Tablebase with two cache layers: in-process LRU + a PERSISTENT sqlite store shared
    across scripts/sessions (the 2026-07-20 I/O lesson: 'cache tablebase probes' -- probes
    recur massively across veto/forceable/rollout scripts; recomputing them each run was
    the biggest hidden on-the-fly cost). WAL mode; cache failures degrade silently to
    direct probing (never kill a run)."""

    def __init__(self, path: str = DEFAULT_SYZYGY, cache_db: str | None = DEFAULT_PROBE_CACHE):
        self.tb = chess.syzygy.open_tablebase(str(path))
        self._db = None
        self._pending = 0
        if cache_db:
            try:
                import sqlite3
                from pathlib import Path as _P
                _P(cache_db).parent.mkdir(parents=True, exist_ok=True)
                self._db = sqlite3.connect(cache_db, timeout=5.0)
                self._db.execute("PRAGMA journal_mode=DELETE")  # DELETE not WAL: many recomputable-cache readers blocked WAL checkpoints -> 14G unbounded growth (2026-07-24)
                # 2026-07-23 disk-full postmortem: with many long-lived fleet readers the
                # WAL never checkpointed and grew to 25GB. Cap it: after any checkpoint
                # sqlite truncates the WAL back to <=256MB.
                self._db.execute("PRAGMA journal_size_limit=268435456")
                self._db.execute("PRAGMA wal_autocheckpoint=10000")
                self._db.execute("CREATE TABLE IF NOT EXISTS probe "
                                 "(fen TEXT PRIMARY KEY, w INTEGER, d INTEGER)")
            except Exception:
                self._db = None

    @lru_cache(maxsize=1_000_000)
    def _probe(self, fen):
        if self._db is not None:
            try:
                row = self._db.execute("SELECT w, d FROM probe WHERE fen=?", (fen,)).fetchone()
                if row is not None:
                    return row[0], row[1]
            except Exception:
                pass
        b = chess.Board(fen)
        try:
            w = self.tb.probe_wdl(b)
        except (KeyError, chess.syzygy.MissingTableError, ValueError, IndexError):
            w = None
        try:
            d = self.tb.probe_dtz(b)
        except (KeyError, chess.syzygy.MissingTableError, ValueError, IndexError):
            d = None
        if self._db is not None:
            try:
                self._db.execute("INSERT OR IGNORE INTO probe VALUES (?,?,?)", (fen, w, d))
                self._pending += 1
                if self._pending >= 500:
                    self._db.commit(); self._pending = 0
            except Exception:
                pass
        return w, d

    def wdl_dtz(self, board):
        return self._probe(board.fen())

    def close(self):
        if self._db is not None:
            try:
                self._db.commit(); self._db.close()
            except Exception:
                pass
        self.tb.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def white_pov_value(board, tb) -> float | None:
    """1.0 win / 0.5 draw / 0.0 loss for White under the 50-move rule, or None."""
    w, _ = tb.wdl_dtz(board)
    if w is None:
        return None
    if board.turn == chess.BLACK:
        w = -w
    return 1.0 if w == 2 else (0.0 if w == -2 else 0.5)


def tb_best_move(board, tb, seen=None):
    """Tablebase-optimal move: keep the best-outcome moves, then -- if winning --
    convert fastest, preferring a zeroing move and avoiding an already-seen
    position; if losing, resist longest. (DTZ-greedy: see module docstring.)"""
    cands = []
    for m in board.legal_moves:
        c = board.copy(stack=False); c.push(m)
        if c.is_checkmate():
            return m
        w, d = tb.wdl_dtz(c)
        if w is None:
            continue
        cands.append((m, c, -w, d))                      # mover_w = -w (child is opp-to-move)
    if not cands:
        return next(iter(board.legal_moves), None)
    best_w = max(x[2] for x in cands)
    best = [x for x in cands if x[2] == best_w]
    if best_w > 0:                                        # winning: fastest, zeroing, no repeat
        def key(x):
            m, c, mw, d = x
            zeroing = 0 if (board.is_capture(m) or
                            board.piece_type_at(m.from_square) == chess.PAWN) else 1
            repeat = 1 if (seen is not None and c.board_fen() in seen) else 0
            return (repeat, abs(d) if d is not None else 999, zeroing)
        return min(best, key=key)[0]
    if best_w < 0:                                        # losing: resist longest
        return max(best, key=lambda x: (abs(x[3]) if x[3] is not None else 0))[0]
    return best[0][0]                                     # drawing: hold it


def rollout(start, eps_white, tb, rng, max_plies=200):
    """White = eps-greedy over optimal, Black = optimal. Play to absorption;
    return White's score in {1.0 win, 0.5 draw, 0.0 loss}."""
    b = start.copy(stack=False)
    seen = set()
    for _ in range(max_plies):
        if b.is_game_over(claim_draw=True):
            break
        if b.turn == chess.WHITE and rng.random() < eps_white:
            moves = list(b.legal_moves)
            m = moves[int(rng.integers(len(moves)))]      # blunder: uniform random
        else:
            m = tb_best_move(b, tb, seen)
        if m is None:
            break
        seen.add(b.board_fen())
        b.push(m)
    out = b.outcome(claim_draw=True)
    if out is None or out.winner is None:
        return 0.5
    return 1.0 if out.winner == chess.WHITE else 0.0


def rollout_dtm(board, tb, cap=200):
    """Plies to mate under tablebase-optimal play (both sides), or None if it
    doesn't reach mate within cap (drawn / coverage gap)."""
    b = board.copy(stack=False)
    seen = set()
    plies = 0
    for _ in range(cap):
        if b.is_checkmate():
            return plies
        if b.is_game_over(claim_draw=True):
            return None
        m = tb_best_move(b, tb, seen)
        if m is None:
            return None
        if b.turn == chess.BLACK:
            seen.add(b.board_fen())
        b.push(m)
        plies += 1
    return None


def rollout_line(board, tb, cap=200):
    """Tablebase-optimal (adversarial, both-sides-optimal) line from a WON position
    to mate. Returns a list of chess.Board positions [p0, p1, ..., mate] along the
    optimal play, or None if it doesn't reach mate within cap. Since play is optimal,
    the true distance-to-mate of p_i is (len-1 - i) plies and, for i<j on the line,
    the reach-distance p_i -> p_j is (j - i) plies -- exact strong-opponent pairwise
    labels for the multi-goal quasimetric (Kaveh 2026-07-26)."""
    b = board.copy(stack=False)
    seen = set()
    line = [b.copy(stack=False)]
    for _ in range(cap):
        if b.is_checkmate():
            return line
        if b.is_game_over(claim_draw=True):
            return None
        m = tb_best_move(b, tb, seen)
        if m is None:
            return None
        if b.turn == chess.BLACK:
            seen.add(b.board_fen())
        b.push(m)
        line.append(b.copy(stack=False))
    return None
