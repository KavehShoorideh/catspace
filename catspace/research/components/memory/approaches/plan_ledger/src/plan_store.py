"""catspace/memory/plan_store.py -- the engine's live memory (M4 work item 1, Kaveh 2026-07-29).

One in-process sqlite file holding LIVE ENGINE STATE AND INTENT -- never training data (datasets
stay npz+DVC). journal_mode=DELETE per the WAL disk-fill scar (tb_cache_delete_mode). No daemon:
per-PLY write rates, sqlite is comfortably off the hot path.

Tables
  plans          append-only intent ledger (the M4 steering verdict reads intent vs realization;
                 later the RL plan-selector's training data)
  plan_outcomes  post-hoc realization per plan row (filled by the game evaluator)
  opponents      per-opponent (Elo, z) posterior persistence (M2c warm-start; M6 dividend)
  armed_tactics  RESERVED for M7 (schema only): blocking_atom = protective SAE atom id from the
                 M3b catalog -- "the Nf6 guards h7" becomes "alarm when atom_X drops".
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

_DDL = """
CREATE TABLE IF NOT EXISTS plans (
  id INTEGER PRIMARY KEY, game_key TEXT NOT NULL, ply INTEGER NOT NULL, side TEXT,
  active_cell INTEGER, portfolio TEXT, scores TEXT, switch_reason TEXT, ts REAL);
CREATE INDEX IF NOT EXISTS ix_plans_game ON plans(game_key, ply);
CREATE TABLE IF NOT EXISTS plan_outcomes (
  plan_id INTEGER PRIMARY KEY REFERENCES plans(id),
  reached INTEGER, plies_to INTEGER, crossed INTEGER, committor_delta REAL, ts REAL);
CREATE TABLE IF NOT EXISTS opponents (
  key TEXT PRIMARY KEY, z BLOB, d_z INTEGER, elo_mean REAL, elo_std REAL,
  n_obs INTEGER, games_seen INTEGER, updated REAL);
-- M7 (reserved; no watcher logic until then)
CREATE TABLE IF NOT EXISTS armed_tactics (
  id INTEGER PRIMARY KEY, opponent_key TEXT, cell INTEGER, line TEXT, payoff REAL,
  blocking_atom INTEGER, status TEXT DEFAULT 'armed', armed_at REAL, fired_at REAL);
"""


class PlanStore:
    def __init__(self, path="data/derived/engine_memory.sqlite"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=DELETE")        # WAL banned (filled the disk twice)
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript(_DDL)
        self.db.commit()

    # ---- plan ledger ----
    def log_plan(self, game_key: str, ply: int, side: str, active_cell: int,
                 portfolio: dict, scores: dict, switch_reason: str = "") -> int:
        cur = self.db.execute(
            "INSERT INTO plans(game_key,ply,side,active_cell,portfolio,scores,switch_reason,ts)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (game_key, ply, side, int(active_cell), json.dumps(portfolio), json.dumps(scores),
             switch_reason, time.time()))
        self.db.commit()
        return int(cur.lastrowid)

    def last_active(self, game_key: str):
        """(active_cell, ply) of the most recent plan in this game -- the hysteresis input."""
        r = self.db.execute("SELECT active_cell, ply FROM plans WHERE game_key=? "
                            "ORDER BY ply DESC, id DESC LIMIT 1", (game_key,)).fetchone()
        return (r[0], r[1]) if r else (None, None)

    def log_outcome(self, plan_id: int, reached: bool, plies_to: int | None,
                    crossed: bool, committor_delta: float):
        self.db.execute(
            "INSERT OR REPLACE INTO plan_outcomes VALUES(?,?,?,?,?,?)",
            (plan_id, int(reached), plies_to, int(crossed), float(committor_delta), time.time()))
        self.db.commit()

    def pending(self, game_key: str):
        """plan rows in this game without outcomes (for the post-game evaluator)."""
        return self.db.execute(
            "SELECT p.id, p.ply, p.active_cell FROM plans p LEFT JOIN plan_outcomes o "
            "ON o.plan_id=p.id WHERE p.game_key=? AND o.plan_id IS NULL ORDER BY p.ply",
            (game_key,)).fetchall()

    def intent_vs_realization(self, game_keys=None):
        """joined ledger rows for the steering verdict (computation lives in the eval script)."""
        q = ("SELECT p.game_key,p.ply,p.active_cell,p.scores,o.reached,o.plies_to,o.crossed,"
             "o.committor_delta FROM plans p JOIN plan_outcomes o ON o.plan_id=p.id")
        if game_keys:
            marks = ",".join("?" * len(game_keys))
            return self.db.execute(q + f" WHERE p.game_key IN ({marks})", list(game_keys)).fetchall()
        return self.db.execute(q).fetchall()

    # ---- opponent persistence (M2c warm-start / M6) ----
    def save_opponent(self, key: str, z: np.ndarray, elo_mean: float, elo_std: float,
                      n_obs: int, games_seen: int):
        z = np.asarray(z, np.float32)
        self.db.execute("INSERT OR REPLACE INTO opponents VALUES(?,?,?,?,?,?,?,?)",
                        (key, z.tobytes(), len(z), float(elo_mean), float(elo_std),
                         int(n_obs), int(games_seen), time.time()))
        self.db.commit()

    def load_opponent(self, key: str):
        r = self.db.execute("SELECT z,d_z,elo_mean,elo_std,n_obs,games_seen FROM opponents "
                            "WHERE key=?", (key,)).fetchone()
        if r is None:
            return None
        return {"z": np.frombuffer(r[0], np.float32).copy(), "elo_mean": r[2], "elo_std": r[3],
                "n_obs": r[4], "games_seen": r[5]}

    def close(self):
        self.db.close()


# ---------------------------------------------------------------------------
def _tests():
    import tempfile, os
    ok = True
    p = os.path.join(tempfile.mkdtemp(), "mem.sqlite")
    st = PlanStore(p)
    assert st.db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete", "WAL banned"
    # plan ledger + hysteresis readback
    a = st.log_plan("g1", 10, "w", 42, {"cells": [42, 7]}, {"42": 0.02}, "init")
    b = st.log_plan("g1", 12, "w", 42, {"cells": [42, 7]}, {"42": 0.03}, "hold")
    assert st.last_active("g1") == (42, 12)
    assert st.last_active("g2") == (None, None), "unknown game -> cold"
    # outcomes: pending -> filled -> joined
    assert [r[0] for r in st.pending("g1")] == [a, b]
    st.log_outcome(a, True, 9, True, -0.31)
    assert [r[0] for r in st.pending("g1")] == [b], "outcome clears pending"
    st.log_outcome(b, False, None, False, 0.0)
    rows = st.intent_vs_realization(["g1"])
    assert len(rows) == 2 and rows[0][4] == 1 and rows[1][4] == 0
    # opponent roundtrip + persistence across reopen
    z = np.arange(16, dtype=np.float32) / 7
    st.save_opponent("maia_or_human_123", z, 1487.0, 55.0, 34, 3)
    st.close()
    st2 = PlanStore(p)
    o = st2.load_opponent("maia_or_human_123")
    assert o and np.allclose(o["z"], z) and o["n_obs"] == 34 and o["games_seen"] == 3
    assert st2.load_opponent("stranger") is None, "cold opponent -> None (population prior)"
    assert st2.last_active("g1") == (42, 12), "ledger survives reopen"
    # reserved M7 table exists, empty
    assert st2.db.execute("SELECT COUNT(*) FROM armed_tactics").fetchone()[0] == 0
    st2.close()
    print("ALL PLAN-STORE TESTS PASSED" if ok else "FAILED")


if __name__ == "__main__":
    _tests()
