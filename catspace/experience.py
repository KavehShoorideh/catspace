"""catspace/experience.py -- THE EXPERIENCE STORE (Kaveh 2026-07-25: 'we need proper data
tracking... games we play, positions we search, when it was added, all in some persistence
layer, and retrain every N new items').

SQLite (WAL -- the tb-probe-cache precedent: multi-worker appends, silent-degrade never
kills a run) as the system of record; export to npz shards in the regime-rollouts schema
(regime id SELF_REGIME) so train_lichess_fb ingests own-play with zero changes.

Provenance per row: when it was added, which game, which engine commit, which field ckpt.
Counters drive the retrain-every-N trigger. Banks persist as FENs elsewhere and re-embed
per field version (facts survive engine change; embeddings are per-field)."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import chess
import numpy as np

SELF_REGIME = 11        # regime channel for own-play (daemon used 1-10)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS games(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scenario TEXT, start_epd TEXT, result TEXT, term TEXT, plies INTEGER,
  ucis TEXT, engine_commit TEXT, field_ckpt TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS positions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id INTEGER, ply INTEGER, epd TEXT, kind TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS exports(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  last_game_id INTEGER, n_games INTEGER, out TEXT, ts REAL);
CREATE INDEX IF NOT EXISTS idx_pos_game ON positions(game_id);
"""


class ExperienceStore:
    def __init__(self, path: str | Path = "data/derived/experience.sqlite"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=10.0)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA)
        self.db.commit()

    def record_game(self, scenario: str, start_epd: str, result: str, term: str,
                    ucis: list[str], searched_epds: list[str],
                    engine_commit: str = "", field_ckpt: str = "") -> int:
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO games(scenario,start_epd,result,term,plies,ucis,engine_commit,field_ckpt,ts) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (scenario, start_epd, result, term, len(ucis), json.dumps(ucis),
             engine_commit, field_ckpt, now))
        gid = cur.lastrowid
        self.db.executemany(
            "INSERT INTO positions(game_id,ply,epd,kind,ts) VALUES(?,?,?,?,?)",
            [(gid, i, e, "root", now) for i, e in enumerate(searched_epds)])
        self.db.commit()
        return gid

    def games_since_export(self) -> int:
        last = self.db.execute("SELECT COALESCE(MAX(last_game_id),0) FROM exports").fetchone()[0]
        return self.db.execute("SELECT COUNT(*) FROM games WHERE id>?", (last,)).fetchone()[0]

    def export_shards(self, out_dir: str | Path, min_games: int = 1) -> int:
        """new-games trajectories -> npz shard in the regime-rollouts schema
        (packed/meta/ply/clock/result/white_elo/black_elo/game_id/regime/anchor_idx);
        returns games exported (0 if below min_games)."""
        from catspace.data.encode import encode_meta, encode_packed
        last = self.db.execute("SELECT COALESCE(MAX(last_game_id),0) FROM exports").fetchone()[0]
        rows = self.db.execute(
            "SELECT id,start_epd,ucis,result FROM games WHERE id>? ORDER BY id", (last,)).fetchall()
        if len(rows) < min_games:
            return 0
        cols = {k: [] for k in ("pk", "mt", "ply", "gid", "res")}
        for gid, start_epd, ucis, result in rows:
            b = chess.Board(start_epd)
            res = 1 if result == "mate" else 0
            for t, u in enumerate([None] + json.loads(ucis)):
                if u is not None:
                    b.push(chess.Move.from_uci(u))
                cols["pk"].append(encode_packed(b)); cols["mt"].append(encode_meta(b))
                cols["ply"].append(t); cols["gid"].append(gid); cols["res"].append(res)
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        n_existing = len(list(out.glob("shard_*.npz")))
        sp = out / f"shard_{n_existing:03d}.npz"
        n = len(cols["pk"])
        np.savez_compressed(
            sp, packed=np.stack(cols["pk"]), meta=np.stack(cols["mt"]),
            ply=np.array(cols["ply"], np.int32), clock=np.full(n, 300.0, np.float32),
            result=np.array(cols["res"], np.int8),
            white_elo=np.full(n, 1800, np.int16), black_elo=np.full(n, 1800, np.int16),
            game_id=np.array(cols["gid"], np.uint32),
            regime=np.full(n, SELF_REGIME, np.int8),
            anchor_idx=np.zeros(n, np.int32))
        self.db.execute("INSERT INTO exports(last_game_id,n_games,out,ts) VALUES(?,?,?,?)",
                        (rows[-1][0], len(rows), str(sp), time.time()))
        self.db.commit()
        return len(rows)

    def close(self):
        self.db.commit(); self.db.close()
