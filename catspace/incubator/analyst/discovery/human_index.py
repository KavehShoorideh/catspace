"""PLAN 78.19 STAGE E: THE HUMAN GAMES AS A TRAINING-ONLY RANK SOURCE.

An index position key -> {move: (n, points)} built from game records, and `HumanSource`, a rank source of the shared shape (`rank(variant, moves) -> [(move, score)]`,
`source`, `training_only`, `close`) that reads it. TRAINING ONLY: sleep's training kit uses it (plan_tools, break_ties, disagreement); nothing of the inference path
(cs_decision, wake's decision, cs_engine solve) imports or reads it. Only lemmas that sleep confirms by its own exact replay enter the store, never the raw frequencies.

Generic: any game with records and a state key. The dependencies are injected: `records_iter` (yields (moves, outcome)), `key_of(variant, moves) -> key` (default the kernel's
`cs_engine.state` key, which records mover and clocks), and the store is stdlib sqlite3. `lichess_records_iter` is the one records supplier here (it is lichess-specific)."""
from __future__ import annotations

import argparse
import json
import sqlite3
import time

import cs_engine as CE

from catspace.io import paths

POINTS = {1: [100, 0], 0: [50, 50], -1: [0, 100]}      # a result from the first mover's view -> [first mover's, second mover's] points


def default_path(game: str) -> str:
    return paths.derived(f"human_index_{game}.sqlite")


def signed(key: int) -> int:
    """sqlite INTEGER is signed 64-bit; the kernel's u64 key is stored as its two's complement."""
    return key - (1 << 64) if key >= (1 << 63) else key


def kernel_key(variant: str, moves: list) -> int:
    return signed(CE.state(variant, None, list(moves))[4])


def points_of(outcome) -> list:
    """An outcome as [first mover's, second mover's] points: that pair itself, or a result 1/0/-1 from the first mover's view."""
    return list(outcome) if isinstance(outcome, (list, tuple)) else POINTS[int(outcome)]


def lichess_records_iter(shards):
    """THE LICHESS SUPPLIER: yields (moves, outcome) for every game of the given shard numbers of data/records/lichess_2019-01 (outcome: the white points pair); the shape
    `build` takes, so any other records source is one more such generator."""
    import pyarrow.parquet as pq
    for n in shards:
        f = paths.records("lichess_2019-01", f"records_{n:05d}.parquet")
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=20_000, columns=["result", "moves"]):
            for r in batch.to_pylist():
                yield r["moves"].split(), POINTS[int(r["result"])]


def _init(con: sqlite3.Connection) -> None:
    con.execute("CREATE TABLE IF NOT EXISTS moves (key INTEGER, move TEXT, n INTEGER, points INTEGER, PRIMARY KEY (key, move)) WITHOUT ROWID")
    con.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")


def _flush(con: sqlite3.Connection, acc: dict) -> None:
    con.executemany("INSERT INTO moves (key, move, n, points) VALUES (?, ?, ?, ?) ON CONFLICT (key, move) DO UPDATE SET n = n + excluded.n, points = points + excluded.points",
                    ((k, m, n, p) for (k, m), (n, p) in acc.items()))
    acc.clear()


def build(game: str, variant: str, records_iter, *, max_ply: int = 40, min_count: int = 1, out_path: "str | None" = None, key_of=kernel_key,
          chunk: int = 20_000, shards=None) -> dict:
    """Index `records_iter` (yields (moves list, outcome)) into the sqlite file `out_path`: per (position key, move) the count n and the points the mover earned in those games
    (the mover's view: an outcome pair is [first mover's, second mover's], the role of a ply is its mover's first-seen order, so any two-player game works). Replays up to
    `max_ply` plies of each game; aggregates `chunk` games in a dict then UPSERTs, so it is incremental (call again with the next shards; the meta adds up) and memory is bounded.
    Rows with n < min_count are dropped at the end (a later build would then count from zero for them, so keep 1 while building in several calls). -> the meta dict."""
    out_path = out_path or default_path(game)
    t0 = time.time()
    con = sqlite3.connect(out_path)
    _init(con)
    acc: dict = {}
    games = 0

    def go():
        _flush(con, acc)
        con.commit()

    for moves, outcome in records_iter:
        pts = points_of(outcome)
        moves = list(moves)[:max_ply]
        games += 1
        first = None
        for i, mv in enumerate(moves):
            try:
                st = CE.state(variant, None, moves[:i])
            except Exception:
                break
            if mv not in st[2]:
                break                                       # an illegal (or foreign) move ends the replay of this game
            if first is None:
                first = st[1]
            role = 0 if st[1] == first else 1
            slot = acc.setdefault((key_of(variant, moves[:i]) if key_of is not kernel_key else signed(st[4]), mv), [0, 0])
            slot[0] += 1
            slot[1] += pts[role]
        if games % chunk == 0:
            go()
    go()
    if min_count > 1:
        con.execute("DELETE FROM moves WHERE n < ?", (min_count,))
    old = dict(con.execute("SELECT k, v FROM meta"))
    positions = con.execute("SELECT COUNT(DISTINCT key) FROM moves").fetchone()[0]
    rows = con.execute("SELECT COUNT(*) FROM moves").fetchone()[0]
    meta = {"game": game, "variant": variant, "shards": json.dumps(sorted(set(json.loads(old.get("shards", "[]"))) | set(shards or []))), "max_ply": str(max_ply),
            "min_count": str(min_count), "games": str(int(old.get("games", 0)) + games), "positions": str(positions), "rows": str(rows)}
    con.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", meta.items())
    con.commit()
    con.close()
    meta["seconds"] = round(time.time() - t0, 1)
    return meta


class HumanSource:
    """A RANK SOURCE over the human index: TRAINING ONLY (sleep's kit; never a decision's). `rank(variant, moves)` -> [(move, n)] best (most played) first, for the moves legal at
    the position after `moves` that the index has with n >= min_count; [] when the position is unknown. `detail` adds the mean points behind each count (the mover's view,
    the outcomes the humans got: evidence, not understanding). Same shape as UciTeacher / EstimateSource."""
    training_only = True
    source = "human"

    def __init__(self, path: "str | None" = None, min_count: int = 1, key_of=kernel_key):
        self.path, self.min_count, self.key_of = path or default_path("chess"), min_count, key_of
        self.con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)

    def detail(self, variant: str, moves: list) -> dict:
        st = CE.state(variant, None, list(moves))
        legal = set(st[2])
        key = signed(st[4]) if self.key_of is kernel_key else self.key_of(variant, moves)
        rows = self.con.execute("SELECT move, n, points FROM moves WHERE key = ? AND n >= ?", (key, self.min_count)).fetchall()
        return {m: {"n": n, "mean_points": p / n} for m, n, p in rows if m in legal}

    def rank(self, variant: str, moves: list, cap: "int | None" = None) -> list:
        d = self.detail(variant, moves)
        out = sorted(((m, v["n"]) for m, v in d.items()), key=lambda x: (-x[1], x[0]))
        return out[:cap] if cap else out

    def __call__(self, variant: str, moves: list) -> "str | None":
        r = self.rank(variant, moves)
        return r[0][0] if r else None

    def close(self) -> None:
        self.con.close()


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description="build the human-games index (plan 78.19 stage E) from the lichess records")
    ap.add_argument("game")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--shards", type=int, default=1, help="the first N shards of data/records/lichess_2019-01")
    ap.add_argument("--first-shard", type=int, default=0)
    ap.add_argument("--max-ply", type=int, default=40)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    shards = list(range(a.first_shard, a.first_shard + a.shards))
    m = build(a.game, a.variant or a.game, lichess_records_iter(shards), max_ply=a.max_ply, min_count=a.min_count, out_path=a.out, shards=shards)
    print(f"OBS human index {a.game}: games {m['games']}, positions {m['positions']}, rows {m['rows']}, max_ply {m['max_ply']}, min_count {m['min_count']}, "
          f"seconds {m['seconds']}, out {a.out or default_path(a.game)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
