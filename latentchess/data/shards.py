"""
data/shards.py — write any PairSource to fixed-size npz shards, and read them
back with bounded memory (one shard + a shuffle buffer) via ShardReader,
itself a PairSource. Also LichessPairSource: geometric-horizon (anchor, goal)
pairs sampled directly from position shards, staying within each game.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from latentchess.data.sources import PairBatch, PairSource


def write_shards(source: PairSource, out_dir, shard_size: int = 1_000_000,
                  batch_size: int = 8192, seed: int = 0) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchors_buf: list[np.ndarray] = []
    goals_buf: list[np.ndarray] = []
    buf_len = 0
    shard_idx = 0
    paths = []

    def flush():
        nonlocal anchors_buf, goals_buf, buf_len, shard_idx
        if buf_len == 0:
            return
        anchors = np.concatenate(anchors_buf)
        goals = np.concatenate(goals_buf)
        path = out_dir / f"shard_{shard_idx:05d}.npz"
        np.savez(path, anchors=anchors, goals=goals)
        paths.append(path)
        shard_idx += 1
        anchors_buf, goals_buf, buf_len = [], [], 0

    for batch in source.batches(batch_size, seed):
        anchors_buf.append(batch.anchors)
        goals_buf.append(batch.goals)
        buf_len += len(batch.anchors)
        if buf_len >= shard_size:
            flush()
    flush()

    manifest = {"n_shards": len(paths), "shard_size": shard_size,
                "total": sum(int(np.load(p)["anchors"].shape[0]) for p in paths)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return paths


class ShardReader:
    """PairSource over on-disk (anchors, goals) shards: shuffles shard order
    per epoch, streams one shard at a time into a fixed-size ring buffer,
    yields batches drawn from the buffer while refilling -- memory bound is
    one shard + the buffer, not the whole dataset."""

    def __init__(self, dir, shuffle_buffer: int = 100_000):
        self.dir = Path(dir)
        self.shuffle_buffer = shuffle_buffer
        self.paths = sorted(self.dir.glob("shard_*.npz"))

    def batches(self, batch_size: int, seed: int) -> Iterator[PairBatch]:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.paths))
        buf_a = np.empty(0, dtype=np.int64)
        buf_g = np.empty(0, dtype=np.int64)

        def refill(buf_a, buf_g, path_idx):
            while len(buf_a) < self.shuffle_buffer and path_idx < len(order):
                data = np.load(self.paths[order[path_idx]])
                buf_a = np.concatenate([buf_a, data["anchors"]])
                buf_g = np.concatenate([buf_g, data["goals"]])
                path_idx += 1
            return buf_a, buf_g, path_idx

        path_idx = 0
        buf_a, buf_g, path_idx = refill(buf_a, buf_g, path_idx)
        while len(buf_a) > 0:
            n = min(batch_size, len(buf_a))
            pick = rng.choice(len(buf_a), size=n, replace=False)
            mask = np.ones(len(buf_a), dtype=bool)
            mask[pick] = False
            yield PairBatch(anchors=buf_a[pick].copy(), goals=buf_g[pick].copy())
            buf_a, buf_g = buf_a[mask], buf_g[mask]
            buf_a, buf_g, path_idx = refill(buf_a, buf_g, path_idx)


class LichessPairSource:
    """Geometric-horizon (anchor, goal) PACKED-POSITION-ROW pairs sampled
    within each game from position shards written by data.lichess.build_shards."""

    def __init__(self, shard_dir, gamma: float):
        self.shard_dir = Path(shard_dir)
        self.gamma = gamma
        self.paths = sorted(self.shard_dir.glob("shard_*.npz"))

    def batches(self, batch_size: int, seed: int) -> Iterator[PairBatch]:
        rng = np.random.default_rng(seed)
        for path in self.paths:
            npz = np.load(path)
            # bind every array ONCE: NpzFile re-reads (and re-allocates) the
            # whole array on every __getitem__, so per-batch npz["packed"][sl]
            # is quadratic io and churns 100MB allocations
            data = {k: npz[k] for k in npz.files}
            game_id = data["game_id"]
            n = len(game_id)
            if n == 0:
                continue
            # offsets[g] = first row index of game g; assumes game_id is
            # non-decreasing within a shard (guaranteed by build_shards).
            change = np.flatnonzero(np.diff(game_id)) + 1
            starts = np.concatenate([[0], change])
            ends = np.concatenate([change, [n]])
            game_of_row = np.repeat(np.arange(len(starts)), ends - starts)
            last_row_of_game = ends[game_of_row] - 1

            rows = np.arange(n)
            k = 1 + rng.geometric(1.0 - self.gamma, size=n)
            goal_rows = np.minimum(rows + k, last_row_of_game)

            has_eval = "eval_cp" in data
            for i in range(0, n, batch_size):
                sl = slice(i, min(i + batch_size, n))
                meta = {
                    "result": data["result"][sl],
                    "white_elo": data["white_elo"][sl],
                    "black_elo": data["black_elo"][sl],
                    "ply": data["ply"][sl],
                    "clock": data["clock"][sl],
                    "game_id": data["game_id"][sl],
                    "board_meta": data["meta"][sl],                # anchor rows
                    "board_meta_g": data["meta"][goal_rows[sl]],   # goal rows
                }
                if has_eval:
                    meta["eval_cp"] = data["eval_cp"][sl]
                yield PairBatch(anchors=data["packed"][sl], goals=data["packed"][goal_rows[sl]], meta=meta)
