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

from catspace.data.sources import PairBatch, PairSource


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


def sample_shard_rows(shard_dir, n: int, seed: int, holdout_only: bool = False,
                       holdout_mod: int = 50) -> list:
    """Seeded position-uniform sample of (shard_file, row) pairs across a
    shard dir; holdout_only restricts to game_id % holdout_mod == 0 games
    (the never-trained population used by eval/audit drivers)."""
    shard_dir = Path(shard_dir)
    per_shard = []
    for path in sorted(shard_dir.glob("shard_*.npz")):
        npz = np.load(path)
        gid = npz["game_id"]
        rows = np.flatnonzero(gid % holdout_mod == 0) if holdout_only else np.arange(len(gid))
        per_shard.append((path.name, rows))
    total = sum(len(r) for _, r in per_shard)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(total, size=min(n, total), replace=False))
    out, offset = [], 0
    for name, rows in per_shard:
        sel = pick[(pick >= offset) & (pick < offset + len(rows))] - offset
        out.extend((name, int(rows[i])) for i in sel)
        offset += len(rows)
    return out


class MixedPairSource:
    """Interleaves batches from two PairSource-like objects (e.g. human
    LichessPairSource + self-play LichessPairSource pointed at a
    selfplay_generate.py output dir) by a fixed mix ratio -- each YIELDED
    BATCH comes entirely from one source or the other (a weighted coin flip
    per batch), rather than mixing within a batch -- simpler, and just as
    effective at typical batch sizes (512).

    2026-07-12 (JOURNAL.md): lets train_lichess_fb.py train on human data
    and self-play data in the same run without changing LichessPairSource
    or the shard format at all -- self-play shards ARE Lichess shards,
    just written by a different generator."""

    def __init__(self, primary: "LichessPairSource", secondary: "LichessPairSource",
                secondary_frac: float):
        assert 0.0 <= secondary_frac <= 1.0
        self.primary = primary
        self.secondary = secondary
        self.secondary_frac = secondary_frac

    def batches(self, batch_size: int, seed: int) -> Iterator[PairBatch]:
        rng = np.random.default_rng(seed)
        it_p = iter(self.primary.batches(batch_size, seed))
        it_s = iter(self.secondary.batches(batch_size, seed + 1))
        epoch = 0
        while True:
            use_secondary = rng.random() < self.secondary_frac
            it = it_s if use_secondary else it_p
            try:
                yield next(it)
            except StopIteration:
                # whichever source ran dry restarts its own epoch (fresh
                # seed) independently -- the two sources are NOT required
                # to be the same size, so they naturally cycle at different
                # rates without desyncing the mix ratio.
                epoch += 1
                if use_secondary:
                    it_s = iter(self.secondary.batches(batch_size, seed + 1 + epoch))
                    yield next(it_s)
                else:
                    it_p = iter(self.primary.batches(batch_size, seed + epoch))
                    yield next(it_p)


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
                    "ply_g": data["ply"][goal_rows[sl]],           # goal ply -- for ply-gap calibration
                    # plies from anchor to its game's END: gates the outcome-AXIS pull
                    # (near-terminal rows of decisive games ~ the forced regions)
                    "plies_to_end": data["ply"][last_row_of_game[sl]] - data["ply"][sl],
                    "clock": data["clock"][sl],
                    "game_id": data["game_id"][sl],
                    "board_meta": data["meta"][sl],                # anchor rows
                    "board_meta_g": data["meta"][goal_rows[sl]],   # goal rows
                }
                if has_eval:
                    meta["eval_cp"] = data["eval_cp"][sl]
                yield PairBatch(anchors=data["packed"][sl], goals=data["packed"][goal_rows[sl]], meta=meta)
