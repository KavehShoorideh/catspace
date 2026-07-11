#!/usr/bin/env python
"""
experiments/label_stockfish.py — label a seeded SAMPLE of shard positions
with a local Stockfish: white-POV cp (mate mapped to +/-(3200-plies), the
shard eval_cp convention) and Stockfish's own WDL (permille, white-POV).

Nodes-limited (default 100k) rather than depth/time-limited: deterministic-ish
across hardware and fast (~20-60ms/position). Output rows reference positions
as (shard_file, row) so any downstream consumer can re-derive boards/meta.

Resumable: rerunning with the same --out extends the file, skipping
already-labeled (shard_file, row) pairs.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

from catspace.data.encode import board_from_packed
from catspace.data.shards import sample_shard_rows
from catspace.io.paths import derived_dir, newest_shard_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--nodes", type=int, default=100_000)
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None, help="default: data/derived/sf_labels.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-only", action="store_true",
                    help="label only game_id%%50==0 games (the eval-audit population)")
    args = ap.parse_args()

    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()
    out_path = Path(args.out) if args.out else derived_dir() / "sf_labels.npz"

    done: set[tuple[str, int]] = set()
    old: dict[str, np.ndarray] = {}
    if out_path.exists():
        prev = np.load(out_path, allow_pickle=False)
        old = {k: prev[k] for k in prev.files}
        done = set(zip([s for s in old["shard_file"].tolist()], old["row"].tolist()))
        print(f"resuming: {len(done)} rows already labeled")

    todo = [(s, r) for s, r in sample_shard_rows(shard_dir, args.n, args.seed, args.holdout_only)
            if (s, r) not in done]
    print(f"labeling {len(todo)} positions from {shard_dir.name} at nodes={args.nodes}")

    # group by shard so each npz is opened once
    by_shard: dict[str, list[int]] = {}
    for s, r in todo:
        by_shard.setdefault(s, []).append(r)

    new = dict(shard_file=[], row=[], cp=[], wdl_w=[], wdl_d=[], wdl_l=[])
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": args.threads, "UCI_ShowWDL": True})
    t0, labeled = time.time(), 0
    try:
        for shard_name, rows in sorted(by_shard.items()):
            npz = np.load(shard_dir / shard_name)
            packed, meta = npz["packed"], npz["meta"]      # bind once
            for row in rows:
                board = board_from_packed(packed[row], meta[row])
                if board.is_game_over():
                    continue
                info = engine.analyse(board, chess.engine.Limit(nodes=args.nodes))
                score = info["score"].white()
                wdl = info.get("wdl")
                w, d, l = (wdl.white() if wdl is not None else (np.nan, np.nan, np.nan))
                new["shard_file"].append(shard_name); new["row"].append(row)
                new["cp"].append(float(score.score(mate_score=3200)))
                new["wdl_w"].append(w); new["wdl_d"].append(d); new["wdl_l"].append(l)
                labeled += 1
                if labeled % 500 == 0:
                    rate = labeled / (time.time() - t0)
                    print(f"  {labeled}/{len(todo)} ({rate:.0f} pos/s)", flush=True)
    finally:
        engine.quit()

    merged = {}
    for k, dtype in (("shard_file", None), ("row", np.int64), ("cp", np.float32),
                     ("wdl_w", np.float32), ("wdl_d", np.float32), ("wdl_l", np.float32)):
        arr = np.array(new[k]) if dtype is None else np.array(new[k], dtype=dtype)
        merged[k] = np.concatenate([old[k], arr]) if old else arr
    np.savez(out_path, **merged)
    print(f"saved {out_path} ({len(merged['row'])} total rows)")


if __name__ == "__main__":
    main()
