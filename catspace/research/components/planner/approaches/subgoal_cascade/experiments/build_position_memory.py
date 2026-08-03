#!/usr/bin/env python
"""
catspace/research/components/planner/approaches/subgoal_cascade/experiments/build_position_memory.py — seed the position MEMORY (the vector
database of seen positions; catspace/memory/store.py) from the Lichess shards.

Samples TRAIN rows (holdout excluded -- the memory may later inform play, and
holdout must stay leak-free), embeds F with the given field on CPU in batches,
labels each row with the game's white-POV result + its CERTIFIED flag
(mate|draw|material-backed win, catspace/data/certified.py), and writes the
store to --out. The play server then appends play_ui/mcts_sim entries online.

Usage:
  .venv/bin/python catspace/research/components/planner/approaches/subgoal_cascade/experiments/build_position_memory.py            # incumbent, n=200k
  .venv/bin/python catspace/research/components/planner/approaches/subgoal_cascade/experiments/build_position_memory.py --ckpt <field>.pt --n 500000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


from catspace.research.tools.chess_specific.chessdata.certified import collect_certified_games
from catspace.research.tools.chess_specific.chessdata.shards import sample_shard_rows
from catspace.io.paths import newest_shard_dir
from catspace.research.components.memory.approaches.vector_store_retrieval.src.store import PositionMemory
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt
from catspace.research.tools.viz.viz.realboard import board_from_row, embed_positions
from catspace.io import paths

HOLDOUT_MOD = 50           # trainer's holdout convention (game_id % 50 == 0)
COLS = ("packed", "meta", "ply", "clock", "result", "white_elo", "black_elo", "game_id")


def load_rows(shard_dir: Path, picks: list) -> dict:
    by_file: dict = {}
    for name, row in picks:
        by_file.setdefault(name, []).append(row)
    out: dict = {k: [] for k in COLS}
    for name, rows in sorted(by_file.items()):
        npz = np.load(shard_dir / name)
        idx = np.array(sorted(rows))
        for k in COLS:
            out[k].append(npz[k][idx])
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=paths.sep("cert_base_full.pt"))
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--out", default=paths.derived("position_memory"))
    ap.add_argument("--shards", default=None)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--resign-material-gap", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cpu"                       # GPU stays with training (project rule)
    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()

    t0 = time.time()
    fb, payload = load_ckpt(Path(args.ckpt), device)
    fb.eval()
    ckpt_tag = f"{Path(args.ckpt).name}@{payload.get('step', '?')}"
    print(f"[stage] load field {ckpt_tag}: {time.time() - t0:.1f}s")

    # TRAIN rows only: oversample, drop holdout games, trim to n
    t0 = time.time()
    picks = sample_shard_rows(shard_dir, int(args.n * 1.1) + 64, seed=args.seed,
                              holdout_only=False)
    rows = load_rows(shard_dir, picks)
    keep = (rows["game_id"] % HOLDOUT_MOD) != 0
    rows = {k: v[keep][:args.n] for k, v in rows.items()}
    n = len(rows["packed"])
    cert = collect_certified_games(shard_dir, args.resign_material_gap)
    cert_row = cert[rows["game_id"]]
    print(f"[stage] sample+load {n} train rows ({100 * cert_row.mean():.1f}% certified): "
          f"{time.time() - t0:.1f}s")

    mem = PositionMemory(dim=fb.d, capacity=max(n + 100_000, 300_000), ckpt_tag=ckpt_tag)
    t0 = time.time()
    for lo in range(0, n, args.batch):
        hi = min(lo + args.batch, n)
        sl = slice(lo, hi)
        F, _ = embed_positions(fb, rows["packed"][sl], rows["meta"][sl],
                               rows["white_elo"][sl], rows["black_elo"][sl],
                               rows["clock"][sl], device)
        fens = [board_from_row(rows["packed"][i], rows["meta"][i]).fen()
                for i in range(lo, hi)]
        mem.add(F, fens, rows["result"][sl].tolist(), cert_row[sl].tolist(),
                source="human", plies=rows["ply"][sl].tolist())
        if (lo // args.batch) % 10 == 0:
            rate = hi / max(time.time() - t0, 1e-9)
            print(f"  {hi}/{n} embedded ({rate:.0f} rows/s, "
                  f"eta {(n - hi) / max(rate, 1):.0f}s)", flush=True)
    print(f"[stage] embed+index {n} rows: {time.time() - t0:.1f}s")

    t0 = time.time()
    mem.save(Path(args.out))
    print(f"[stage] save -> {args.out}: {time.time() - t0:.1f}s")
    print(f"VERDICT MEMORY_BUILT n={len(mem)} dim={mem.dim} ckpt={ckpt_tag} "
          f"certified_frac={float(np.mean(cert_row)):.3f}")


if __name__ == "__main__":
    main()
