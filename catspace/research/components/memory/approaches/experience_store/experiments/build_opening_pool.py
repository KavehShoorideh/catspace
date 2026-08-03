#!/usr/bin/env python
"""catspace/research/components/memory/approaches/experience_store/experiments/build_opening_pool.py -- extract a pool of DISTINCT human-opening starting
positions (Kaveh 2026-07-31: "take only the first few opening plies from the human database
as starting positions, then run full-strength SF-vs-SF continuations on top of those").

Scans real lichess games (data/records/lichess_2019-01), dedupes by move-prefix at --ply plies,
keeps the FEN + prefix + how many real games shared that prefix (weight, for optional sampling),
writes an FEN-per-line pool capped at --limit (most-played prefixes first -- real opening theory,
not one-off idiosyncratic lines).
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import chess
import pandas as pd
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=paths.records("lichess_2019-01"))
    ap.add_argument("--ply", type=int, default=8)
    ap.add_argument("--limit", type=int, default=100_000)
    ap.add_argument("--shards", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=paths.derived("opening_pool_ply8.txt"))
    args = ap.parse_args()
    t0 = time.time()

    files = sorted(glob.glob(f"{args.records}/records_*.parquet"))
    if args.shards:
        files = files[:args.shards]
    counts: dict[str, int] = {}
    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=["moves"])
        for mv in df["moves"]:
            toks = mv.split(" ")
            if len(toks) >= args.ply:
                key = " ".join(toks[:args.ply])
                counts[key] = counts.get(key, 0) + 1
        print(f"  shard {i+1}/{len(files)}: {len(counts):,} distinct so far "
              f"[{time.time()-t0:.0f}s]", flush=True)
        if len(counts) >= args.limit * 3:      # plenty of headroom, stop scanning early
            break

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:args.limit]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(args.out, "w") as fh:
        for prefix, cnt in ranked:
            b = chess.Board()
            for u in prefix.split(" "):
                b.push_uci(u)
            fh.write(f"{b.fen()}\t{cnt}\t{prefix}\n")
            n_ok += 1
    print(f"\n=== wrote {args.out}: {n_ok:,} distinct ply-{args.ply} positions "
          f"(top-weighted by real game count) [{time.time()-t0:.0f}s] ===")
    print("DONE build_opening_pool", flush=True)


if __name__ == "__main__":
    main()
