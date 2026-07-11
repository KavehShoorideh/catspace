#!/usr/bin/env python
"""
experiments/build_lichess_shards.py — stream-filter-encode-shard a Lichess
.pgn.zst dump into position shards under data/shards/, bounded by --max-games
and/or --max-gb so a laptop SSD can't be filled by accident.

Get monthly dumps from https://database.lichess.org/ (lichess_db_standard_
rated_YYYY-MM.pgn.zst, ~28-33 GB compressed per 2024-2026 month -- this
script streams it directly, never decompressing to disk).

Try it on the committed fixture first:
    python experiments/build_lichess_shards.py \\
        --pgn tests/fixtures/lichess_mini.pgn.zst --max-games 100 --max-gb 0.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

from latentchess.data.lichess import GameFilter, build_shards
from latentchess.io.paths import shards_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgn", required=True, help="path to a lichess_db_*.pgn.zst dump")
    ap.add_argument("--out", default=None, help="output shard dir (default: data/shards/<pgn stem>)")
    ap.add_argument("--min-elo", type=int, default=1000)
    ap.add_argument("--max-elo", type=int, default=4000)
    ap.add_argument("--min-base-seconds", type=int, default=180)
    ap.add_argument("--min-plies", type=int, default=20)
    ap.add_argument("--skip-first-plies", type=int, default=10)
    ap.add_argument("--min-clock-s", type=float, default=30.0)
    ap.add_argument("--include-bots", action="store_true")
    ap.add_argument("--shard-positions", type=int, default=1_000_000)
    ap.add_argument("--max-games", type=int, default=50_000)
    ap.add_argument("--max-gb", type=float, default=2.0)
    args = ap.parse_args()

    pgn_path = Path(args.pgn)
    out_dir = Path(args.out) if args.out else shards_dir() / pgn_path.stem.replace(".pgn", "")

    gf = GameFilter(
        min_elo=args.min_elo, max_elo=args.max_elo, min_base_seconds=args.min_base_seconds,
        min_plies=args.min_plies, skip_first_plies=args.skip_first_plies,
        min_clock_s=args.min_clock_s, exclude_bots=not args.include_bots,
    )
    manifest = build_shards(pgn_path, gf, out_dir,
                             shard_positions=args.shard_positions,
                             max_games=args.max_games, max_gb=args.max_gb)

    print(f"scanned {manifest['games_scanned']} header-passing games, "
          f"kept {manifest['games_kept']} contributing {manifest['positions']} positions "
          f"across {len(manifest['shards'])} shard(s) -> {out_dir}")


if __name__ == "__main__":
    main()
