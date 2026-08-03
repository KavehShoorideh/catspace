#!/usr/bin/env python
"""experiments/build_game_records.py -- STAGE A of the identity-preserving data pipeline.

Streams a .pgn.zst dump (lichess, or any engine PGN with the same headers) and emits COMPACT,
IDENTITY-PRESERVING game records as parquet shards. One row per game:
  game_id, source, white_id, black_id, white_elo, black_elo, result (+1/0/-1), n_plies,
  time_control, termination, white_title, black_title, moves (space-joined UCI).

Why game records, not position shards: (1) IDENTITY -- keeps usernames / engine names so the
z-encoder can group by player_id (masked at train time); (2) LOSSLESS + TINY -- ~40 UCI moves =
~200 bytes/game vs ~7KB/position for lc0 112-plane; reconstruct positions in ANY encoding on
demand; (3) the natural unit for BALANCING (outcome + strength are per-game) -- Stage B resamples
these records to fix the draw-starvation + strength-skew that data_distribution_check.py caught.
Engine PGNs (CCRL/fastchess) carry identity in the same headers -> ingest into the SAME schema with
--source, so the universal-z manifold (humans + engines) is one dataset.

Reuses catspace.research.tools.chess_specific.chessdata.lichess.stream_filtered_games (the streaming header-prefilter). No decompress
to disk. Run a SHORT prefix first (--pgn ...prefix256mb...) to validate before the full month.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.research.tools.chess_specific.chessdata.lichess import GameFilter, stream_filtered_games

_RESULT_MAP = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}

_SCHEMA = pa.schema([
    ("game_id", pa.int64()), ("source", pa.string()),
    ("white_id", pa.string()), ("black_id", pa.string()),
    ("white_elo", pa.int16()), ("black_elo", pa.int16()),
    ("result", pa.int8()), ("n_plies", pa.int32()),
    ("time_control", pa.string()), ("termination", pa.string()),
    ("white_title", pa.string()), ("black_title", pa.string()),
    ("moves", pa.string()),
])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgn", default="data/lichess/lichess_db_standard_rated_2019-01.prefix256mb.pgn.zst")
    ap.add_argument("--out", default="data/records/lichess_2019-01")
    ap.add_argument("--source", default="lichess")
    ap.add_argument("--max-games", type=int, default=0, help="0 = no cap (stream to EOF)")
    ap.add_argument("--games-per-shard", type=int, default=200_000)
    ap.add_argument("--min-elo", type=int, default=1000)
    ap.add_argument("--max-elo", type=int, default=4000)
    ap.add_argument("--min-plies", type=int, default=20)
    ap.add_argument("--exclude-bots", type=int, default=1, help="1=drop BOT-titled games (default for human z); 0=keep (engine ingestion)")
    ap.add_argument("--tolerate-truncation", type=int, default=1, help="1=treat a truncated final frame as EOF (prefix downloads)")
    args = ap.parse_args()

    gf = GameFilter(min_elo=args.min_elo, max_elo=args.max_elo, min_plies=args.min_plies,
                    exclude_bots=bool(args.exclude_bots))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    max_games = args.max_games or None
    t0 = time.time()
    print(f"[game-records] {args.pgn} -> {out_dir}  source={args.source} filter={gf}", flush=True)

    cols = {k: [] for k in _SCHEMA.names}
    shard_idx = 0; kept = 0; shards = []

    def flush():
        nonlocal shard_idx
        if not cols["game_id"]:
            return
        tbl = pa.table({k: cols[k] for k in _SCHEMA.names}, schema=_SCHEMA)
        path = out_dir / f"records_{shard_idx:05d}.parquet"
        pq.write_table(tbl, path, compression="zstd")
        shards.append({"file": path.name, "n": len(cols["game_id"])})
        shard_idx += 1
        for k in cols:
            cols[k] = []

    for game in stream_filtered_games(args.pgn, gf, max_games=max_games,
                                      tolerate_truncation=bool(args.tolerate_truncation)):
        h = game.headers
        moves = [m.uci() for m in game.mainline_moves()]
        if len(moves) < gf.min_plies:                      # min_plies is a game-length floor
            continue
        cols["game_id"].append(kept)
        cols["source"].append(args.source)
        cols["white_id"].append(h.get("White", "?"))
        cols["black_id"].append(h.get("Black", "?"))
        cols["white_elo"].append(int(h.get("WhiteElo", 0) or 0))
        cols["black_elo"].append(int(h.get("BlackElo", 0) or 0))
        cols["result"].append(_RESULT_MAP.get(h.get("Result", ""), 0))
        cols["n_plies"].append(len(moves))
        cols["time_control"].append(h.get("TimeControl", ""))
        cols["termination"].append(h.get("Termination", ""))
        cols["white_title"].append(h.get("WhiteTitle", ""))
        cols["black_title"].append(h.get("BlackTitle", ""))
        cols["moves"].append(" ".join(moves))
        kept += 1
        if len(cols["game_id"]) >= args.games_per_shard:
            flush()
            print(f"  shard {shard_idx-1}: {kept} games kept [{time.time()-t0:.0f}s]", flush=True)
    flush()

    manifest = dict(source=args.source, pgn=args.pgn, filter=gf.__dict__,
                    games_kept=kept, shards=shards, elapsed_s=round(time.time() - t0, 1))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n=== {out_dir}: {kept} games in {len(shards)} shard(s) [{time.time()-t0:.0f}s] ===")
    print("DONE build_game_records", flush=True)


if __name__ == "__main__":
    main()
