#!/usr/bin/env python
"""experiments/build_player_dataset.py -- M2b data: extract the games of HIGH-VOLUME players from the
full 19.35M-game records (they carry usernames), grouped per player, for training the style residual
z. Two passes over the parquet shards: (1) count games/player -> select players with >= --min-games;
(2) extract those players' games. Player identity is NAME-MASKED to a stable hash (locked decision 4:
group by player, never feed the name). Output parquet: one row per (tracked_player, game) with the
tracked player's color, both Elos, result, and the game's UCI moves.
"""
from __future__ import annotations

import argparse, glob, hashlib, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pid(user: str) -> np.uint64:
    return np.uint64(int.from_bytes(hashlib.blake2b(user.encode("utf-8", "replace"), digest_size=8).digest(), "big"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="data/records/lichess_2019-01")
    ap.add_argument("--min-games", type=int, default=50)
    ap.add_argument("--n-players", type=int, default=5000, help="0 = all qualifying players")
    ap.add_argument("--out", default="data/records/player_games")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    files = sorted(glob.glob(str(Path(args.records) / "*.parquet")))

    # pass 1: count games per player
    cnt = Counter()
    for f in files:
        t = pq.read_table(f, columns=["white_id", "black_id"]).to_pydict()
        cnt.update(t["white_id"]); cnt.update(t["black_id"])
    qualifying = [u for u, c in cnt.items() if c >= args.min_games]
    if args.n_players and len(qualifying) > args.n_players:
        qualifying = list(rng.choice(qualifying, size=args.n_players, replace=False))
    selected = set(qualifying)
    print(f"[player-dataset] {len(cnt):,} players | >= {args.min_games} games: "
          f"{sum(c >= args.min_games for c in cnt.values()):,} | selected {len(selected):,} [{time.time()-t0:.0f}s]", flush=True)

    # pass 2: extract selected players' games (row per tracked player x game)
    RMAP = {1: 1, -1: -1, 0: 0}
    cols = {k: [] for k in ("player_id", "player_elo", "opp_elo", "player_white", "result", "n_plies", "moves")}
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0; rows = 0
    def flush():
        nonlocal shard_idx
        if not cols["player_id"]:
            return
        tbl = pa.table({"player_id": pa.array(cols["player_id"], pa.uint64()),
                        "player_elo": pa.array(cols["player_elo"], pa.int16()),
                        "opp_elo": pa.array(cols["opp_elo"], pa.int16()),
                        "player_white": pa.array(cols["player_white"], pa.bool_()),
                        "result": pa.array(cols["result"], pa.int8()),
                        "n_plies": pa.array(cols["n_plies"], pa.int32()),
                        "moves": pa.array(cols["moves"], pa.string())})
        pq.write_table(tbl, out_dir / f"players_{shard_idx:04d}.parquet", compression="zstd")
        shard_idx += 1
        for k in cols: cols[k] = []

    for f in files:
        t = pq.read_table(f, columns=["white_id", "black_id", "white_elo", "black_elo", "result", "moves", "n_plies"]).to_pydict()
        for wi, bi, we, be, res, mv, npl in zip(t["white_id"], t["black_id"], t["white_elo"],
                                                 t["black_elo"], t["result"], t["moves"], t["n_plies"]):
            for name, is_white, pe, oe in ((wi, True, we, be), (bi, False, be, we)):
                if name in selected:
                    cols["player_id"].append(int(pid(name))); cols["player_elo"].append(int(pe))
                    cols["opp_elo"].append(int(oe)); cols["player_white"].append(is_white)
                    cols["result"].append(int(res)); cols["n_plies"].append(int(npl)); cols["moves"].append(mv)
                    rows += 1
        if len(cols["player_id"]) >= 300_000:
            flush()
    flush()
    print(f"\n=== {out_dir}: {rows:,} (player,game) rows, {shard_idx} shards, {len(selected):,} players "
          f"[{time.time()-t0:.0f}s] ===")
    print("DONE build_player_dataset", flush=True)


if __name__ == "__main__":
    main()
