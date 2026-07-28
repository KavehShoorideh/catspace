#!/usr/bin/env python
"""experiments/build_player_dataset.py -- M2b data: extract the games of HIGH-VOLUME players from the
full records (they carry usernames), grouped per player, for training the style residual z. Two passes
over the parquet shards, BOTH restricted to a single SPEED bucket (locked M2b: single time control by
construction -- match Maia-2's base type; avoids the M2a mixed-TC confound): (1) count games/player in
that speed -> select players with >= --min-games; (2) extract those players' games in that speed.
Player identity is NAME-MASKED to a stable hash (locked decision 4: group by player, never feed the
name). Provisional pool (< --prov-threshold games) is pooled to estimate the rating-conditioned prior
p(z|Elo); those players get NO individual z (their z is tied to mu(Elo)) -- see MILESTONES M2b.
Output parquet: one row per (tracked_player, game): color, both Elos, result, moves, provisional flag,
time_control.
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


def speed(tc: str) -> str:
    """lichess speed bucket from a "base+inc" time control (estimated seconds = base + 40*inc)."""
    if not tc or tc == "-" or "+" not in tc:
        return "other"
    try:
        b, i = tc.split("+"); est = int(b) + 40 * int(i)
    except Exception:
        return "other"
    if est < 29:   return "ultra"
    if est < 179:  return "bullet"
    if est < 479:  return "blitz"
    if est < 1499: return "rapid"
    return "classical"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="data/records/lichess_2019-01")
    ap.add_argument("--speed", default="rapid", help="single speed bucket to keep (rapid matches Maia-2 rapid base)")
    ap.add_argument("--min-games", type=int, default=40)
    ap.add_argument("--n-players", type=int, default=6000, help="0 = all qualifying players")
    ap.add_argument("--prov-threshold", type=int, default=20, help="players with < this many games = PROVISIONAL (pooled prior)")
    ap.add_argument("--prov-players", type=int, default=40000, help="how many provisional players to sample for the prior")
    ap.add_argument("--out", default="data/records/player_games_rapid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    files = sorted(glob.glob(str(Path(args.records) / "*.parquet")))

    # pass 1: count games per player WITHIN the target speed bucket
    cnt = Counter()
    for f in files:
        t = pq.read_table(f, columns=["white_id", "black_id", "time_control"]).to_pydict()
        for wi, bi, tc in zip(t["white_id"], t["black_id"], t["time_control"]):
            if speed(tc) == args.speed:
                cnt[wi] += 1; cnt[bi] += 1
    qualifying = [u for u, c in cnt.items() if c >= args.min_games]
    if args.n_players and len(qualifying) > args.n_players:
        qualifying = list(rng.choice(qualifying, size=args.n_players, replace=False))
    individual = set(qualifying)
    prov_pool = [u for u, c in cnt.items() if c < args.prov_threshold]
    if args.prov_players and len(prov_pool) > args.prov_players:
        prov_pool = list(rng.choice(prov_pool, size=args.prov_players, replace=False))
    provisional = set(prov_pool)
    selected = individual | provisional
    print(f"[player-dataset speed={args.speed}] {len(cnt):,} players in speed | individual(>= {args.min_games}): "
          f"{len(individual):,} of {sum(c >= args.min_games for c in cnt.values()):,} | provisional(< {args.prov_threshold}): "
          f"{len(provisional):,} of {sum(c < args.prov_threshold for c in cnt.values()):,} [{time.time()-t0:.0f}s]", flush=True)

    # pass 2: extract selected players' games IN THE TARGET SPEED (row per tracked player x game)
    cols = {k: [] for k in ("player_id", "player_elo", "opp_elo", "player_white", "result",
                            "n_plies", "moves", "provisional", "time_control")}
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
                        "moves": pa.array(cols["moves"], pa.string()),
                        "provisional": pa.array(cols["provisional"], pa.bool_()),
                        "time_control": pa.array(cols["time_control"], pa.string())})
        pq.write_table(tbl, out_dir / f"players_{shard_idx:04d}.parquet", compression="zstd")
        shard_idx += 1
        for k in cols: cols[k] = []

    for f in files:
        t = pq.read_table(f, columns=["white_id", "black_id", "white_elo", "black_elo", "result",
                                       "moves", "n_plies", "time_control"]).to_pydict()
        for wi, bi, we, be, res, mv, npl, tc in zip(t["white_id"], t["black_id"], t["white_elo"],
                                                     t["black_elo"], t["result"], t["moves"],
                                                     t["n_plies"], t["time_control"]):
            if speed(tc) != args.speed:
                continue
            for name, is_white, pe, oe in ((wi, True, we, be), (bi, False, be, we)):
                if name in selected:
                    cols["player_id"].append(int(pid(name))); cols["player_elo"].append(int(pe))
                    cols["opp_elo"].append(int(oe)); cols["player_white"].append(is_white)
                    cols["result"].append(int(res)); cols["n_plies"].append(int(npl)); cols["moves"].append(mv)
                    cols["provisional"].append(name in provisional); cols["time_control"].append(tc)
                    rows += 1
        if len(cols["player_id"]) >= 300_000:
            flush()
    flush()
    print(f"\n=== {out_dir}: {rows:,} (player,game) rows, {shard_idx} shards, {len(selected):,} players "
          f"[{time.time()-t0:.0f}s] ===")
    print("DONE build_player_dataset", flush=True)


if __name__ == "__main__":
    main()
