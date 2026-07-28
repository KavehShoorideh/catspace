#!/usr/bin/env python
"""experiments/m2b_sample.py -- M2b step 1: sample MIDGAME positions (ply >= --min-ply) where the
TRACKED player is to move, from data/records/player_games_rapid. Emits one row per sampled position
with the position fen, the ACTUAL move played (the style label), both Elos, and a split tag.

Splits (by PLAYER, so held-out players are genuinely unseen):
  train    : individual players (>= 40 rapid games) used for joint training; each owns a free Delta row
  heldout  : individual players held out entirely -> z recovered post-hoc at eval (the generalization test)
  prov     : provisional players (< 20 games) -> Delta tied to 0, their moves ESTIMATE the prior mu(Elo)
Only TRAIN players get a contiguous `pidx` (Delta-table index); heldout/prov carry pidx=-1.
Output: data/derived/m2b/positions.parquet.
"""
from __future__ import annotations

import argparse, glob, sys, time
from pathlib import Path

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="data/records/player_games_rapid")
    ap.add_argument("--out", default="data/derived/m2b/positions.parquet")
    ap.add_argument("--n-individual", type=int, default=0, help="subsample this many individual players (0=all)")
    ap.add_argument("--n-prov", type=int, default=0, help="subsample this many provisional players (0=all)")
    ap.add_argument("--no-prov", action="store_true", help="exclude provisional players entirely "
                    "(mu=0 runs: they contribute zero gradient, so caching them is wasted compute)")
    ap.add_argument("--per-player", type=int, default=350, help="cap positions per individual player")
    ap.add_argument("--per-prov-player", type=int, default=8, help="cap positions per provisional player")
    ap.add_argument("--per-game", type=int, default=6)
    ap.add_argument("--min-ply", type=int, default=16)
    ap.add_argument("--heldout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)

    files = sorted(glob.glob(str(Path(args.records) / "*.parquet")))
    # discover players / provisional flags
    ind_ids, prov_ids = set(), set()
    for f in files:
        t = pq.read_table(f, columns=["player_id", "provisional"]).to_pydict()
        for pidv, pr in zip(t["player_id"], t["provisional"]):
            (prov_ids if pr else ind_ids).add(pidv)
    ind_ids = np.array(sorted(ind_ids), dtype=np.uint64)
    prov_ids = np.array(sorted(prov_ids), dtype=np.uint64)
    if args.n_individual and len(ind_ids) > args.n_individual:
        ind_ids = np.sort(rng.choice(ind_ids, args.n_individual, replace=False))
    if args.no_prov:
        prov_ids = np.array([], dtype=np.uint64)
    elif args.n_prov and len(prov_ids) > args.n_prov:
        prov_ids = np.sort(rng.choice(prov_ids, args.n_prov, replace=False))
    ind_set, prov_set = set(ind_ids.tolist()), set(prov_ids.tolist())

    # split individual players -> train / heldout; assign pidx only to train
    perm = rng.permutation(len(ind_ids))
    n_hold = int(len(ind_ids) * args.heldout_frac)
    heldout_set = set(ind_ids[perm[:n_hold]].tolist())
    train_ids = ind_ids[perm[n_hold:]]
    pidx_of = {int(pid_): i for i, pid_ in enumerate(train_ids.tolist())}
    print(f"[m2b-sample] individual {len(ind_ids):,} (train {len(train_ids):,} / heldout {len(heldout_set):,}) | "
          f"provisional {len(prov_ids):,} [{time.time()-t0:.0f}s]", flush=True)

    per_player_cnt = {}                                    # player_id -> positions kept
    cols = {k: [] for k in ("player_id", "pidx", "prov", "elo_self", "elo_oppo", "white",
                            "fen", "played", "ply", "game_id", "split")}
    gid = 0
    for f in files:
        t = pq.read_table(f).to_pydict()
        for i in range(len(t["player_id"])):
            pidv = t["player_id"][i]
            if pidv not in ind_set and pidv not in prov_set:
                continue
            is_prov = pidv in prov_set
            cap = args.per_prov_player if is_prov else args.per_player
            if per_player_cnt.get(pidv, 0) >= cap:
                gid += 1; continue
            white = bool(t["player_white"][i]); es = int(t["player_elo"][i]); eo = int(t["opp_elo"][i])
            moves = t["moves"][i].split()
            n = len(moves)
            cand_ply = [p for p in range(args.min_ply, n - 1) if (p % 2 == 0) == white]
            if not cand_ply:
                gid += 1; continue
            take = min(args.per_game, len(cand_ply), cap - per_player_cnt.get(pidv, 0))
            sel = set(rng.choice(cand_ply, take, replace=False).tolist())
            split = "prov" if is_prov else ("heldout" if pidv in heldout_set else "train")
            pidx = pidx_of.get(int(pidv), -1) if split == "train" else -1
            b = chess.Board(); ok = True
            for p in range(n):
                if p in sel and not b.is_game_over():
                    cols["player_id"].append(int(pidv)); cols["pidx"].append(pidx)
                    cols["prov"].append(is_prov); cols["elo_self"].append(es); cols["elo_oppo"].append(eo)
                    cols["white"].append(white); cols["fen"].append(b.fen()); cols["played"].append(moves[p])
                    cols["ply"].append(p); cols["game_id"].append(gid); cols["split"].append(split)
                    per_player_cnt[pidv] = per_player_cnt.get(pidv, 0) + 1
                try:
                    b.push_uci(moves[p])
                except Exception:
                    ok = False; break
            gid += 1
        print(f"  scanned {f.split('/')[-1]} | positions {len(cols['fen']):,} [{time.time()-t0:.0f}s]", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tbl = pa.table({
        "player_id": pa.array(cols["player_id"], pa.uint64()),
        "pidx": pa.array(cols["pidx"], pa.int32()),
        "prov": pa.array(cols["prov"], pa.bool_()),
        "elo_self": pa.array(cols["elo_self"], pa.int16()),
        "elo_oppo": pa.array(cols["elo_oppo"], pa.int16()),
        "white": pa.array(cols["white"], pa.bool_()),
        "fen": pa.array(cols["fen"], pa.string()),
        "played": pa.array(cols["played"], pa.string()),
        "ply": pa.array(cols["ply"], pa.int16()),
        "game_id": pa.array(cols["game_id"], pa.int32()),
        "split": pa.array(cols["split"], pa.string())})
    pq.write_table(tbl, args.out, compression="zstd")
    n = len(cols["fen"])
    sp = {s: cols["split"].count(s) for s in ("train", "heldout", "prov")}
    n_train_players = len({p for p, s in zip(cols["player_id"], cols["split"]) if s == "train"})
    print(f"\n=== {args.out}: {n:,} positions | split {sp} | train-players {n_train_players:,} "
          f"[{time.time()-t0:.0f}s] ===")
    print("DONE m2b_sample", flush=True)


if __name__ == "__main__":
    main()
