#!/usr/bin/env python
"""
experiments/decompose_demo.py — first real-board run of the M1.5 geodesic-
midpoint decomposer (planner/decompose.py): take middlegame positions from
HOLDOUT games, set the goal z = zMATE_W (unit-normalized so both legs of the
bottleneck are cosines), and recursively split the hop through a pool of
holdout waypoint positions.

Thresholds are calibrated from the data, not hand-set:
  tau_exec   median reach of positions <= 10 plies before a white win — "as
             reachable as mate is when mate is actually imminent"
  tau_floor  q10 of reach over all starts — below this we're in territory the
             field thinks is unlikely for anyone

Verdicts:
  FRAC_IMPROVED    starts where the best split beats the direct reach
  MEAN_GAIN        mean (plan_bottleneck - direct) over starts
  FRAC_EXECUTABLE  starts decomposed into all-executable hops
  block histogram + example plans (FEN chains, plies) + stage timings
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from catspace.data.encode import board_from_packed
from catspace.data.shards import sample_shard_rows
from catspace.io.paths import derived_dir, newest_shard_dir
from catspace.nn.fb import load_ckpt, pick_device
from catspace.nn.features import feature_planes, omega_ids
from catspace.planner.decompose import WaypointPool, decompose, hop_reach

PLANNER_ELO, PLANNER_CLOCK = 1800, 300.0


def load_rows(shard_dir: Path, picks: list) -> dict:
    """Gather sampled (file,row) picks into one dict of stacked arrays."""
    by_file: dict = {}
    for name, row in picks:
        by_file.setdefault(name, []).append(row)
    cols = ("packed", "meta", "ply", "result", "game_id")
    out: dict = {k: [] for k in cols}
    for name, rows in sorted(by_file.items()):
        npz = np.load(shard_dir / name)
        idx = np.array(sorted(rows))
        for k in cols:
            out[k].append(npz[k][idx])
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def embed_rows(fb, data, device, batch=2048):
    """(F, B) under the planner's omega, unit rows, numpy."""
    Fs, Bs = [], []
    om_row = omega_ids(np.array([PLANNER_ELO]), np.array([PLANNER_ELO]),
                       np.array([PLANNER_CLOCK]))[0]
    n = len(data["packed"])
    for i in range(0, n, batch):
        planes = torch.from_numpy(feature_planes(
            data["packed"][i:i + batch], data["meta"][i:i + batch])).to(device)
        om = torch.from_numpy(np.tile(om_row, (len(planes), 1))).to(device)
        Fs.append(fb.embed_F(planes, om).cpu().numpy())
        Bs.append(fb.embed_B(planes).cpu().numpy())
    return np.concatenate(Fs), np.concatenate(Bs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-pool", type=int, default=20_000)
    ap.add_argument("--n-starts", type=int, default=200)
    ap.add_argument("--start-ply-lo", type=int, default=20)
    ap.add_argument("--start-ply-hi", type=int, default=40)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--dry-gain", type=float, default=0.02)
    ap.add_argument("--device", default="cpu",
                    help="default cpu: demo-sized, and MPS may be busy training")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()
    ckpt_path = Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt"
    device = pick_device(args.device)

    t0 = time.time()
    fb, payload = load_ckpt(ckpt_path, device)
    fb.eval()
    zg = payload["zgoals"]["MATE_W"].numpy().astype(np.float32)
    z_goal = zg / np.linalg.norm(zg)          # unit: leg2 on the same cosine scale as leg1
    print(f"load: {time.time() - t0:.1f}s  shards={shard_dir.name} device={device}")

    t0 = time.time()
    picks = sample_shard_rows(shard_dir, args.n_pool + 4 * args.n_starts,
                              seed=args.seed, holdout_only=True)
    data = load_rows(shard_dir, picks)
    print(f"sample+load {len(data['ply'])} holdout rows: {time.time() - t0:.1f}s")

    # starts: middlegame slice; pool: everything else
    is_start = ((data["ply"] >= args.start_ply_lo) & (data["ply"] <= args.start_ply_hi))
    start_idx = np.flatnonzero(is_start)[: args.n_starts]
    pool_idx = np.setdiff1d(np.arange(len(data["ply"])), start_idx)[: args.n_pool]

    t0 = time.time()
    F_all, B_all = embed_rows(fb, data, device)
    print(f"embed {len(F_all)} rows (F+B): {time.time() - t0:.1f}s")

    pool = WaypointPool(F=F_all[pool_idx], B=B_all[pool_idx],
                        labels=[(int(i)) for i in pool_idx])

    # ---------------------------------------------------------- calibration
    # near-win anchor: positions <= 10 plies before the END OF THEIR OWN GAME
    # in white wins, from one full shard (the sampled rows are too sparse to
    # recover per-game finality)
    t0 = time.time()
    npz = np.load(sorted(shard_dir.glob("shard_*.npz"))[0])
    gid, ply, res = npz["game_id"], npz["ply"], npz["result"]
    last_ply = np.zeros(gid.max() + 1, dtype=ply.dtype)
    np.maximum.at(last_ply, gid, ply)
    nw_rows = np.flatnonzero((res == 1) & (ply >= last_ply[gid] - 10)
                             & (gid % 50 == 0))            # holdout games only
    rng = np.random.default_rng(args.seed)
    nw_rows = rng.choice(nw_rows, size=min(2000, len(nw_rows)), replace=False)
    nw_data = {k: npz[k][np.sort(nw_rows)] for k in ("packed", "meta")}
    F_nw, _ = embed_rows(fb, nw_data, device)
    tau_exec = float(np.median(F_nw @ z_goal))
    reach_starts = F_all[start_idx] @ z_goal
    tau_floor = float(np.quantile(reach_starts, 0.10))
    print(f"calibration: tau_exec={tau_exec:.4f} (n_near_win={len(nw_rows)}) "
          f"tau_floor={tau_floor:.4f}  [{time.time() - t0:.1f}s]")

    # ---------------------------------------------------------- decompose
    t0 = time.time()
    results = []
    for si in start_idx:
        dec = decompose(F_all[si], z_goal, pool, tau_exec=tau_exec,
                        tau_floor=tau_floor, dry_gain=args.dry_gain,
                        max_depth=args.max_depth)
        results.append((si, dec))
    dt = time.time() - t0
    print(f"decompose {len(results)} starts: {dt:.1f}s ({1000 * dt / len(results):.1f} ms/start)")

    direct = np.array([hop_reach(F_all[si], z_goal) for si, _ in results])
    bottle = np.array([dec.plan_bottleneck for _, dec in results])
    gain = bottle - direct
    execf = np.array([dec.executable for _, dec in results])
    rules = Counter(dec.block_rule for _, dec in results if dec.block_rule)
    n_way = np.array([len(dec.waypoints) for _, dec in results])

    print(f"\nVERDICT FRAC_IMPROVED={float((gain > 0).mean()):.3f} "
          f"MEAN_GAIN={float(gain.mean()):.4f} "
          f"FRAC_EXECUTABLE={float(execf.mean()):.3f} "
          f"MEAN_WAYPOINTS={float(n_way.mean()):.2f}")
    print(f"block rules: {dict(rules)}")

    # waypoint sanity: do chosen waypoints sit LATER in games than the starts?
    way_plies = np.array([data["ply"][dec.pool.labels[w]]
                          for _, dec in results for w in dec.waypoints])
    if way_plies.size:
        print(f"waypoint ply: mean {way_plies.mean():.1f} vs start ply mean "
              f"{data['ply'][start_idx].mean():.1f} (pool mean {data['ply'][pool_idx].mean():.1f})")

    print("\nexample plans (best-gain starts):")
    for i in np.argsort(-gain)[:3]:
        si, dec = results[i]
        start_fen = board_from_packed(data["packed"][si], data["meta"][si]).fen()
        print(f"  direct {direct[i]:+.4f} -> bottleneck {bottle[i]:+.4f} "
              f"({'executable' if dec.executable else dec.block_rule}, "
              f"{len(dec.waypoints)} waypoints)")
        print(f"    start (ply {int(data['ply'][si])}): {start_fen}")
        for w in dec.waypoints:
            r = dec.pool.labels[w]
            fen = board_from_packed(data["packed"][r], data["meta"][r]).fen()
            print(f"    via   (ply {int(data['ply'][r])}): {fen}")


if __name__ == "__main__":
    main()
