#!/usr/bin/env python
"""
experiments/viz/build_decompose_viewer.py — populate
catspace/viz/templates/decompose_viewer.html: run the M1.5 meet-in-the-middle
decomposer (catspace/planner/decompose.py) on real middlegame starts (same
recipe as experiments/decompose_demo.py: tau_exec/tau_floor calibrated from
the data, not hand-set) and serialize the resulting hop trees, waypoint
chains, and population histograms. Boards are stored as FEN + ply only (not
pre-rendered SVG) -- the template draws whichever board is clicked, on demand.
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
from catspace.io.paths import derived_dir, generated_dir, newest_shard_dir
from catspace.nn.fb import load_ckpt, pick_device
from catspace.nn.features import feature_planes, omega_ids
from catspace.planner.decompose import WaypointPool, decompose, hop_reach
from catspace.viz.build_html import build_html

PLANNER_ELO, PLANNER_CLOCK = 1800, 300.0


def load_rows(shard_dir: Path, picks: list) -> dict:
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
    Fs, Bs = [], []
    om_row = omega_ids(np.array([PLANNER_ELO]), np.array([PLANNER_ELO]), np.array([PLANNER_CLOCK]))[0]
    n = len(data["packed"])
    for i in range(0, n, batch):
        planes = torch.from_numpy(feature_planes(
            data["packed"][i:i + batch], data["meta"][i:i + batch])).to(device)
        om = torch.from_numpy(np.tile(om_row, (len(planes), 1))).to(device)
        Fs.append(fb.embed_F(planes, om).cpu().numpy())
        Bs.append(fb.embed_B(planes).cpu().numpy())
    return np.concatenate(Fs), np.concatenate(Bs)


def board_info(data: dict, row: int) -> dict:
    return dict(fen=board_from_packed(data["packed"][row], data["meta"][row]).fen(),
               ply=int(data["ply"][row]))


def serialize_node(node, data: dict, pool: WaypointPool) -> dict:
    wp = None
    if node.waypoint is not None:
        wp = board_info(data, pool.labels[node.waypoint])
    d = dict(reach=round(float(node.reach), 4), depth=node.depth, status=node.status,
            bottleneck=(round(float(node.bottleneck), 4) if node.bottleneck is not None else None),
            detail=node.detail, wp=wp, left=None, right=None)
    if node.left is not None:
        d["left"] = serialize_node(node.left, data, pool)
        d["right"] = serialize_node(node.right, data, pool)
    return d


def hist(values: np.ndarray, n_bins: int = 20, lo=None, hi=None):
    lo = float(values.min()) if lo is None else lo
    hi = float(values.max()) if hi is None else hi
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(values, bins=n_bins, range=(lo, hi))
    return [round(float(e), 3) for e in edges], [int(c) for c in counts]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-pool", type=int, default=20_000)
    ap.add_argument("--n-starts", type=int, default=60)
    ap.add_argument("--start-ply-lo", type=int, default=20)
    ap.add_argument("--start-ply-hi", type=int, default=40)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--dry-gain", type=float, default=0.02)
    ap.add_argument("--n-show", type=int, default=24, help="starts serialized with full trees")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()
    ckpt_path = Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt"
    device = pick_device(args.device)

    t0 = time.time()
    fb, payload = load_ckpt(ckpt_path, device)
    fb.eval()
    zg = payload["zgoals"]["MATE_W"].numpy().astype(np.float32)
    z_goal = zg / np.linalg.norm(zg)
    step = payload.get("step", "?")
    print(f"load: {time.time() - t0:.1f}s  ckpt step={step}  shards={shard_dir.name} device={device}")

    t0 = time.time()
    picks = sample_shard_rows(shard_dir, args.n_pool + 4 * args.n_starts, seed=args.seed, holdout_only=True)
    data = load_rows(shard_dir, picks)
    print(f"sample+load {len(data['ply'])} holdout rows: {time.time() - t0:.1f}s")

    is_start = (data["ply"] >= args.start_ply_lo) & (data["ply"] <= args.start_ply_hi)
    start_idx = np.flatnonzero(is_start)[: args.n_starts]
    pool_idx = np.setdiff1d(np.arange(len(data["ply"])), start_idx)[: args.n_pool]

    t0 = time.time()
    F_all, B_all = embed_rows(fb, data, device)
    print(f"embed {len(F_all)} rows (F+B): {time.time() - t0:.1f}s")

    pool = WaypointPool(F=F_all[pool_idx], B=B_all[pool_idx], labels=[int(i) for i in pool_idx])

    t0 = time.time()
    npz = np.load(sorted(shard_dir.glob("shard_*.npz"))[0])
    gid, ply, res = npz["game_id"], npz["ply"], npz["result"]
    last_ply = np.zeros(gid.max() + 1, dtype=ply.dtype)
    np.maximum.at(last_ply, gid, ply)
    nw_rows = np.flatnonzero((res == 1) & (ply >= last_ply[gid] - 10) & (gid % 50 == 0))
    rng = np.random.default_rng(args.seed)
    nw_rows = rng.choice(nw_rows, size=min(2000, len(nw_rows)), replace=False)
    nw_data = {k: npz[k][np.sort(nw_rows)] for k in ("packed", "meta")}
    F_nw, _ = embed_rows(fb, nw_data, device)
    # fb.np_score_matrix == plain dot on non-quasimetric checkpoints; the only
    # correctly-calibrated score on quasimetric ones (2026-07-12)
    sp = fb.np_score_matrix
    tau_exec = float(np.median(sp(F_nw, z_goal[None, :])[:, 0]))
    reach_starts = sp(F_all[start_idx], z_goal[None, :])[:, 0]
    tau_floor = float(np.quantile(reach_starts, 0.10))
    print(f"calibration: tau_exec={tau_exec:.4f} (n_near_win={len(nw_rows)}) "
          f"tau_floor={tau_floor:.4f}  [{time.time() - t0:.1f}s]")

    t0 = time.time()
    results = []
    for si in start_idx:
        dec = decompose(F_all[si], z_goal, pool, tau_exec=tau_exec, tau_floor=tau_floor,
                        dry_gain=args.dry_gain, max_depth=args.max_depth, score_pairs=sp)
        results.append((int(si), dec))
    dt = time.time() - t0
    print(f"decompose {len(results)} starts: {dt:.1f}s ({1000 * dt / len(results):.1f} ms/start)")

    direct = np.array([hop_reach(F_all[si], z_goal, sp) for si, _ in results])
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

    way_plies = np.array([data["ply"][dec.pool.labels[w]]
                          for _, dec in results for w in dec.waypoints])
    start_plies_all = data["ply"][start_idx]
    if way_plies.size:
        print(f"waypoint ply: mean {way_plies.mean():.1f} vs start ply mean "
              f"{start_plies_all.mean():.1f} (pool mean {data['ply'][pool_idx].mean():.1f})")

    # ------------------------------------------------------------- serialize
    order = np.argsort(-gain)
    show_idx = list(order[: args.n_show])
    for i in order[::-1][:3]:                     # worst 3 for contrast
        if i not in show_idx:
            show_idx.append(i)

    starts_payload = []
    for i in show_idx:
        si, dec = results[i]
        waypoints = [board_info(data, pool.labels[w]) for w in dec.waypoints]
        starts_payload.append(dict(
            direct=round(float(direct[i]), 4), bottleneck=round(float(bottle[i]), 4),
            gain=round(float(gain[i]), 4), executable=bool(dec.executable),
            block_rule=dec.block_rule, start=board_info(data, si),
            tree=serialize_node(dec.root, data, pool), waypoints=waypoints))

    gain_bins, gain_counts = hist(gain, 20)
    if way_plies.size:
        lo = float(min(way_plies.min(), start_plies_all.min()))
        hi = float(max(way_plies.max(), start_plies_all.max()))
        ply_bins, way_counts = hist(way_plies, 20, lo, hi)
        _, start_counts = hist(start_plies_all, 20, lo, hi)
    else:
        ply_bins, way_counts = hist(start_plies_all, 20)
        start_counts = [0] * 20

    data_out = dict(
        meta=dict(title=f"catspace — decomposition explorer  ·  ckpt step {step}  ·  "
                        f"{len(results)} starts, {len(pool)} waypoint pool"),
        verdict=dict(frac_improved=float((gain > 0).mean()), mean_gain=float(gain.mean()),
                    frac_executable=float(execf.mean()), mean_waypoints=float(n_way.mean()),
                    tau_exec=tau_exec, tau_floor=tau_floor, block_rules=dict(rules)),
        hist_gain=dict(bins=gain_bins, counts=gain_counts),
        hist_ply=dict(bins=ply_bins, waypoint_counts=way_counts, start_counts=start_counts),
        starts=starts_payload,
    )

    out = Path(args.out) if args.out else generated_dir() / "decompose-viewer.html"
    template = Path(__file__).resolve().parents[2] / "catspace" / "viz" / "templates" / "decompose_viewer.html"
    build_html(template, data_out, out)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
