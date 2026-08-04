#!/usr/bin/env python
"""catspace/research/components/planner/approaches/atlas_region_stats/experiments/transition_map.py -- MAP THE HUMAN TRANSITION POINTS (Kaveh's metastability focus).
UMAP-projects the trained reachability field's embedding phi and marks where humans crossed BETWEEN
the three basins (Win / Draw / Loss). Chess = 3 metastable basins; the connections are the transition
points -- the moves where the outcome flipped under real human play.

Per sampled position: phi (for UMAP), committor c=P(win), and the 3-basin distribution (W/D/L) from
the ending head. Two transition notions:
  STATIC   : the position is CONTESTED -- no basin has probability >= theta_static (on the ridge).
  EMPIRICAL: along the real game line, the committor JUMPS |dc| >= theta_emp to the next position
             (the human made an outcome-flipping move) -- literally "where humans transitioned".
Thresholds are chosen from the data (histogram valley / a sparse-minority ridge), not fixed blindly.
Output: a UMAP figure colored by basin/committor with transition points highlighted, + a stats print.
"""
from __future__ import annotations

import argparse, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from catspace.research.components.encoder.approaches.reachability_field.experiments.train_clock_field import ClockField
from catspace.research.tools.training_infra.train.scaffold import resolve_device
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=paths.experiment("field_fullgame_v3_final.pt"))
    ap.add_argument("--data", default=paths.derived("field_fullgame_v1.npz"))
    ap.add_argument("--games", type=int, default=1200, help="number of games to sample trajectories from")
    ap.add_argument("--theta-emp", type=float, default=0.0, help="0 = pick from the |dc| distribution")
    ap.add_argument("--out", default=paths.experiment("transition_map.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device(args.device if hasattr(args, "device") else "auto")

    p = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = p["cfg"]; net = ClockField(cfg["d"], ch=cfg["ch"], blocks=cfg["blocks"], in_planes=112).to(dev)
    net.load_state_dict(p["state_dict"]); net.eval()
    z = np.load(args.data); planes = z["planes"]; game = z["game"]; ply = z["ply"]
    rng = np.random.default_rng(args.seed)

    # pick whole GAMES (need trajectories for the empirical committor jump)
    by_game = defaultdict(list)
    for i in range(len(game)):
        by_game[int(game[i])].append(i)
    games = [g for g in by_game if len(by_game[g]) >= 3]
    sel_games = rng.choice(games, size=min(args.games, len(games)), replace=False)
    idx = np.array(sorted(i for g in sel_games for i in by_game[g]))
    print(f"[transition-map] {len(sel_games)} games, {len(idx)} positions | device {dev}", flush=True)

    import torch.nn.functional as F
    phi_l, c_l, dist_l = [], [], []
    for i in range(0, len(idx), 2048):
        x = torch.from_numpy(planes[idx[i:i+2048]].astype(np.float32)).to(dev)
        with torch.no_grad():
            phi_l.append(net.phi(x).cpu().numpy())
            logits = net.d_mate_and_end(x)[1]
            pe = F.softmax(logits, 1).cpu().numpy()
            c_l.append(pe[:, 0])                                 # committor = P(win)
            dist_l.append(np.stack([pe[:, 0], pe[:, 1:5].sum(1), pe[:, 5]], 1))   # W/D/L
    phi = np.concatenate(phi_l); comm = np.concatenate(c_l); dist = np.concatenate(dist_l)
    basin = dist.argmax(1)                                        # 0=win 1=draw 2=loss

    # EMPIRICAL transition: committor jump to the next sampled position on the SAME game line
    pos_of = {v: k for k, v in enumerate(idx)}
    dc = np.zeros(len(idx))
    for g in sel_games:
        rows = sorted(by_game[g], key=lambda i: ply[i])
        for a, b in zip(rows[:-1], rows[1:]):
            dc[pos_of[a]] = comm[pos_of[b]] - comm[pos_of[a]]     # signed committor change ahead
    adc = np.abs(dc)
    # threshold from the data: the "knee" -- 85th percentile of |dc| (sparse minority = the crossings)
    theta = args.theta_emp or float(np.percentile(adc[adc > 0], 85))
    trans = adc >= theta
    print(f"|dc| distribution: p50 {np.percentile(adc,50):.3f} p85 {np.percentile(adc,85):.3f} "
          f"p95 {np.percentile(adc,95):.3f} max {adc.max():.3f}")
    print(f"THRESHOLD theta_emp = {theta:.3f} -> {trans.mean():.1%} transition points "
          f"| basins: win {(basin==0).mean():.0%} draw {(basin==1).mean():.0%} loss {(basin==2).mean():.0%}", flush=True)

    # UMAP the embedding
    import umap
    print("  running UMAP...", flush=True)
    xy = umap.UMAP(n_neighbors=25, min_dist=0.15, metric="euclidean", random_state=args.seed).fit_transform(phi)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(17, 8))
    # LEFT: committor field (basins) -- blue win .. white ridge .. red loss
    sc = ax[0].scatter(xy[:, 0], xy[:, 1], c=comm, cmap="coolwarm_r", s=5, alpha=0.6)
    ax[0].set_title("Reachability field (UMAP) colored by committor c=P(win)\nblue=win basin  white=ridge  red=loss basin")
    plt.colorbar(sc, ax=ax[0], label="committor c")
    # RIGHT: quiet points faint by basin; TRANSITION points highlighted
    col = np.array(["#3b6fb0", "#7a7a7a", "#c04040"])            # win / draw / loss
    ax[1].scatter(xy[~trans, 0], xy[~trans, 1], c=col[basin[~trans]], s=4, alpha=0.25)
    ax[1].scatter(xy[trans, 0], xy[trans, 1], c="black", s=22, marker="X",
                  edgecolors="yellow", linewidths=0.5, label=f"transition (|dc|>={theta:.2f})")
    ax[1].set_title(f"Human TRANSITION points (committor jump |dc|>={theta:.2f}, top-15%)\n"
                    f"faint = quiet basin points (blue win / grey draw / red loss)")
    ax[1].legend(loc="upper right")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"VERDICT transition-map: {trans.sum()} transition pts / {len(idx)} | theta {theta:.3f} "
          f"-> {args.out} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
