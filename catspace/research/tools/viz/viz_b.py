#!/usr/bin/env python
"""catspace/research/tools/viz/viz_b.py -- visualize the B (goal-side) embedding (Kaveh 2026-07-21). B(g) is where "mate is a
cluster" should live: near-mate (low-DTM) positions ought to cluster together, and the geometry should be
graded by DTM. t-SNE of B(g) over the endgame set, colored by DTM (main) and by piece count (panel 2), with
the near-mate (DTM<=3) region marked -- so we can see whether the goal side has the clustered/graded
structure the planner assumes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
from catspace.io import paths
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default=paths.sep("iqe_nucleus_gn.pt"))
    ap.add_argument("--dtm-npz", default=paths.derived("dtm_endgame.npz"))
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--out", default=paths.experiment("viz_b.png"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    dz = np.load(args.dtm_npz); dtm = np.asarray(dz["dtm"]).astype(float)
    P, M = np.asarray(dz["packed"]), np.asarray(dz["meta"])
    idx = np.flatnonzero(dtm > 0); idx = idx[rng.permutation(len(idx))[:args.n]]
    pc = np.array([len(board_from_packed(P[i], M[i]).piece_map()) for i in idx])
    dk = dtm[idx]
    with torch.no_grad():
        B = fb.embed_B(torch.from_numpy(feature_planes(P[idx], M[idx])).to(dev)).cpu().numpy()
    XY = TSNE(n_components=2, perplexity=30, init="pca", random_state=args.seed).fit_transform(B)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    dcol = np.clip(dk, 0, np.quantile(dk, 0.95))
    order = np.argsort(-dcol)                                    # near-mate on top
    s0 = ax[0].scatter(XY[order, 0], XY[order, 1], c=dcol[order], cmap="viridis_r", s=12, alpha=0.8)
    mate = dk <= 3
    ax[0].scatter(XY[mate, 0], XY[mate, 1], facecolors="none", edgecolors="red", s=45, lw=0.8, label=f"near-mate DTM<=3 ({mate.sum()})")
    ax[0].set_title("B (goal) embedding, colored by DTM\n(bright=near mate; is the mate region a cluster?)")
    ax[0].legend(loc="upper right"); fig.colorbar(s0, ax=ax[0], label="DTM (plies to mate)")

    s1 = ax[1].scatter(XY[:, 0], XY[:, 1], c=pc, cmap="tab10", s=12, alpha=0.8)
    ax[1].set_title("same B embedding, colored by piece count\n(does the goal side separate by material?)")
    fig.colorbar(s1, ax=ax[1], label="piece count")
    # cohesion: mean pairwise 2D distance within near-mate vs overall (rough cluster check)
    def spread(pts):
        c = pts.mean(0); return float(np.sqrt(((pts - c) ** 2).sum(1)).mean())
    coh = spread(XY[mate]) / (spread(XY) + 1e-9)
    fig.suptitle(f"B-field embedding -- field={Path(args.field).stem}  "
                 f"(near-mate cluster spread / overall spread = {coh:.2f}; <1 => mate IS a tighter cluster)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"VERDICT VIZ_B saved {args.out} | n={args.n} near-mate={mate.sum()} "
          f"mate-cluster-spread/overall={coh:.3f} (<1 => mate clusters tighter)")


if __name__ == "__main__":
    main()
