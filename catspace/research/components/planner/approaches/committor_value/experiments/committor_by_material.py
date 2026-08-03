#!/usr/bin/env python
"""catspace/research/components/planner/approaches/committor_value/experiments/committor_by_material.py -- stack the committor / outcome distribution BY MATERIAL
CLASS (Kaveh's insight): near the end of the game the states split CLEANLY into many branches with
LOW transitions (sharp, committed outcome basins); earlier (more material) they're a JUMBLE that
overlaps. Directly visualizes why the metastable OUTCOME structure emerges only at low material,
while the midgame is fast-mixing.

Ridgeline: committor c=P(win) distribution per piece-count bucket (many pieces top -> few bottom).
Right: mean transition prob (leak = 1 - p_own_basin) and outcome bimodality per bucket -- the
"barrier" rising as material falls.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

from catspace.research.components.encoder.approaches.reachability_field.experiments.train_clock_field import ClockField
from catspace.research.tools.training_infra.train.scaffold import resolve_device
from catspace.io import paths

BUCKETS = [(27, 33), (23, 27), (19, 23), (15, 19), (11, 15), (8, 11), (5, 8), (3, 5)]  # pieces [lo,hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=paths.experiment("field_fullgame_v3_final.pt"))
    ap.add_argument("--data", default=paths.derived("field_fullgame_v1.npz"))
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--out", default=paths.experiment("committor_by_material.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); rng = np.random.default_rng(args.seed)

    p = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = p["cfg"]; net = ClockField(cfg["d"], ch=cfg["ch"], blocks=cfg["blocks"], in_planes=112).to(dev)
    net.load_state_dict(p["state_dict"]); net.eval()
    z = np.load(args.data); planes = z["planes"]
    sub = rng.integers(0, len(planes), min(args.n, len(planes)))
    pieces = planes[sub][:, 0:12].reshape(len(sub), 12, -1).sum(axis=(1, 2)).astype(int)

    import torch.nn.functional as F
    comm, leak = [], []
    for i in range(0, len(sub), 4096):
        x = torch.from_numpy(planes[sub[i:i+4096]].astype(np.float32)).to(dev)
        with torch.no_grad():
            pe = F.softmax(net.d_mate_and_end(x)[1], 1).cpu().numpy()
        wdl = np.stack([pe[:, 0], pe[:, 1:5].sum(1), pe[:, 5]], 1); wdl /= wdl.sum(1, keepdims=True)
        comm.append(pe[:, 0]); leak.append(1 - wdl.max(1))
    comm = np.concatenate(comm); leak = np.concatenate(leak)

    from scipy.stats import gaussian_kde
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 9), width_ratios=[2.4, 1])
    xs = np.linspace(0, 1, 200)
    print("piece-bucket | n | mean-committor | mean-leak | bimodality (0..1, high=split)")
    means_leak, labels, bicoef = [], [], []
    for k, (lo, hi) in enumerate(BUCKETS):                    # top row = most pieces
        m = (pieces >= lo) & (pieces < hi)
        y0 = len(BUCKETS) - 1 - k                             # stack: many pieces at TOP
        lab = f"{lo}-{hi-1}p"
        labels.append(lab)
        if m.sum() < 30:
            means_leak.append(np.nan); bicoef.append(np.nan); continue
        c = comm[m]
        kde = gaussian_kde(np.clip(c, 1e-3, 1-1e-3)); dens = kde(xs); dens = dens / dens.max() * 0.9
        ax[0].fill_between(xs, y0, y0 + dens, alpha=0.7, color=plt.get_cmap("viridis")(k / len(BUCKETS)))
        ax[0].plot(xs, y0 + dens, color="k", lw=0.6, alpha=0.5)
        ax[0].text(-0.02, y0 + 0.15, f"{lab}\n(n={m.sum()})", ha="right", va="bottom", fontsize=9)
        # Sarle bimodality coefficient: (skew^2 + 1) / kurtosis  (>0.55 ~ bimodal / split)
        from scipy.stats import skew, kurtosis
        b = (skew(c) ** 2 + 1) / (kurtosis(c, fisher=True) + 3)
        means_leak.append(leak[m].mean()); bicoef.append(b)
        print(f"  {lab:7} | {m.sum():6d} | {c.mean():.3f} | {leak[m].mean():.3f} | {b:.3f}")
    ax[0].set_yticks([]); ax[0].set_xlabel("committor c = P(win)")
    ax[0].set_title("Committor distribution stacked by MATERIAL class\n(top=opening, many pieces -> bottom=endgame, few pieces)")
    ax[0].set_xlim(-0.12, 1.02)

    yy = np.arange(len(BUCKETS))[::-1]
    ax[1].plot(means_leak, yy, "o-", color="#c04040", label="mean transition prob (leak)")
    ax[1].plot(bicoef, yy, "s-", color="#3b6fb0", label="outcome bimodality (split)")
    ax[1].axvline(0.55, color="#3b6fb0", ls=":", lw=0.8)
    ax[1].set_yticks(yy[::-1]); ax[1].set_yticklabels(labels[::-1], fontsize=8)
    ax[1].set_xlabel("value"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[1].set_title("leak DOWN + bimodality UP\nas material falls (barrier rises)")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"VERDICT committor-by-material -> {args.out} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
