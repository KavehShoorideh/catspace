#!/usr/bin/env python
"""experiments/transition_bands.py -- the 3 metastable basins as 3 BANDS full of states, each state
colored by its TRANSITION PROBABILITY to a DIFFERENT band (Kaveh). Chess = {Win, Draw, Loss} basins;
a state's leak probability = the outcome-distribution mass OUTSIDE its own basin = 1 - p_own_basin.
Deep-basin (quiet) states leak ~0; boundary (transition) states leak toward another band.

Per state (from the trained field's ending head): W/D/L distribution -> basin = argmax, own_p, and
leak = 1 - own_p, with the leak split by target band. Renders: (A) 3 bands, y = leak probability,
color = leak (transition prob to ANY other band); (B) same, color = the DOMINANT target band
(directionality); (C) the aggregate 3x3 basin->basin mean transition-probability matrix.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_clock_field import ClockField
from catspace.research.tools.training_infra.train.scaffold import resolve_device

BASINS = ["Win", "Draw", "Loss"]
BCOL = {0: "#3b6fb0", 1: "#7a7a7a", 2: "#c04040"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/field_fullgame_v3_final.pt")
    ap.add_argument("--data", default="data/derived/field_fullgame_v1.npz")
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--out", default="artifacts/experiments/transition_bands.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); rng = np.random.default_rng(args.seed)

    p = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = p["cfg"]; net = ClockField(cfg["d"], ch=cfg["ch"], blocks=cfg["blocks"], in_planes=112).to(dev)
    net.load_state_dict(p["state_dict"]); net.eval()
    z = np.load(args.data); planes = z["planes"]
    sub = rng.integers(0, len(planes), min(args.n, len(planes)))

    import torch.nn.functional as F
    dist = []
    for i in range(0, len(sub), 2048):
        x = torch.from_numpy(planes[sub[i:i+2048]].astype(np.float32)).to(dev)
        with torch.no_grad():
            pe = F.softmax(net.d_mate_and_end(x)[1], 1).cpu().numpy()
        dist.append(np.stack([pe[:, 0], pe[:, 1:5].sum(1), pe[:, 5]], 1))     # W/D/L
    dist = np.concatenate(dist); dist /= dist.sum(1, keepdims=True)
    basin = dist.argmax(1); own_p = dist[np.arange(len(dist)), basin]; leak = 1 - own_p
    # dominant target band (the largest of the OTHER two)
    target = np.empty(len(dist), int)
    for i in range(len(dist)):
        others = [j for j in range(3) if j != basin[i]]
        target[i] = others[int(np.argmax(dist[i, others]))]
    # aggregate 3x3 mean transition-probability matrix M[i,j] = mean over basin-i states of p_j (j!=i)
    M = np.zeros((3, 3))
    for i in range(3):
        m = basin == i
        if m.any():
            M[i] = dist[m].mean(0)
        M[i, i] = 0.0

    print(f"[bands] {len(sub)} states | basin sizes W {int((basin==0).sum())} D {int((basin==1).sum())} "
          f"L {int((basin==2).sum())}", flush=True)
    print("BASIN->BASIN mean transition probability (rows=from, cols=to):")
    print("        ->Win   ->Draw  ->Loss")
    for i in range(3):
        print(f"  {BASINS[i]:5} {M[i,0]:6.3f}  {M[i,1]:6.3f}  {M[i,2]:6.3f}   (mean leak {leak[basin==i].mean():.3f})")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[3, 3, 1.6])
    axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1]); axM = fig.add_subplot(gs[2])

    for ax, mode, title in [(axA, "leak", "colored by leak = P(transition to ANY other band)"),
                            (axB, "target", "colored by DOMINANT target band")]:
        for b in range(3):
            m = basin == b
            x = b + (rng.random(m.sum()) - 0.5) * 0.8          # jitter to fill the band
            y = leak[m]
            if mode == "leak":
                sc = ax.scatter(x, y, c=leak[m], cmap="viridis", s=6, alpha=0.6, vmin=0, vmax=0.66)
            else:
                ax.scatter(x, y, c=[BCOL[t] for t in target[m]], s=6, alpha=0.55)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(BASINS)
        ax.set_ylabel("transition probability (leak = 1 - p_own_basin)")
        ax.set_ylim(-0.02, 1.0); ax.set_title(title)
        ax.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.5)
    fig.colorbar(sc, ax=axA, label="leak prob", fraction=0.046)
    axB.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc=BCOL[i], mec="none", label=f"->{BASINS[i]}") for i in range(3)],
               loc="upper right", fontsize=8)

    im = axM.imshow(M, cmap="magma", vmin=0, vmax=0.5)
    axM.set_xticks(range(3)); axM.set_xticklabels(BASINS); axM.set_yticks(range(3)); axM.set_yticklabels(BASINS)
    axM.set_xlabel("to basin"); axM.set_ylabel("from basin"); axM.set_title("mean transition matrix")
    for i in range(3):
        for j in range(3):
            axM.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                     color="white" if M[i, j] < 0.3 else "black", fontsize=11)
    fig.colorbar(im, ax=axM, fraction=0.046)
    fig.suptitle("The three metastable basins as bands; color = transition probability to a different band", fontsize=13)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"VERDICT transition-bands -> {args.out} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
