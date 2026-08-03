#!/usr/bin/env python
"""experiments/transition_time.py -- the 3 metastable basins over the GAME TIMELINE (Kaveh: one axis
= move number). Three stacked bands (Win / Draw / Loss); x = move number; each state colored by its
transition probability to a DIFFERENT band (leak = 1 - p_own_basin). Companion: mean transition
probability vs move number per basin -- does the game get more transition-prone (leakier basins) as
it goes, and when do win<->loss swings peak.
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
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--max-move", type=int, default=70)
    ap.add_argument("--out", default="artifacts/experiments/transition_time.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); rng = np.random.default_rng(args.seed)

    p = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = p["cfg"]; net = ClockField(cfg["d"], ch=cfg["ch"], blocks=cfg["blocks"], in_planes=112).to(dev)
    net.load_state_dict(p["state_dict"]); net.eval()
    z = np.load(args.data); planes = z["planes"]; ply = z["ply"]
    sub = rng.integers(0, len(planes), min(args.n, len(planes)))
    move = (ply[sub] // 2) + 1                                    # full move number from ply

    import torch.nn.functional as F
    dist = []
    for i in range(0, len(sub), 2048):
        x = torch.from_numpy(planes[sub[i:i+2048]].astype(np.float32)).to(dev)
        with torch.no_grad():
            pe = F.softmax(net.d_mate_and_end(x)[1], 1).cpu().numpy()
        dist.append(np.stack([pe[:, 0], pe[:, 1:5].sum(1), pe[:, 5]], 1))
    dist = np.concatenate(dist); dist /= dist.sum(1, keepdims=True)
    basin = dist.argmax(1); leak = 1 - dist[np.arange(len(dist)), basin]

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2.2, 1], sharex=True)

    # TOP: 3 stacked bands, x = move number, within-band y = leak, color = leak
    band_h = 1.0
    for row, b in enumerate([0, 1, 2]):                            # Win top, Draw mid, Loss bottom
        m = (basin == b) & (move <= args.max_move)
        y0 = (2 - row) * (band_h + 0.15)                          # stack bands vertically
        sc = ax[0].scatter(move[m] + (rng.random(m.sum()) - 0.5) * 0.7,
                           y0 + leak[m] * band_h, c=leak[m], cmap="viridis",
                           s=6, alpha=0.55, vmin=0, vmax=0.66)
        ax[0].axhline(y0, color="k", lw=0.6, alpha=0.3)
        ax[0].text(-1.5, y0 + band_h / 2, BASINS[b], ha="right", va="center", fontsize=12,
                   color=BCOL[b], weight="bold")
    ax[0].set_yticks([]); ax[0].set_ylabel("3 basins as bands (within-band height = transition prob)")
    ax[0].set_title("The three metastable basins over the game timeline — color = transition probability to another band")
    fig.colorbar(sc, ax=ax[0], label="leak = P(transition to other band)", fraction=0.04)

    # BOTTOM: mean transition probability vs move number, per basin
    moves = np.arange(1, args.max_move + 1)
    for b in range(3):
        ml = np.array([leak[(basin == b) & (move == mm)].mean() if ((basin == b) & (move == mm)).sum() >= 8
                       else np.nan for mm in moves])
        ax[1].plot(moves, ml, color=BCOL[b], lw=2, label=BASINS[b])
    ax[1].set_xlabel("move number"); ax[1].set_ylabel("mean transition prob")
    ax[1].set_title("Mean transition probability vs move number, per basin")
    ax[1].legend(); ax[1].grid(alpha=0.3); ax[1].set_xlim(1, args.max_move)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    # quick numeric trend
    early = leak[move <= 15].mean(); mid = leak[(move > 15) & (move <= 35)].mean(); late = leak[move > 35].mean()
    print(f"[transition-time] mean leak: opening(<=15) {early:.3f} | midgame(16-35) {mid:.3f} | late(>35) {late:.3f}")
    print(f"VERDICT -> {args.out} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
