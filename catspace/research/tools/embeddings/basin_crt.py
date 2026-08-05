#!/usr/bin/env python
"""basin_crt.py -- the CRT-gun / Stern-Gerlach view (Kaveh 2026-08-04).

The beam is fired from the START position at the left, travels rightward along the PLY axis, is
deflected transversely by the field, and lands on a screen at the right where it has split three
ways: White win (up), draw (centre), Black win (down).

This is the SIDE view. The polar plot was looking down the barrel -- it shows the angular split but
collapses the axis the whole picture is about. Same data, rotated, plus the screen.

Deflection is WHITE-POV: the poles are mover-POV and the mover alternates every ply, so a raw
mover-POV deflection would zigzag once per ply and the beam would be noise.

Trajectories are FULL-GAME replays, every ply, so a beam path is continuous rather than 8 scattered
dots. Games are sampled UNIFORMLY -- the beam should show the real population (81% of engine games
draw), not a balanced one.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np


def parity_smooth(v, w=2):
    """Cancel the ply-parity alternation. The mover swaps every ply and the field is
    turn-dependent (lag-1 autocorrelation 0.42 vs lag-2 0.63), so a raw beam path sawtooths at
    full amplitude and reads as noise rather than as a beam. A width-2 mean removes a pure parity
    alternation exactly and leaves the real deflection."""
    return np.convolve(v, np.ones(w) / w, mode="same") if len(v) >= w else v


def split_games(ply):
    """The cache concatenates games; ply resets to 0 at each start, so a non-+1 step is a boundary."""
    b = np.flatnonzero(np.diff(ply) != 1) + 1
    return np.split(np.arange(len(ply)), b)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="artifacts/experiments/basin_tent_1ply_data.npz")
    ap.add_argument("--max-ply", type=int, default=110)
    ap.add_argument("--beams", type=int, default=420, help="trajectories drawn per panel")
    ap.add_argument("--out-prefix", default="artifacts/experiments/basin_crt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    z = np.load(args.cache)
    rng = np.random.default_rng(args.seed)
    D = {k: (z[f"{k}_x"], z[f"{k}_p"]) for k in ("human", "SF-vs-SF")}

    plt.rcParams.update({"figure.facecolor": "#0b0b10", "axes.facecolor": "#0b0b10",
                         "text.color": "#e8e8ee", "axes.labelcolor": "#e8e8ee",
                         "xtick.color": "#9a9aa8", "ytick.color": "#9a9aa8",
                         "axes.edgecolor": "#3a3a48"})
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], hspace=0.28, wspace=0.03)

    for row, (name, (x, ply)) in enumerate(D.items()):
        segs = [s for s in split_games(ply) if len(s) >= 8]
        ax = fig.add_subplot(gs[row, 0]); sc = fig.add_subplot(gs[row, 1], sharey=None)
        # glow: the accumulated beam, additive-looking via a log density
        xs_all, ps_all = [], []
        for sgm in segs:
            v = parity_smooth(x[sgm]); v[0], v[-1] = x[sgm][0], x[sgm][-1]
            xs_all.append(v); ps_all.append(ply[sgm])
        xs_all = np.concatenate(xs_all); ps_all = np.concatenate(ps_all)
        m = ps_all <= args.max_ply
        H, xe, ye = np.histogram2d(ps_all[m], xs_all[m],
                                   bins=[np.arange(0, args.max_ply + 2, 2),
                                         np.linspace(-1, 1, 121)])
        ax.pcolormesh(xe, ye, np.ma.masked_less(H, 1).T, cmap="magma",
                      norm=LogNorm(vmin=1, vmax=max(H.max(), 10)), shading="flat", zorder=1)
        # individual beam paths
        pick = rng.choice(len(segs), min(args.beams, len(segs)), replace=False)
        landed = []
        for i in pick:
            s = segs[i]
            xs, ps = x[s], ply[s]
            k = ps <= args.max_ply
            if k.sum() < 6:
                continue
            sm = parity_smooth(xs[k])
            sm[0], sm[-1] = xs[k][0], xs[k][-1]              # keep true endpoints
            ax.plot(ps[k], sm, "-", color="#8fd8ff", lw=0.45, alpha=0.13, zorder=2)
            landed.append(xs[k][-1])
        landed = np.array(landed)
        # the gun
        ax.plot([0], [0], marker="o", ms=9, color="#ffe08a", zorder=5)
        ax.annotate("gun\n(start position)", (0, 0), textcoords="offset points", xytext=(16, 26),
                    color="#ffe08a", fontsize=8.5, ha="left")
        ax.axhline(0, color="#3a3a48", lw=.7, ls=":")
        ax.set_xlim(0, args.max_ply); ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("ply  (beam axis)"); ax.set_ylabel("deflection\nP(White wins) - P(Black wins)")
        ax.set_title(f"{name}", color="#e8e8ee", loc="left")
        # the phosphor screen: where the beam lands
        sc.hist(landed, bins=np.linspace(-1, 1, 61), orientation="horizontal",
                color="#8fd8ff", alpha=.85)
        sc.set_ylim(-1.05, 1.05); sc.set_yticks([]); sc.set_xticks([])
        sc.set_title("screen", fontsize=9, color="#9a9aa8")
        for yy, lab, cc in [(0.93, "White wins", "#7CFF9E"), (0.0, "draw", "#c8c8d4"),
                            (-0.93, "Black wins", "#FF8A8A")]:
            sc.text(sc.get_xlim()[1]*0.98, yy, lab, fontsize=8, color=cc, ha="right", va="center")
        print(f"  {name}: {len(pick)} beams, {len(landed)} landed | "
              f"|deflection| at screen: median {np.median(np.abs(landed)):.3f}", flush=True)
    fig.suptitle("The beam: fired from the start position, deflected along the ply axis, "
                 "splitting three ways onto the screen", color="#e8e8ee", fontsize=13)
    fig.savefig(f"{args.out_prefix}.png", dpi=150, facecolor="#0b0b10", bbox_inches="tight")
    print(f"wrote {args.out_prefix}.png [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
