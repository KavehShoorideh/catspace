#!/usr/bin/env python
"""basin_crt.py -- deflection against ply, with the landing distribution (Kaveh 2026-08-04).

Games leave the START position at the left, run rightward along the PLY axis, are deflected
transversely by the field, and arrive at the right having separated three ways: White win (up),
draw (centre), Black win (down).

This is the SIDE view. The polar plot collapsed the ply axis, which is the axis the whole picture
is about. Same data, rotated, plus the landing distribution.

CELL SHAPE: bins are chosen so cells render roughly SQUARE. An earlier version used 2-ply x 0.0167
bins on a wide short panel, which made each cell 7.1x wider than tall -- the visible horizontal
rectangles. That was the bin aspect against the panel aspect, NOT the 6-ply sampler comb: this view
uses full-game replays, so every ply is present.

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
    ap.add_argument("--beams", type=int, default=420, help="individual game paths drawn per panel")
    ap.add_argument("--zoom-ply", type=int, default=20, help="upper ply for the zoomed panel")
    ap.add_argument("--replay-cap", type=int, default=120,
                    help="the ply cap the cached replays used. A path reaching it is a game STILL "
                         "IN PROGRESS, not a game that ended -- 45%% of SF games hit it (median SF "
                         "length is 127 plies). Conflating the two puts truncated paths in the "
                         "arrival distribution at whatever deflection they happened to be at.")
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
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, width_ratios=[4.6, 1.8, 0.9], hspace=0.30, wspace=0.10)

    def panel(ax, ps_all, xs_all, segs_pick, hi, ny):
        """Density + individual paths on [0, hi]. ny chosen so cells are ~square, see the note above."""
        m = ps_all <= hi
        H, xe, ye = np.histogram2d(ps_all[m], xs_all[m],
                                   bins=[np.arange(0, hi + 2, 1), np.linspace(-1, 1, ny + 1)])
        ax.pcolormesh(xe, ye, np.ma.masked_less(H, 1).T, cmap="magma",
                      norm=LogNorm(vmin=1, vmax=max(H.max(), 10)), shading="flat", zorder=1)
        for xs, ps in segs_pick:
            k = ps <= hi
            if k.sum() < 4:
                continue
            sm = parity_smooth(xs[k]); sm[0], sm[-1] = xs[k][0], xs[k][-1]
            ax.plot(ps[k], sm, "-", color="#8fd8ff", lw=0.45, alpha=0.13, zorder=2)
        ax.plot([0], [0], marker="o", ms=8, color="#ffe08a", zorder=5)
        ax.axhline(0, color="#3a3a48", lw=.7, ls=":")
        ax.set_xlim(0, hi); ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("ply")

    for row, (name, (x, ply)) in enumerate(D.items()):
        segs = [s_ for s_ in split_games(ply) if len(s_) >= 8]
        xs_all, ps_all = [], []
        for sgm in segs:
            v = parity_smooth(x[sgm]); v[0], v[-1] = x[sgm][0], x[sgm][-1]
            xs_all.append(v); ps_all.append(ply[sgm])
        xs_all = np.concatenate(xs_all); ps_all = np.concatenate(ps_all)
        pick = rng.choice(len(segs), min(args.beams, len(segs)), replace=False)
        picked = [(x[segs[i]], ply[segs[i]]) for i in pick]

        ax = fig.add_subplot(gs[row, 0])
        panel(ax, ps_all, xs_all, picked, args.max_ply, 42)
        ax.set_ylabel("deflection\nP(White wins) - P(Black wins)")
        ax.set_title(name, color="#e8e8ee", loc="left")
        ax.annotate("start position", (0, 0), textcoords="offset points", xytext=(14, 24),
                    color="#ffe08a", fontsize=8.5, ha="left")

        azm = fig.add_subplot(gs[row, 1])
        panel(azm, ps_all, xs_all, picked, args.zoom_ply, 90)
        azm.set_yticklabels([])
        azm.set_title(f"zoom: plies 0-{args.zoom_ply}", color="#9a9aa8", fontsize=10, loc="left")

        ended, cut = [], []
        for xx, pp in picked:
            k = pp <= args.max_ply
            if k.sum() < 4:
                continue
            (cut if pp[-1] > args.max_ply else ended).append(xx[k][-1])
        ended, cut = np.array(ended), np.array(cut)
        sc = fig.add_subplot(gs[row, 2])
        bins = np.linspace(-1, 1, 61)
        sc.hist(ended, bins=bins, orientation="horizontal", color="#8fd8ff", alpha=.9,
                label=f"ended ({len(ended)})")
        sc.hist(cut, bins=bins, orientation="horizontal", color="#6b6b7a", alpha=.7,
                label=f"still going ({len(cut)})")
        sc.legend(fontsize=6.5, frameon=False, loc="lower right", labelcolor="#9a9aa8")
        sc.set_ylim(-1.05, 1.05); sc.set_yticks([]); sc.set_xticks([])
        sc.set_title("where they\narrive", fontsize=9, color="#9a9aa8")
        for yy, lab, cc in [(0.93, "White wins", "#7CFF9E"), (0.0, "draw", "#c8c8d4"),
                            (-0.93, "Black wins", "#FF8A8A")]:
            sc.text(sc.get_xlim()[1]*0.98, yy, lab, fontsize=8, color=cc, ha="right", va="center")
        print(f"  {name}: {len(ended)} ended / {len(cut)} still going | median |deflection| "
              f"at a GENUINE end {np.median(np.abs(ended)):.3f} | at ply {args.zoom_ply}: "
              f"{np.median(np.abs(xs_all[(ps_all >= args.zoom_ply-1) & (ps_all <= args.zoom_ply+1)])):.3f}",
              flush=True)

    fig.suptitle("Deflection against ply: games leave the start position and separate three ways\n"
                 "arrival panel separates games that ENDED from those still in progress at the cap",
                 color="#e8e8ee", fontsize=11.5)
    fig.savefig(f"{args.out_prefix}.png", dpi=150, facecolor="#0b0b10", bbox_inches="tight")
    print(f"wrote {args.out_prefix}.png [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
