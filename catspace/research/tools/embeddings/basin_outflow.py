#!/usr/bin/env python
"""basin_outflow.py -- OUTWARD flow in the deflection-vs-ply frame, and human minus SF.

Now that the coordinate system exists, the flow is well posed. The key quantity is not raw drift
but the OUTWARD component -- whether a game is moving away from the centre (committing to a result)
or back toward it (returning to balance):

    outward(cell) = mean over steps of  sign(x) * dx/dply

positive = deflection growing in the direction it already leans (committing), negative = returning
to balance. This is signed the same way on both halves, so the two wings can be averaged together
instead of cancelling, which is what a raw mean drift does.

Three fixes over the earlier hazard map, each of which changed a conclusion once already:
  * FULL-GAME replays, so no 6-ply sampler comb;
  * PARITY-SMOOTHED deflection before differencing -- the field is turn-dependent (lag-1
    autocorrelation 0.42 vs lag-2 0.63) and the raw per-ply jitter is ~2.5x the mean drift, which
    previously manufactured a spurious "humans diverge from balance" signal;
  * cells compared only where BOTH populations have enough steps, with counts reported.
"""
from __future__ import annotations
import argparse, time
import numpy as np


def parity_smooth(v, w=2):
    return np.convolve(v, np.ones(w) / w, mode="same") if len(v) >= w else v


def split_games(ply):
    return np.split(np.arange(len(ply)), np.flatnonzero(np.diff(ply) != 1) + 1)


def outflow(x, ply, gx, gy, horizon, min_count):
    """-> (outward mean, raw dx mean, count) per (ply, deflection) cell."""
    X0, P0, DX = [], [], []
    for s in split_games(ply):
        if len(s) < horizon + 2:
            continue
        v = parity_smooth(x[s]); v[0], v[-1] = x[s][0], x[s][-1]
        h = horizon
        X0.append(v[:-h]); P0.append(ply[s][:-h]); DX.append((v[h:] - v[:-h]) / h)
    X0, P0, DX = np.concatenate(X0), np.concatenate(P0), np.concatenate(DX)
    out = np.sign(X0) * DX                                   # +ve = moving away from centre
    n, _, _ = np.histogram2d(P0, X0, bins=[gy, gx])
    so, _, _ = np.histogram2d(P0, X0, bins=[gy, gx], weights=out)
    sd, _, _ = np.histogram2d(P0, X0, bins=[gy, gx], weights=DX)
    with np.errstate(invalid="ignore", divide="ignore"):
        mo = np.where(n >= min_count, so / np.maximum(n, 1), np.nan)
        md = np.where(n >= min_count, sd / np.maximum(n, 1), np.nan)
    return mo, md, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="artifacts/experiments/basin_tent_1ply_data.npz")
    ap.add_argument("--max-ply", type=int, default=100)
    ap.add_argument("--nx", type=int, default=25)
    ap.add_argument("--ply-bin", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=40)
    ap.add_argument("--edge", type=float, default=0.90)
    ap.add_argument("--out-prefix", default="artifacts/experiments/basin_outflow")
    args = ap.parse_args()
    t0 = time.time()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    z = np.load(args.cache)
    gx = np.linspace(-1, 1, args.nx + 1)
    gy = np.arange(0, args.max_ply + args.ply_bin, args.ply_bin)
    F = {k: outflow(z[f"{k}_x"], z[f"{k}_p"], gx, gy, args.horizon, args.min_count)
         for k in ("human", "SF-vs-SF")}
    (ho, hd, hc), (so, sd, sc) = F["human"], F["SF-vs-SF"]
    cx = 0.5 * (gx[:-1] + gx[1:]); cy = 0.5 * (gy[:-1] + gy[1:])
    both = (hc >= args.min_count) & (sc >= args.min_count)
    interior = both & (np.abs(cx)[None, :] <= args.edge)
    diff = np.where(interior, ho - so, np.nan)

    vmax = np.nanmax(np.abs([np.nanmax(np.abs(ho)), np.nanmax(np.abs(so))]))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharey=True)
    for ax, (nm, M) in zip(axes[:2], [("human", ho), ("SF-vs-SF", so)]):
        pc = ax.pcolormesh(gx, gy, np.ma.masked_invalid(M), cmap="PuOr_r",
                           norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax), shading="flat")
        ax.set_title(f"{nm}: outward flow", color="#1c1b19")
        fig.colorbar(pc, ax=ax, shrink=.8, label="mean sign(x)*dx per ply")
    dm = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 1
    pc = axes[2].pcolormesh(gx, gy, np.ma.masked_invalid(diff), cmap="RdBu_r",
                            norm=TwoSlopeNorm(vcenter=0, vmin=-dm, vmax=dm), shading="flat")
    axes[2].set_title("human - SF   (red = humans commit faster)", color="#1c1b19")
    fig.colorbar(pc, ax=axes[2], shrink=.8, label="difference in outward flow")
    for ax in axes:
        ax.invert_yaxis(); ax.set_xlabel("deflection"); ax.axvline(0, color="#8a8985", lw=.7, ls=":")
        for e in (-args.edge, args.edge):
            ax.axvline(e, color="#8a8985", lw=.6, alpha=.5)
    axes[0].set_ylabel("ply")
    fig.suptitle("Outward flow: is a game committing to a result (positive) or returning to "
                 "balance (negative)?\nparity-smoothed, full-game replays, "
                 f"{args.horizon}-ply horizon; |deflection|>{args.edge} excluded (wall effect)")
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}.png", dpi=140)

    print(f"{'ply band':>10s} {'human out':>10s} {'SF out':>9s} {'diff':>8s}   reading")
    for a, b in [(0, 20), (20, 40), (40, 60), (60, 100)]:
        m = (cy >= a) & (cy < b)
        hh = np.nanmean(np.where(interior, ho, np.nan)[m]); ss = np.nanmean(np.where(interior, so, np.nan)[m])
        rd = "humans commit faster" if hh > ss else "engines commit faster"
        print(f"  {a:>3d}-{b:<5d} {hh:>10.4f} {ss:>9.4f} {hh-ss:>+8.4f}   {rd}")
    print(f"\noverall interior mean outward flow: human {np.nanmean(np.where(interior,ho,np.nan)):+.4f} | "
          f"SF {np.nanmean(np.where(interior,so,np.nan)):+.4f}")
    print(f"wrote {args.out_prefix}.png [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
