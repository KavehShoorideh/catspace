#!/usr/bin/env python
"""basin_potential.py -- Kaveh 2026-08-04: trajectories as vectors -> summed flow field ->
scalar POTENTIAL -> human-minus-Stockfish difference.

Why a potential rather than differencing arrows. The hazard map differenced raw drift vectors,
which is noisy and hard to read: a vector field has no natural "height", so you cannot see where
a population is being held versus where it is being pushed through. Integrating the drift gives a
landscape, and a landscape has wells, barriers and slopes you can point at.

The decomposition is 1-D per ply row, NOT a 2-D Helmholtz-Hodge, and that is deliberate. The
motion is (dx, +1): the ply component is identically 1 for every step, so there is no genuine 2-D
circulation to decompose -- a Hodge split would be degenerate, with the entire "curl" being
d(drift)/d(ply). What the data actually supports is the classic effective potential of a 1-D
stochastic process, per ply band:

    drift-based    phi_v(x)  = -integral v(x) dx           (v = mean per-ply displacement)
    density-based  phi_p(x)  = -log P(x)                   (the observed landscape)

These are INDEPENDENT estimates of the same object -- one from how trajectories move, one from
where they accumulate. For a stationary 1-D diffusion they agree up to the diffusion scaling
(phi_v ~ phi_p when D is roughly constant). Reporting both and their correlation is the honest
check on whether a landscape picture is meaningful here at all; if they disagree, the flow is not
gradient-like and the potential should not be trusted.

The output is phi_human - phi_SF: where the two populations' landscapes differ. A well in the
difference means humans are held somewhere engines are not.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def split_games(ply):
    """Cache concatenates games without separators, but ply resets to 0 at each game start, so a
    non-+1 step marks a boundary. Returns a mask of valid within-game consecutive steps."""
    return np.diff(ply) == 1


def drift_and_density(x, ply, gx, band, min_count):
    """-> (centres, drift v(x), density P(x), n) for one ply band."""
    m = (ply >= band[0]) & (ply < band[1])
    xin = x[m]
    dens, _ = np.histogram(xin, bins=gx, density=False)
    ok = split_games(ply) & (ply[:-1] >= band[0]) & (ply[:-1] < band[1])
    x0, dx = x[:-1][ok], np.diff(x)[ok]
    n, _ = np.histogram(x0, bins=gx)
    s, _ = np.histogram(x0, bins=gx, weights=dx)
    s2, _ = np.histogram(x0, bins=gx, weights=dx ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(n >= min_count, s / np.maximum(n, 1), np.nan)
        var = np.where(n >= min_count, s2 / np.maximum(n, 1) - v ** 2, np.nan)
    Dc = np.maximum(var, 1e-6) / 2.0                        # diffusion coefficient, D = var/2
    P = dens / max(dens.sum(), 1)
    return v, P, n, Dc


def potentials(v, P, gx, Dc, eps=1e-6):
    """phi_v from the drift and phi_p from the density (-log P), both up to an additive constant.

    phi_v = -integral (v / D) dx, NOT -integral v dx. For a 1-D diffusion the stationary density
    is P ~ exp(-phi) with v = -D dphi/dx, so the drift must be divided by the diffusion
    coefficient before integrating. Omitting 1/D suppresses phi_v by a factor ~1/D (here D ~ 0.3,
    so ~3x) and makes the two estimates look like they disagree when they do not -- which is
    exactly what the first version of this plot showed."""
    w = np.diff(gx)
    vv = np.nan_to_num(v / np.maximum(Dc, 1e-6), nan=0.0)
    phi_v = -np.cumsum(vv * w)
    phi_p = -np.log(np.maximum(P, eps))
    phi_v = phi_v - np.nanmin(phi_v)
    phi_p = phi_p - np.nanmin(phi_p)
    return phi_v, phi_p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="artifacts/experiments/basin_tent_1ply_data.npz")
    ap.add_argument("--nx", type=int, default=41)
    ap.add_argument("--min-count", type=int, default=30)
    ap.add_argument("--edge", type=float, default=0.90,
                    help="exclude |x| beyond this from the reported difference: the wall forces "
                         "drift inward and density to pile, so the extremum always lands there")
    ap.add_argument("--corr-gate", type=float, default=0.45,
                    help="below this, the drift- and density-derived potentials disagree and the "
                         "landscape picture is NOT trustworthy -- the band is reported as such "
                         "rather than quoted")
    ap.add_argument("--out-prefix", default="artifacts/experiments/basin_potential")
    args = ap.parse_args()
    t0 = time.time()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(args.cache)
    D = {k: (z[f"{k}_x"], z[f"{k}_p"]) for k in ("human", "SF-vs-SF")}
    gx = np.linspace(-1, 1, args.nx + 1)
    cx = 0.5 * (gx[:-1] + gx[1:])
    bands = [(0, 20), (20, 40), (40, 60), (60, 90), (90, 130)]

    fig, axes = plt.subplots(2, len(bands), figsize=(4 * len(bands), 8), sharex=True)
    print(f"{'band':>10s} {'corr(phi_v,phi_p)':>18s} {'max |dphi|':>11s} {'x at max':>9s}  reading")
    rows = []
    for bi, band in enumerate(bands):
        out = {}
        for name, (x, p) in D.items():
            v, P, n, Dc = drift_and_density(x, p, gx, band, args.min_count)
            out[name] = potentials(v, P, gx, Dc) + (v, P, n)
        (hv, hp, hvv, hP, hn) = out["human"]
        (sv, sp, svv, sP, sn) = out["SF-vs-SF"]
        good = np.isfinite(hv) & np.isfinite(hp) & np.isfinite(sv) & np.isfinite(sp)
        corr = np.corrcoef(np.r_[hv[good], sv[good]], np.r_[hp[good], sp[good]])[0, 1]
        d = (hp - sp)                                  # density-based difference (better sampled)
        interior = np.abs(cx) <= args.edge
        di = np.where(interior, d, np.nan)
        j = int(np.nanargmax(np.abs(di)))
        rows.append((band, corr, float(di[j]), float(cx[j])))
        if corr < args.corr_gate:
            reading = "NOT TRUSTWORTHY (drift and density potentials disagree)"
        else:
            reading = ("deeper well for humans" if di[j] < 0 else "deeper well for engines")
        print(f"  {band[0]:>3d}-{band[1]:<4d} {corr:>18.3f} {abs(di[j]):>11.2f} {cx[j]:>+9.2f}  {reading}")

        ax = axes[0, bi]
        ax.plot(cx, hp, "-", color="#2a78d6", lw=2, label="human")
        ax.plot(cx, sp, "-", color="#e34948", lw=2, label="SF-vs-SF")
        ax.plot(cx, hv, "--", color="#2a78d6", lw=1, alpha=.7, label="human (from drift)")
        ax.plot(cx, sv, "--", color="#e34948", lw=1, alpha=.7, label="SF (from drift)")
        ax.set_title(f"ply {band[0]}-{band[1]}"); ax.set_ylim(0, 12)
        if bi == 0:
            ax.set_ylabel("potential  (-log P, arb. units)"); ax.legend(fontsize=7, frameon=False)
        ax2 = axes[1, bi]
        ax2.axhline(0, color="#8a8985", lw=.8)
        ax2.fill_between(cx, d, 0, where=d < 0, color="#2a78d6", alpha=.45, lw=0)
        ax2.fill_between(cx, d, 0, where=d > 0, color="#e34948", alpha=.45, lw=0)
        ax2.set_xlabel("P(White wins) - P(Black wins)")
        for a_ in (ax, ax2):
            a_.axvspan(-1.02, -args.edge, color="#8a8985", alpha=.16, lw=0)
            a_.axvspan(args.edge, 1.02, color="#8a8985", alpha=.16, lw=0)
        if corr < args.corr_gate:
            ax.text(0, 11, "potential NOT valid here\n(absorption breaks equilibrium)",
                    ha="center", va="top", fontsize=7.5, color="#d03b3b")
        if bi == 0:
            ax2.set_ylabel("phi(human) - phi(SF)\nblue = deeper well for humans")
    fig.suptitle("Trajectory flow -> scalar potential -> human minus Stockfish\n"
                 "solid = -log P (density), dashed = -integral of the drift; agreement between the "
                 "two is the check that a landscape picture is meaningful")
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}.png", dpi=140)
    print(f"\nmean corr(phi_from_drift, phi_from_density) = "
          f"{np.mean([r[1] for r in rows]):.3f}   (low => flow is not gradient-like, distrust the potential)")
    print(f"wrote {args.out_prefix}.png [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
