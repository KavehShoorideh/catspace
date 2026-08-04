#!/usr/bin/env python
"""catspace/research/tools/embeddings/basin_hazard_flow.py -- Kaveh 2026-08-04: the HAZARD map.

Take the density flow (mean per-ply drift) separately for human and for SF-vs-SF over the same
grid, subtract them, and look for localised pockets where the two populations move DIFFERENTLY
from the same place. Those pockets are the hazards: regions where humans systematically drift
toward a worse basin while engines hold, i.e. where human play leaks and engine play does not.

Why a VECTOR difference and not a difference of speeds. "Humans move faster here" is far weaker
than "humans move LEFT here while engines hold steady" -- the direction is the whole content of a
hazard. So the field compared is the mean per-ply displacement vector per cell, and the hazard
score is the magnitude of the DIFFERENCE of those vectors, not the difference of magnitudes. A
cell where both populations drift equally fast in the same direction is not a hazard; a cell where
they drift equally fast in OPPOSITE directions is the strongest hazard there is, and only the
vector difference separates those two cases.

Grid: the tent's (x, ply) plane, x = P(White wins) - P(Black wins) in WHITE-POV -- the poles are
mover-POV and the mover alternates every ply, so mover-POV drift would alternate sign every step
and the flow field would be noise.

Both populations are replayed FULL-GAME (every ply) and sampled UNIFORMLY (not stratified by
result): a hazard map is a statement about where the populations actually go, so it must be
population-representative. Stratifying would manufacture engine drift that the real 81%-draw
population does not have.

Cells are only compared where BOTH populations have enough samples (--min-count each), and the
count is reported -- a "hazard" resting on 3 human plies and 2 engine plies is noise wearing a
hat.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from catspace.research.tools.training_infra.losses import basin_logp
from catspace.research.tools.embeddings.basin_tent import white_pov_x, COLOR_WHITE_WIN, COLOR_BLACK_WIN
from catspace.research.tools.embeddings.basin_simplex_chart import INK, MUTED
from catspace.research.tools.embeddings.basin_tent_fullgames import replay


def uniform_sf(tsv, n, rng):
    """n SF games sampled UNIFORMLY. Two passes over a 57MB TSV: count lines, pick indices, then
    keep only those. Never materializes 100k move lists, and never stratifies -- a hazard map must
    reflect the real population (81% draws), not a balanced one."""
    with open(tsv) as f:
        total = sum(1 for _ in f)
    keep = set(rng.choice(total, min(n, total), replace=False).tolist())
    out = []
    with open(tsv) as f:
        for i, line in enumerate(f):
            if i in keep:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3:
                    out.append((int(p[0]), int(p[1]), p[2].split(), ""))
    return out


def uniform_human(records_dir, n, rng):
    """n human games sampled UNIFORMLY across shards, same reasoning as uniform_sf."""
    import pyarrow.parquet as pq
    shards = sorted(Path(records_dir).glob("records_*.parquet"))
    counts = [pq.ParquetFile(s).metadata.num_rows for s in shards]
    total = sum(counts)
    pick = np.sort(rng.choice(total, min(n, total), replace=False))
    out, base = [], 0
    for sh, c in zip(shards, counts):
        want = pick[(pick >= base) & (pick < base + c)] - base
        if len(want):
            d = pq.read_table(sh, columns=["game_id", "result", "moves", "termination"]).to_pydict()
            for j in want:
                out.append((int(d["game_id"][j]), int(d["result"][j]),
                            d["moves"][j].split(), d["termination"][j]))
        base += c
    return out


def flow_field(x, ply, gx, gy, min_count, horizon=1, smooth=1):
    """horizon>1 / smooth>1 exist to defeat a SELECTION-ON-NOISE artifact, not for looks.

    The field is not temporally smooth (lag-1 autocorrelation 0.42 vs lag-2 0.63) and median
    ply-to-ply jitter is ~0.2, while the near-zero x bin is only ~0.077 wide. So a position can
    land in that bin by jitter alone, and the next ply reverts toward its true value -- which
    fabricates drift pointing AWAY from zero. Binning on a smoothed x and measuring displacement
    over `horizon` plies makes the signal exceed the jitter; if a pattern survives both, it is not
    the artifact."""
    """Mean per-ply displacement vector per (x, ply) cell, plus the per-cell sample count.

    Vectorized via weighted 2-d histograms -- one pass per component, no loop over points."""
    xs = x
    if smooth > 1:                                           # bin on a de-jittered x
        k = np.ones(smooth) / smooth
        xs = np.convolve(np.nan_to_num(x, nan=0.0), k, mode="same")
        xs[~np.isfinite(x)] = np.nan
    h = horizon
    x0, y0 = xs[:-h], ply[:-h]
    dx = (x[h:] - x[:-h]) / h                                # per-ply rate over the horizon
    dy = ply[h:] - ply[:-h]
    ok = (dy == h) & np.isfinite(x0) & np.isfinite(dx)       # same game, no separator
    x0, y0, dx = x0[ok], y0[ok], dx[ok]
    dy = dy[ok]
    rng = [[gx[0], gx[-1]], [gy[0], gy[-1]]]
    cnt, _, _ = np.histogram2d(x0, y0, bins=[gx, gy])
    sx, _, _ = np.histogram2d(x0, y0, bins=[gx, gy], weights=dx)
    sy, _, _ = np.histogram2d(x0, y0, bins=[gx, gy], weights=dy.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = sx / cnt, sy / cnt
    mask = cnt >= min_count
    return np.where(mask, mx, np.nan), np.where(mask, my, np.nan), cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/movie4/iqe_4pole_30k_latest.pt")
    ap.add_argument("--onnx", default="assets/engines/lc0/t1-256x10.onnx")
    ap.add_argument("--sf-moves", default="data/derived/opening_pool_sfsf_moves.tsv")
    ap.add_argument("--human-records", default="data/records/lichess_2019-01")
    ap.add_argument("--n-games", type=int, default=700, help="games per source, UNIFORM sample")
    ap.add_argument("--max-ply", type=int, default=120)
    ap.add_argument("--nx", type=int, default=26)
    ap.add_argument("--ply-bin", type=int, default=6)
    ap.add_argument("--min-count", type=int, default=40, help="per cell, per population")
    ap.add_argument("--out-prefix", default="artifacts/experiments/basin_hazard")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=4,
                    help="measure displacement over this many plies (per-ply rate). >1 defeats "
                         "selection-on-noise: jitter is ~0.2 vs a 0.077-wide bin.")
    ap.add_argument("--smooth", type=int, default=2,
                    help="ply-smoothing applied to x for BINNING only (2 cancels ply parity)")
    ap.add_argument("--replot", default="", help="re-plot from a saved *_flow.npz, skipping replay")
    ap.add_argument("--edge", type=float, default=0.90,
                    help="|x| beyond this is a BOUNDARY cell: drift there can only point inward, "
                         "so magnitude is inflated by the wall rather than by real divergence. "
                         "Excluded from the ranked pockets and hatched on the map.")
    args = ap.parse_args()
    t0 = time.time()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if args.replot:
        _z = np.load(args.replot)
        gx, gy = _z["gx"], _z["gy"]
        hx, hy, hc = _z["hx"], _z["hy"], _z["hc"]
        sx_, sy_, sc = _z["sx"], _z["sy"], _z["sc"]
        _render(args, plt, LinearSegmentedColormap, gx, gy, hx, hy, hc, sx_, sy_, sc, t0)
        return

    from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField
    field = ReachabilityField(onnx=args.onnx, head=args.ckpt)
    if not field.has_poles:
        raise SystemExit(f"{args.ckpt} has no trained poles")
    rng = np.random.default_rng(args.seed)

    # UNIFORM game sample: a hazard map must be population-representative (see module docstring).
    pools = {"human": uniform_human(args.human_records, args.n_games, rng),
             "SF-vs-SF": uniform_sf(args.sf_moves, args.n_games, rng)}
    data = {}
    for name, pool in pools.items():
        X, P = [], []
        for _, _, ucis, _ in pool:
            planes, _, _ = replay(ucis, args.max_ply)
            if planes is None or len(planes) < 4:
                continue
            with torch.no_grad():
                ps = []
                for i in range(0, len(planes), 4096):
                    phi = field.phi_from_planes(list(planes[i:i + 4096].astype(np.float32)))
                    ps.append(basin_logp(field.head.d_poles(phi),
                                         field.head.temperature).exp().cpu().numpy())
            pr = np.concatenate(ps)
            ply = np.arange(len(pr))
            X.append(white_pov_x(pr, ply)); P.append(ply)
            X.append(np.array([np.nan])); P.append(np.array([-999]))   # game separator
        data[name] = (np.concatenate(X), np.concatenate(P))
        print(f"  {name}: {len(pool)} games, {len(data[name][0]):,} positions "
              f"[{time.time()-t0:.0f}s]", flush=True)

    gx = np.linspace(-1, 1, args.nx + 1)
    gy = np.arange(0, args.max_ply + args.ply_bin, args.ply_bin)
    F = {n: flow_field(x, p, gx, gy, args.min_count, args.horizon, args.smooth)
         for n, (x, p) in data.items()}
    (hx, hy, hc), (sx_, sy_, sc) = F["human"], F["SF-vs-SF"]
    np.savez(f"{args.out_prefix}_flow.npz", gx=gx, gy=gy, hx=hx, hy=hy, sx=sx_, sy=sy_,
             hc=hc, sc=sc)
    _render(args, plt, LinearSegmentedColormap, gx, gy, hx, hy, hc, sx_, sy_, sc, t0)


def _render(args, plt, LSC, gx, gy, hx, hy, hc, sx_, sy_, sc, t0):
    """Plot ONLY the horizontal drift. dy is identically +1 (ply always advances by one), so it
    carries no information and, drawn as a quiver component, swamps every arrow with a constant
    vertical streak -- which is exactly what the first render did. It also means the 'vector
    difference' reduces to the difference in horizontal drift, since dy cancels."""
    both = (hc >= args.min_count) & (sc >= args.min_count)
    ddx = hx - sx_
    hazard = np.where(both, np.abs(ddx), np.nan)
    cx = 0.5 * (gx[:-1] + gx[1:]); cy = 0.5 * (gy[:-1] + gy[1:])
    CX, CY = np.meshgrid(cx, cy, indexing="ij")
    edge = np.abs(CX) > args.edge                      # boundary cells: drift can only point inward
    interior = both & ~edge

    scale = max(np.nanmax(np.abs(hazard)) if np.isfinite(hazard).any() else 1.0, 1e-6) / 0.16
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    for ax, (nm, fxv, c, col) in zip(axes[:2],
                                     [("human", hx, hc, "#2a78d6"), ("SF-vs-SF", sx_, sc, "#e34948")]):
        m = c >= args.min_count
        ax.quiver(CX[m], CY[m], fxv[m], np.zeros(m.sum()), color=col,
                  angles="xy", scale_units="xy", scale=scale, width=0.005)
        ax.set_title(f"{nm} drift per ply  ({int(m.sum())} cells)", color=INK)
    div = LSC.from_list("hz", ["#f0efec", "#f7c948", "#d03b3b"])
    pc = axes[2].pcolormesh(gx, gy, np.ma.masked_invalid(np.where(edge, np.nan, hazard)).T,
                            cmap=div, shading="flat")
    axes[2].quiver(CX[interior], CY[interior], ddx[interior], np.zeros(interior.sum()),
                   color=INK, angles="xy", scale_units="xy", scale=scale, width=0.004, alpha=0.6)
    for i in range(len(cx)):
        if abs(cx[i]) > args.edge:
            axes[2].axvspan(gx[i], gx[i + 1], color="#8a8985", alpha=0.18, lw=0)
    fig.colorbar(pc, ax=axes[2], shrink=0.8, label="|human drift - engine drift|  per ply")
    axes[2].set_title(f"HAZARD  ({int(interior.sum())} interior cells; grey = boundary, excluded)",
                      color=INK)
    for ax in axes:
        ax.invert_yaxis(); ax.set_xlim(-1.02, 1.02); ax.set_ylim(args.max_ply, 0)
        ax.axvline(0, color=MUTED, lw=0.7, ls=":")
        ax.set_xlabel("P(White wins) - P(Black wins)")
    axes[0].set_ylabel("ply")
    fig.suptitle("Where do humans drift differently from engines? (horizontal drift per ply)\n"
                 "grey bands = |x|>%.2f, where the wall forces drift inward and inflates it" % args.edge)
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}_flow.png", dpi=140)

    hz = np.where(interior, hazard, np.nan)
    q = np.nanquantile(hz, 0.90) if np.isfinite(hz).any() else np.nan
    print(f"\nTOP HAZARD POCKETS -- INTERIOR ONLY (|x| <= {args.edge}), both pops >= {args.min_count}")
    print(f"  {'x':>7s} {'ply':>5s} {'hazard':>8s} {'human':>9s} {'engine':>9s} {'n_hu':>6s} {'n_sf':>6s}")
    hot = sorted(np.argwhere(np.nan_to_num(hz, nan=-1) >= q).tolist(),
                 key=lambda ij: -hz[ij[0], ij[1]])[:10]
    for i, j in hot:
        print(f"  {cx[i]:>+7.2f} {cy[j]:>5.0f} {hz[i,j]:>8.4f} {hx[i,j]:>+9.4f} {sx_[i,j]:>+9.4f} "
              f"{int(hc[i,j]):>6d} {int(sc[i,j]):>6d}")
    print(f"wrote {args.out_prefix}_flow.png [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
