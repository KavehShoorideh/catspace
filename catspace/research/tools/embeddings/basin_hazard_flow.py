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
from catspace.research.tools.embeddings.basin_tent_fullgames import sf_games, human_games, replay


def flow_field(x, ply, gx, gy, min_count):
    """Mean per-ply displacement vector per (x, ply) cell, plus the per-cell sample count.

    Vectorized via weighted 2-d histograms -- one pass per component, no loop over points."""
    x0, y0 = x[:-1], ply[:-1]
    dx, dy = np.diff(x), np.diff(ply)
    ok = dy == 1                                             # consecutive plies within one game
    x0, y0, dx, dy = x0[ok], y0[ok], dx[ok], dy[ok]
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
    args = ap.parse_args()
    t0 = time.time()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField
    field = ReachabilityField(onnx=args.onnx, head=args.ckpt)
    if not field.has_poles:
        raise SystemExit(f"{args.ckpt} has no trained poles")
    rng = np.random.default_rng(args.seed)

    # UNIFORM game sample: a hazard map must be population-representative (see module docstring).
    pools = {"human": human_games(args.human_records, 10 ** 9, rng),
             "SF-vs-SF": sf_games(args.sf_moves, 10 ** 9, rng)}
    data = {}
    for name, pool in pools.items():
        pick = rng.choice(len(pool), min(args.n_games, len(pool)), replace=False)
        X, P = [], []
        for j in pick:
            _, _, ucis, _ = pool[j]
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
        print(f"  {name}: {len(pick)} games, {len(data[name][0]):,} positions "
              f"[{time.time()-t0:.0f}s]", flush=True)

    gx = np.linspace(-1, 1, args.nx + 1)
    gy = np.arange(0, args.max_ply + args.ply_bin, args.ply_bin)
    F = {n: flow_field(x, p, gx, gy, args.min_count) for n, (x, p) in data.items()}
    (hx, hy, hc), (sx_, sy_, sc) = F["human"], F["SF-vs-SF"]

    # THE HAZARD: magnitude of the VECTOR difference, only where both populations are well sampled.
    both = (hc >= args.min_count) & (sc >= args.min_count)
    ddx, ddy = hx - sx_, hy - sy_
    hazard = np.where(both, np.hypot(ddx, ddy), np.nan)

    cx = 0.5 * (gx[:-1] + gx[1:]); cy = 0.5 * (gy[:-1] + gy[1:])
    CX, CY = np.meshgrid(cx, cy, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    for ax, (nm, (fx_, fy_, c)) in zip(axes[:2], F.items()):
        m = c >= args.min_count
        ax.quiver(CX[m], CY[m], fx_[m], fy_[m], color="#2a78d6" if nm == "human" else "#e34948",
                  angles="xy", scale_units="xy", scale=0.02, width=0.005)
        ax.set_title(f"{nm} flow  ({int(m.sum())} cells)", color=INK)
    div = LinearSegmentedColormap.from_list("hz", ["#f0efec", "#f7c948", "#d03b3b"])
    pc = axes[2].pcolormesh(gx, gy, np.ma.masked_invalid(hazard).T, cmap=div, shading="flat")
    q = np.nanquantile(hazard, 0.9) if np.isfinite(hazard).any() else np.nan
    hot = np.argwhere(np.nan_to_num(hazard, nan=-1) >= q)
    axes[2].quiver(CX[both], CY[both], ddx[both], ddy[both], color=INK,
                   angles="xy", scale_units="xy", scale=0.02, width=0.004, alpha=0.55)
    fig.colorbar(pc, ax=axes[2], shrink=0.8, label="|human flow - engine flow|  per ply")
    axes[2].set_title(f"HAZARD = |vector difference|  ({int(both.sum())} comparable cells)", color=INK)
    for ax in axes:
        ax.invert_yaxis(); ax.set_xlim(-1.02, 1.02); ax.set_ylim(args.max_ply, 0)
        ax.axvline(0, color=MUTED, lw=0.7, ls=":")
        ax.set_xlabel("P(White wins) - P(Black wins)")
    axes[0].set_ylabel("ply")
    fig.suptitle("Density-flow difference: where do humans move differently from engines?\n"
                 "hazard = magnitude of the VECTOR difference (direction matters, not just speed)")
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}_flow.png", dpi=140)

    print(f"\nTOP HAZARD POCKETS (>= 90th pct, both populations >= {args.min_count} plies/cell)")
    print(f"  {'x':>7s} {'ply':>6s} {'hazard':>8s} {'human drift':>22s} {'engine drift':>22s} "
          f"{'n_hu':>6s} {'n_sf':>6s}")
    order = sorted(hot.tolist(), key=lambda ij: -hazard[ij[0], ij[1]])[:12]
    for i, j in order:
        print(f"  {cx[i]:>+7.2f} {cy[j]:>6.0f} {hazard[i,j]:>8.4f} "
              f"({hx[i,j]:>+.4f},{hy[i,j]:>+.2f}) {'':>3s} ({sx_[i,j]:>+.4f},{sy_[i,j]:>+.2f}) "
              f"{int(hc[i,j]):>6d} {int(sc[i,j]):>6d}")
    np.savez(f"{args.out_prefix}_flow.npz", gx=gx, gy=gy, hazard=hazard,
             hx=hx, hy=hy, sx=sx_, sy=sy_, hc=hc, sc=sc)
    print(f"wrote {args.out_prefix}_flow.png / .npz [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
