#!/usr/bin/env python
"""experiments/basin_simplex_chart.py -- Kaveh 2026-08-03: chart the W/D/L basins of the
three-pole IQE field as a TRIANGLE, and the field's asymmetry as flow on that triangle.

Why a ternary plot and not UMAP. The barycentric coordinates (p_win, p_draw, p_loss) have
exactly 2 free dimensions, so the triangle is a FAITHFUL 2-d picture: no dimensionality
reduction, no symmetrization, no distortion. The previous basin charts (basin_umap_compare.py)
ran UMAP on phi with a EUCLIDEAN metric, which silently discarded the IQE quasimetric
altogether -- they could only ever show a symmetrized shadow of the field. Here the three poles
are the vertices, each pole's attractor field weakens outward, and the undetermined middle is
populated by genuinely ambiguous positions rather than emptied by construction.

Because both datasets are read through ONE field trained on both, human and SF-vs-SF land in the
SAME canonical coordinate system -- directly comparable, which two separately-trained fields
never were.

Figures:
  1. Ternary density per dataset (log color scale) -- where each population's basins sit.
  2. Log-ratio (ilr) density -- the standard compositional-data coordinates. The committor is
     DEGENERATE inside basins (nearly all mass at p~0/1), so the raw triangle squashes basin
     interiors into its corners; the log-ratio view is what resolves them, and is where "the
     attractor extends out and weakens" is actually visible. Same reason the enhanced-sampling
     literature uses a logarithm of the committor as its collective variable.
  3. Drift-vector field: mean per-ply displacement in barycentric coordinates. This is the
     ASYMMETRY made visible -- d(s->g) != d(g->s) has a direction, and this is that direction.
     The construction is the classical drift-vector model for asymmetric proximity data (fit
     position from the symmetric part, draw the skew part as arrows), which is defined only in
     two dimensions -- exactly our case.
  4. Ambiguity: distribution of max_k p_k per dataset. Kaveh's prediction that a lot of
     positions stay ambiguous even in near-perfect SF-vs-SF play is falsifiable, and this is
     the figure that tests it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.losses import basin_logp, WIN, DRAW, LOSS
from catspace.encoder.iqe_head import IQEHead

# Palette: same validated reference values the 2026-08-02 basin figures used, so the new
# figures sit alongside the old ones. Status good/critical for win/loss, neutral gray for draw.
COLOR_WIN, COLOR_DRAW, COLOR_LOSS = "#0ca30c", "#8a8985", "#d03b3b"
COLOR_HUMAN, COLOR_SF = "#2a78d6", "#e34948"
INK, MUTED = "#1c1b19", "#8a8985"

# Equilateral triangle: win at bottom-left, draw at bottom-right, loss at apex.
VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
VERT_NAMES = ["mover wins", "draw", "mover loses"]


def bary_to_xy(p):
    """(N,3) simplex weights -> (N,2) cartesian inside the triangle. Pure tensor op."""
    return p @ VERTS


def ilr(p, eps=1e-12):
    """(N,3) -> (N,2) isometric log-ratio coordinates (Aitchison). The standard, distortion-free
    log coordinates for compositional data; unlike the raw simplex it does not pile basin
    interiors into the corners."""
    q = np.clip(p, eps, None)
    return np.stack([np.log(q[:, 0] / q[:, 1]) / np.sqrt(2.0),
                     np.log(q[:, 0] * q[:, 1] / q[:, 2] ** 2) / np.sqrt(6.0)], axis=1)


def draw_triangle(ax):
    tri = np.vstack([VERTS, VERTS[:1]])
    ax.plot(tri[:, 0], tri[:, 1], "-", color=MUTED, lw=1.0)
    for (x, y), name, c in zip(VERTS, VERT_NAMES, [COLOR_WIN, COLOR_DRAW, COLOR_LOSS]):
        ax.plot([x], [y], "o", color=c, ms=7, zorder=5)
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(0, 10 if y > 0.1 else -14), ha="center", fontsize=9, color=INK)
    ax.set_aspect("equal"); ax.axis("off")


def load_head(ckpt, device):
    p = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = p["cfg"]
    net = IQEHead(in_ch=cfg["in_ch"], d=cfg["d"], components=cfg["components"],
                  adapter_ch=cfg["adapter_ch"]).to(device)
    missing, _ = net.load_compat(p["state_dict"])
    if any(m.startswith(("poles", "log_T")) for m in missing):
        raise SystemExit(f"{ckpt} has no trained W/D/L poles ({missing}) -- its basin readout "
                         f"would be untrained noise. Train with --combined first.")
    net.eval()
    return net


@torch.no_grad()
def probs_for(net, feats_path, local_rows, device, batch=8192):
    """Trunk features -> (N,3) basin probabilities. Reads the precomputed memmap directly (the
    trunk is never re-run), in sorted order for sequential memmap access."""
    mm = np.load(feats_path, mmap_mode="r")
    order = np.argsort(local_rows)
    out = np.empty((len(local_rows), 3), np.float32)
    for i in range(0, len(order), batch):
        sl = order[i:i + batch]
        x = torch.from_numpy(np.asarray(mm[local_rows[sl]], dtype=np.float32)).to(device)
        d = net.d_poles(net.phi(x))
        out[sl] = basin_logp(d, net.temperature).exp().cpu().numpy()
    return out


def binned_mean_vectors(xy, dxy, bins, rng_xy):
    """Vectorized drift field: mean displacement per cell via weighted 2-d histograms (no loop
    over points). Returns cell centers and mean dx, dy, and the per-cell count."""
    cnt, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=rng_xy)
    sx, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=rng_xy, weights=dxy[:, 0])
    sy, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=rng_xy, weights=dxy[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = sx / cnt, sy / cnt
    cx = 0.5 * (xe[:-1] + xe[1:]); cy = 0.5 * (ye[:-1] + ye[1:])
    return cx, cy, mx, my, cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/iqe_poles_both_latest.pt")
    ap.add_argument("--combined", default="data/derived/field_combined_v1.npz")
    ap.add_argument("--n", type=int, default=150000, help="positions sampled per dataset")
    ap.add_argument("--min-count", type=int, default=60, help="min points per drift cell to draw")
    ap.add_argument("--bins", type=int, default=22)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="artifacts/experiments/basin_simplex")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    t0 = time.time()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    net = load_head(args.ckpt, args.device)
    z = np.load(args.combined, allow_pickle=True)
    meta = eval(str(z["_meta"][0]))
    rng = np.random.default_rng(args.seed)

    # `orig_source` marks human vs SF-vs-SF; `source` indexes which FEATURE FILE to read, and is
    # all-zero once the subset has been materialized into one contiguous file. Splitting on the
    # wrong one silently merges the two populations into a single panel.
    split = z["orig_source"] if "orig_source" in z.files else z["source"]
    datasets = {}
    for name, src in [("human", 0), ("SF-vs-SF", 1)]:
        idx = np.flatnonzero(split == src)
        take = np.sort(rng.choice(idx, min(args.n, len(idx)), replace=False))
        p = probs_for(net, meta["feats"][int(z["source"][take[0]])],
                      z["local_row"][take], args.device)
        datasets[name] = dict(p=p, xy=bary_to_xy(p), game=z["game"][take], ply=z["ply"][take],
                              y=z["y"][take])
        print(f"  {name}: n={len(take):,} mean max-p {p.max(1).mean():.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: ternary density -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6))
    for ax, (name, d) in zip(axes, datasets.items()):
        hb = ax.hexbin(d["xy"][:, 0], d["xy"][:, 1], gridsize=48, bins="log",
                       cmap="Blues" if name == "human" else "Reds", mincnt=1, linewidths=0)
        draw_triangle(ax)
        ax.set_title(f"{name}  (n={len(d['p']):,})", color=INK)
        fig.colorbar(hb, ax=ax, shrink=0.72, label="positions per cell (log)")
    fig.suptitle("W/D/L basins on the probability simplex -- one field, both datasets, "
                 "mover-POV\ncorners = the three poles; centre = genuinely undetermined")
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}_1_ternary.png", dpi=140)

    # ---- Figure 2: log-ratio (ilr) density ------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, (name, d) in zip(axes2, datasets.items()):
        v = ilr(d["p"])
        hb = ax.hexbin(v[:, 0], v[:, 1], gridsize=55, bins="log", mincnt=1, linewidths=0,
                       cmap="Blues" if name == "human" else "Reds")
        ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_xlabel("log(win/draw) / √2"); ax.set_ylabel("log(win·draw/loss²) / √6")
        ax.set_title(f"{name} -- log-ratio coordinates", color=INK)
        fig2.colorbar(hb, ax=ax, shrink=0.72, label="positions per cell (log)")
    fig2.suptitle("Same basins in log-ratio coordinates -- resolves the basin INTERIORS that the "
                  "raw simplex squashes into its corners")
    fig2.tight_layout(); fig2.savefig(f"{args.out_prefix}_2_logratio.png", dpi=140)

    # ---- Figure 3: drift-vector field (the asymmetry) --------------------------------------
    # Per-ply drift from consecutive SAMPLED positions of the same game. The dataset stores a
    # per-game subsample, so consecutive rows are generally NOT consecutive plies -- the
    # displacement is divided by the actual ply gap to get a per-ply rate, and pairs spanning
    # more than --max-gap plies are dropped rather than silently averaged over a long interval.
    MAX_GAP = 12
    fig3, axes3 = plt.subplots(1, 2, figsize=(12.5, 6))
    for ax, (name, d) in zip(axes3, datasets.items()):
        g, ply, xy = d["game"], d["ply"], d["xy"]
        o = np.lexsort((ply, g))
        g, ply, xy = g[o], ply[o], xy[o]
        same = g[:-1] == g[1:]
        gap = (ply[1:] - ply[:-1]).astype(np.float64)
        ok = same & (gap > 0) & (gap <= MAX_GAP)
        src_xy = xy[:-1][ok]
        dxy = (xy[1:][ok] - src_xy) / gap[ok, None]
        cx, cy, mx, my, cnt = binned_mean_vectors(
            src_xy, dxy, args.bins, [[0, 1], [0, np.sqrt(3) / 2]])
        X, Y = np.meshgrid(cx, cy, indexing="ij")
        m = cnt >= args.min_count
        mag = np.hypot(mx, my)
        # Arrow length is AUTO-SCALED so the median drift spans ~0.6 of a cell. A fixed scale
        # made near-terminal cells (which move across most of the simplex in one ply) shoot far
        # outside the triangle and swamp everything else -- the first render was unreadable.
        cell = 1.0 / args.bins
        med = float(np.median(mag[m])) if m.any() else 1.0
        scale = max(med, 1e-9) / (0.6 * cell)
        # Long arrows are CLIPPED in length only (direction preserved) and the count is printed --
        # a silent cap would read as "nothing moves fast here".
        cap = float(np.quantile(mag[m], 0.85)) if m.any() else 1.0
        over = int((mag[m] > cap).sum())
        f = np.where(mag > cap, cap / np.maximum(mag, 1e-12), 1.0)
        ax.hexbin(xy[:, 0], xy[:, 1], gridsize=48, bins="log", mincnt=1, linewidths=0,
                  cmap="Greys", alpha=0.35)
        ax.quiver(X[m], Y[m], (mx * f)[m], (my * f)[m],
                  color=COLOR_HUMAN if name == "human" else COLOR_SF,
                  angles="xy", scale_units="xy", scale=scale, width=0.004)
        print(f"  drift {name}: {int(m.sum())} cells >= {args.min_count} pts | median |drift| "
              f"{med:.4f}/ply | {over} cells length-clipped at the 85th pct ({cap:.3f})")
        draw_triangle(ax)
        ax.set_title(f"{name} -- mean per-ply drift  ({int(m.sum())} cells)", color=INK)
    fig3.suptitle("The field's ASYMMETRY as flow: mean per-ply drift toward the basins\n"
                  "(drift-vector model -- arrows are the skew part, position the symmetric part)")
    fig3.tight_layout(); fig3.savefig(f"{args.out_prefix}_3_drift.png", dpi=140)

    # ---- Figure 4: ambiguity ---------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(7.5, 5))
    rows = []
    for name, d in datasets.items():
        conf = d["p"].max(1)
        ax4.hist(conf, bins=60, range=(1 / 3, 1), histtype="step", lw=2,
                 density=True, label=name, color=COLOR_HUMAN if name == "human" else COLOR_SF)
        rows.append((name, float(conf.mean()), float((conf < 0.5).mean()),
                     float((conf > 0.9).mean())))
    ax4.axvline(1 / 3, color=MUTED, lw=0.8)
    ax4.annotate("fully undetermined\n(centre of triangle)", (1 / 3, ax4.get_ylim()[1] * 0.85),
                 xytext=(6, 0), textcoords="offset points", fontsize=8, color=MUTED)
    ax4.set_xlabel("confidence  max_k p_k"); ax4.set_ylabel("density")
    ax4.set_title("How much of each population is genuinely ambiguous?", color=INK)
    ax4.legend(frameon=False)
    fig4.tight_layout(); fig4.savefig(f"{args.out_prefix}_4_ambiguity.png", dpi=140)

    print("\nAMBIGUITY (the headline number)")
    print(f"  {'dataset':10s} {'mean conf':>10s} {'frac <0.5':>10s} {'frac >0.9':>10s}")
    for name, mc, lo, hi in rows:
        print(f"  {name:10s} {mc:>10.3f} {lo:>10.3f} {hi:>10.3f}")
    np.savez(f"{args.out_prefix}_data.npz",
             **{f"{n}_{k}": v for n, d in datasets.items() for k, v in d.items()})
    print(f"wrote {args.out_prefix}_{{1_ternary,2_logratio,3_drift,4_ambiguity}}.png + _data.npz "
          f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
