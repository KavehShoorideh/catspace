#!/usr/bin/env python
"""basin_trilateration.py -- Kaveh 2026-08-04: embed by DISTANCE TO THE POLES, nothing else.

"push from start according to ply, and pull to each pole according to distance to pole."

Each position's 2-D location is determined solely by its four quasimetric pole distances: place the
four poles at fixed anchor positions, then solve for the point whose plane distances best match
d(s -> P_k). That is trilateration / landmark embedding -- every point solves INDEPENDENTLY, in
closed-ish form, with no neighbour graph, no SGD over the whole set, and nothing that can diverge.

This is why the earlier kNN layouts kept failing: they were using neighbour structure to recover a
geometry the pole distances already specify exactly. Worse, the poles are FARTHER from positions
than positions are from each other (symmetrised d(pos->pole) ~ 116 vs median pos-pos 91), so in a
kNN graph the anchors get almost no edges and drift to the periphery together.

Distances enter as log1p, the same compression every distance term in the objective uses, so the
plane is in the field's own units rather than raw interval length.

The fit is OVERDETERMINED (4 anchors, 2 unknowns), so the residual is meaningful: it says how much
of the four-distance structure a plane can actually hold. It is reported, not hidden -- a point
with a large residual is one the 2-D picture is lying about.
"""
from __future__ import annotations
import argparse, time
import numpy as np, torch

from catspace.research.components.encoder.approaches.reachability_field.src.iqe_head import IQEHead
from catspace.research.tools.training_infra.losses import WIN, DRAW, LOSS
from catspace.research.tools.embeddings.basin_simplex_chart import (
    COLOR_WIN, COLOR_DRAW, COLOR_LOSS, INK, MUTED)

START = 3
PNAME = {WIN: "mover wins", DRAW: "draw", LOSS: "mover loses", START: "START"}
PCOL = {WIN: COLOR_WIN, DRAW: COLOR_DRAW, LOSS: COLOR_LOSS, START: "#7b5cd6"}


def anchors(r_out=1.0):
    """START at the origin (games are pushed away from it), the three outcomes on a circle."""
    A = np.zeros((4, 2))
    for i, k in enumerate((WIN, DRAW, LOSS)):
        th = np.pi / 2 + i * 2 * np.pi / 3
        A[k] = [r_out * np.cos(th), r_out * np.sin(th)]
    A[START] = [0.0, 0.0]
    return A


def trilaterate(R, A, iters=60, lr=0.6, seed=0):
    """R (N,4) target plane distances, A (4,2) anchors -> Y (N,2), vectorized Gauss-Newton-ish.

    Minimises sum_k (||y - A_k|| - R_k)^2 per point. Started from the weighted anchor centroid,
    which is already close, so a few damped steps converge."""
    rng = np.random.default_rng(seed)
    w = 1.0 / np.maximum(R, 1e-6)
    Y = (w[:, :, None] * A[None]).sum(1) / w.sum(1)[:, None]
    Y = Y + rng.normal(0, 1e-3, Y.shape)                    # break exact ties at the centroid
    for _ in range(iters):
        d = Y[:, None, :] - A[None, :, :]                   # (N,4,2)
        n = np.linalg.norm(d, axis=2)                       # (N,4)
        u = d / np.maximum(n, 1e-9)[:, :, None]
        resid = (n - R)[:, :, None] * u                     # gradient of 0.5*sum resid^2
        Y = Y - lr * resid.mean(1)
    d = np.linalg.norm(Y[:, None, :] - A[None], axis=2)
    return Y, np.sqrt(((d - R) ** 2).mean(1))               # (N,) rms distance error


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/movie4/iqe_4pole_30k_latest.pt")
    ap.add_argument("--combined", default="data/derived/field_combined_sub600k.npz")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--out-prefix", default="artifacts/experiments/basin_trilat")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    t0 = time.time()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    p = torch.load(args.ckpt, map_location=args.device, weights_only=False); cfg = p["cfg"]
    net = IQEHead(in_ch=cfg["in_ch"], d=cfg["d"], components=cfg["components"],
                  adapter_ch=cfg["adapter_ch"]).to(args.device)
    net.load_compat(p["state_dict"]); net.eval()

    z = np.load(args.combined, allow_pickle=True)
    meta = eval(str(z["_meta"][0])); mm = np.load(meta["feats"][0], mmap_mode="r")
    split = z["orig_source"] if "orig_source" in z.files else z["source"]
    rng = np.random.default_rng(0)
    take = np.sort(np.concatenate([rng.choice(np.flatnonzero(split == s), args.n, replace=False)
                                   for s in (0, 1)]))
    y_lab, source = z["y"][take], split[take]

    with torch.no_grad():
        Dl = []
        for i in range(0, len(take), 8192):
            x = torch.from_numpy(np.asarray(mm[z["local_row"][take[i:i + 8192]]],
                                            dtype=np.float32)).to(args.device)
            e = net.phi(x)
            Dl.append(torch.cat([net.d_poles(e), net.d_from_start(e)[:, None]], 1).cpu().numpy())
    Dq = np.concatenate(Dl)
    R = np.log1p(np.maximum(Dq, 0))
    R = R / np.median(R[:, START])                          # unit: a typical distance from START
    A = anchors(r_out=float(np.median(R[:, :3])))
    Y, err = trilaterate(R, A)
    print(f"trilaterated {len(Y):,} points [{time.time()-t0:.0f}s] | rms distance error: "
          f"median {np.median(err):.3f}, p90 {np.percentile(err,90):.3f} "
          f"(anchor radius {np.linalg.norm(A[WIN]):.2f})")

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
    cols = np.array([COLOR_WIN, COLOR_DRAW, COLOR_LOSS])[y_lab]
    for ax, (nm, m) in zip(axes[:2], [("human", source == 0), ("SF-vs-SF", source == 1)]):
        o = rng.permutation(int(m.sum()))
        ax.scatter(Y[m][o, 0], Y[m][o, 1], s=3, c=cols[m][o], alpha=.30, linewidths=0)
        ax.set_title(f"{nm}  (n={int(m.sum()):,})", color=INK)
    h = axes[2].hexbin(Y[:, 0], Y[:, 1], C=err, reduce_C_function=np.median,
                       gridsize=60, cmap="viridis")
    fig.colorbar(h, ax=axes[2], shrink=.8, label="rms distance error (2-D fit quality)")
    axes[2].set_title("where the plane FAILS to hold the 4 distances", color=INK)
    for ax in axes:
        for k in (WIN, DRAW, LOSS, START):
            ax.plot(*A[k], "o", ms=13, color=PCOL[k], mec="black", mew=1.2, zorder=6)
            ax.annotate(PNAME[k], A[k], textcoords="offset points", xytext=(0, 15),
                        ha="center", fontsize=9, color=INK, zorder=7)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Trilateration: each position placed ONLY by its four pole distances\n"
                 "pushed from START by ply, pulled to each outcome pole by its distance -- "
                 "no neighbour graph, every point solved independently")
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}.png", dpi=140)
    for k in (WIN, DRAW, LOSS, START):
        print(f"  median log1p d(pos -> {PNAME[k]:>12s}) = {np.median(R[:, k]):.3f}")
    np.savez(f"{args.out_prefix}.npz", Y=Y, err=err, y=y_lab, source=source, A=A)
    print(f"wrote {args.out_prefix}.png [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
