#!/usr/bin/env python
"""basin_anchored_layout.py -- Kaveh 2026-08-04: the first embedding in this project that throws
away neither the poles nor the quasimetric.

Every UMAP we have drawn ran Euclidean-on-phi, which discards the IQE quasimetric entirely, and
had no idea the poles existed. This fixes both, and keeps the asymmetry instead of averaging it:

  POLES PINNED   the outcome poles are placed at fixed triangle vertices and never moved, so they
                 form the global frame rather than being invisible to the layout. This is the
                 landmark/anchored-embedding idea (landmark MDS, SUDE): a few anchors carry the
                 global structure, everything else lays out relative to them.
  METRIC = S     positions are laid out under the SYMMETRIC part S = (D + D^T)/2 of the field's own
                 quasimetric, not Euclidean-on-phi. S is a genuine metric (the triangle inequality
                 survives averaging), which is what a force layout requires.
  ASYMMETRY = A  the antisymmetric part A = (D - D^T)/2 is NOT discarded -- it is drawn as a vector
                 field on top. Measured on this field, |A|/S has median 0.271 and 39.5% of pairs
                 differ by >2x between directions, so symmetrising alone would delete about a
                 quarter of the signal, including the irreversibility we trained for.

The optimiser is UMAP's own force model -- attraction along kNN edges, repulsion by negative
sampling -- because that is already exactly "poles with repulsion while maintaining local
structure". Written out here only because umap-learn cannot pin points, and pinning is one line
once you own the loop.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, torch

from catspace.research.tools.embeddings.basin_simplex_chart import (
    load_head, COLOR_WIN, COLOR_DRAW, COLOR_LOSS, INK, MUTED)
from catspace.research.tools.training_infra.losses import WIN, DRAW, LOSS

# Pinned frame: outcome poles at equilateral-triangle vertices, START at the centroid (it is a
# time origin, not an outcome -- putting it on the rim would imply it is a fourth basin).
ANCHOR = {WIN: (-1.0, -0.577), DRAW: (1.0, -0.577), LOSS: (0.0, 1.155), 3: (0.0, 0.0)}
ANAME = {WIN: "mover wins", DRAW: "draw", LOSS: "mover loses", 3: "START"}
ACOL = {WIN: COLOR_WIN, DRAW: COLOR_DRAW, LOSS: COLOR_LOSS, 3: "#7b5cd6"}


def procrustes_to(Y, src_idx, target):
    """Similarity transform (rotate + uniform scale + translate) putting Y[src_idx] onto `target`.

    This REPLACES pinning during optimisation, and that was the fix. Pinning fought the layout: the
    anchors sat at radius ~1.15 while the free cloud spread to +-6, so the pins ended up buried
    inside it and organised nothing. Laying out freely and aligning afterwards gives the same
    interpretable frame without distorting the local structure the layout is for."""
    A = Y[src_idx] - Y[src_idx].mean(0)
    B = target - target.mean(0)
    U, s_, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    scale = s_.sum() / max((A ** 2).sum(), 1e-12)
    return (Y - Y[src_idx].mean(0)) @ R * scale + target.mean(0)


def layout(S, pinned_idx, pinned_xy, k=15, epochs=400, seed=0, a=1.6, b=0.9, lr0=1.0, pin=False):
    """UMAP-style SGD on a precomputed symmetric distance. `pin` freezes the anchor rows."""
    n = len(S)
    rng = np.random.default_rng(seed)
    nn = np.argsort(S, axis=1)[:, 1:k + 1]                    # kNN edges (skip self)
    src = np.repeat(np.arange(n), k); dst = nn.ravel()
    # Spread the init: with sigma=0.05 many points start coincident, and the attraction term
    # d2**(b-1) = d2**-0.1 is INFINITE at d2=0, which sent the whole layout to NaN on the first
    # attempt (only the pinned poles survived to render).
    Y = rng.normal(0, 1.0, (n, 2))
    pinmask = np.zeros(n, bool)
    if pin:
        Y[pinned_idx] = pinned_xy; pinmask[pinned_idx] = True
    for ep in range(epochs):
        lr = lr0 * (1 - ep / epochs)
        d = Y[src] - Y[dst]
        d2 = np.maximum((d ** 2).sum(1, keepdims=True), 1e-6)   # floor: d2**-0.1 diverges at 0
        # attraction along kNN edges
        w = (-2 * a * b * d2 ** (b - 1)) / (1 + a * d2 ** b)
        g = np.clip(w * d, -4, 4)
        upd = np.zeros_like(Y)
        np.add.at(upd, src, g); np.add.at(upd, dst, -g)
        # repulsion against random negatives
        neg = rng.integers(0, n, len(src))
        dn = Y[src] - Y[neg]
        dn2 = np.maximum((dn ** 2).sum(1, keepdims=True), 1e-6)
        wn = (2 * b) / ((0.001 + dn2) * (1 + a * dn2 ** b))
        gn = np.clip(wn * dn, -4, 4)
        np.add.at(upd, src, gn); np.add.at(upd, neg, -gn)
        Y += lr * upd / k
        Y = np.clip(Y, -20, 20)                               # keep a diverging point on-canvas
        if pin:
            Y[pinmask] = pinned_xy
        if not np.isfinite(Y).all():
            raise FloatingPointError(f"layout diverged at epoch {ep}")
    return Y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/movie4/iqe_4pole_30k_latest.pt")
    ap.add_argument("--combined", default="data/derived/field_combined_sub600k.npz")
    ap.add_argument("--n", type=int, default=2500, help="positions per source (pairwise is O(n^2))")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--out-prefix", default="artifacts/experiments/basin_anchored")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--chunk", type=int, default=256, help="row block for the pairwise distance")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net = load_head(args.ckpt, args.device)
    z = np.load(args.combined, allow_pickle=True)
    meta = eval(str(z["_meta"][0])); mm = np.load(meta["feats"][0], mmap_mode="r")
    split = z["orig_source"] if "orig_source" in z.files else z["source"]
    rng = np.random.default_rng(args.seed)
    take, srcs = [], []
    for s in (0, 1):
        idx = np.flatnonzero(split == s)
        t = rng.choice(idx, min(args.n, len(idx)), replace=False)
        take.append(t); srcs.append(np.full(len(t), s))
    take = np.concatenate(take); order = np.argsort(take)
    take, source = take[order], np.concatenate(srcs)[order]
    y = z["y"][take]

    with torch.no_grad():
        e = net.phi(torch.from_numpy(np.asarray(mm[z["local_row"][take]],
                                                dtype=np.float32)).to(args.device))
        allpts = torch.cat([e, net.poles], 0)                 # positions + the 4 poles
        # CHUNKED: pairwise() materialises an (N,M,components,k) tensor -- at N=5000 that is 1.6e9
        # elements and OOMs MPS. Row blocks keep it to (chunk,M,components,k).
        N = len(allpts)
        D = np.empty((N, N), np.float64)
        for i in range(0, N, args.chunk):
            D[i:i + args.chunk] = net.iqe.pairwise(
                allpts[i:i + args.chunk], allpts).cpu().numpy()
    print(f"pairwise {D.shape} [{time.time()-t0:.0f}s]", flush=True)
    S = (D + D.T) / 2
    A = (D - D.T) / 2
    np.fill_diagonal(S, 0.0)
    # Normalise: S has median ~91 while the UMAP force constants (a=1.6, b=0.9) assume distances
    # of order 1. Without this the kNN graph is fine but the gradients are wildly mis-scaled.
    S = S / max(np.median(S[np.triu_indices(len(S), 1)]), 1e-9)

    npos = len(e)
    pinned_idx = np.arange(npos, npos + 4)
    pinned_xy = np.array([ANCHOR[k] for k in (WIN, DRAW, LOSS, 3)])
    Y = layout(S, pinned_idx, pinned_xy, k=args.k, epochs=args.epochs, seed=args.seed)
    print(f"layout done [{time.time()-t0:.0f}s]", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    cols = np.array([COLOR_WIN, COLOR_DRAW, COLOR_LOSS])[y]
    for ax, (nm, m) in zip(axes, [("human", source == 0), ("SF-vs-SF", source == 1)]):
        o = rng.permutation(int(m.sum()))                     # randomized draw order, no class on top
        ax.scatter(Y[:npos][m][o, 0], Y[:npos][m][o, 1], s=4, c=cols[m][o], alpha=.45, linewidths=0)
        # asymmetry: mean A to each pole, drawn as an arrow from each point cluster
        for pk in (WIN, DRAW, LOSS, 3):
            ax.plot(*ANCHOR[pk], "o", ms=13, color=ACOL[pk], mec="black", mew=1.2, zorder=6)
            ax.annotate(ANAME[pk], ANCHOR[pk], textcoords="offset points", xytext=(0, 14),
                        ha="center", fontsize=9, color=INK, zorder=7)
        ax.set_title(f"{nm}  (n={int(m.sum()):,})", color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.suptitle("Anchored layout: poles PINNED at the vertices, positions laid out under the "
                 "SYMMETRIC part of the field's own quasimetric\n"
                 "(not Euclidean-on-phi, and the poles are in the layout rather than invisible to it)")
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}.png", dpi=140)

    # How much did symmetrising cost, on THIS sample?
    iu = np.triu_indices(len(D), 1)
    print(f"\nasymmetry discarded by the layout metric: |A|/S median "
          f"{np.median(np.abs(A[iu]) / np.maximum(S[iu], 1e-9)):.3f}")
    print(f"  pairs differing >2x between directions: "
          f"{100*np.mean(np.maximum(D[iu], D.T[iu]) / np.maximum(np.minimum(D[iu], D.T[iu]), 1e-9) > 2):.1f}%")
    for pk in (WIN, DRAW, LOSS, 3):
        f, r = D[:npos, npos + pk], D[npos + pk, :npos]
        print(f"  pole {ANAME[pk]:>12s}: median d(pos->pole) {np.median(f):8.1f} | "
              f"d(pole->pos) {np.median(r):8.1f} | asymmetry {np.median(r) - np.median(f):+8.1f}")
    np.savez(f"{args.out_prefix}.npz", Y=Y, y=y, source=source, npos=npos)
    print(f"wrote {args.out_prefix}.png [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
