#!/usr/bin/env python
"""basin_perply_umap.py -- Kaveh 2026-08-04: bin by ply, then UMAP WITHIN each bin.

The measured problem: 54%% of phi variance is linearly explained by elapsed ply and only 1.8%% by
outcome. So any layout over all plies is a layout of game phase, and the basin structure is a small
residual riding on it. Stratifying fixes that by construction -- hold ply nearly constant inside a
bin and it CANNOT dominate the layout, because it barely varies. What is left to organise the
points is the outcome structure.

This is the cleanest attack on the ply-dominance problem: it removes the nuisance variable rather
than hoping the algorithm looks past it.

UMAP runs on the SYMMETRIC part of the field's own quasimetric (metric='precomputed'), not
Euclidean-on-phi. Every earlier UMAP in this project silently discarded the quasimetric. The
asymmetry that symmetrising drops is real (median |A|/S = 0.27, and 39.5%% of pairs differ >2x by
direction) but a symmetric metric is what UMAP requires, and it is stated rather than glossed.
"""
from __future__ import annotations
import argparse, time
import numpy as np, torch

from catspace.research.components.encoder.approaches.reachability_field.src.iqe_head import IQEHead
from catspace.research.tools.embeddings.basin_simplex_chart import (
    COLOR_WIN, COLOR_DRAW, COLOR_LOSS, INK, MUTED)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/movie4/iqe_4pole_30k_latest.pt")
    ap.add_argument("--combined", default="data/derived/field_combined_sub600k.npz")
    ap.add_argument("--bands", default="0,12,24,40,60,100")
    ap.add_argument("--n", type=int, default=1100, help="positions per band per source")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--out-prefix", default="artifacts/experiments/basin_perply_umap")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    t0 = time.time()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    p = torch.load(args.ckpt, map_location=args.device, weights_only=False); cfg = p["cfg"]
    net = IQEHead(in_ch=cfg["in_ch"], d=cfg["d"], components=cfg["components"],
                  adapter_ch=cfg["adapter_ch"]).to(args.device)
    net.load_compat(p["state_dict"]); net.eval()
    z = np.load(args.combined, allow_pickle=True)
    meta = eval(str(z["_meta"][0])); mm = np.load(meta["feats"][0], mmap_mode="r")
    split = z["orig_source"] if "orig_source" in z.files else z["source"]
    rng = np.random.default_rng(0)
    edges = [int(v) for v in args.bands.split(",")]
    bands = list(zip(edges[:-1], edges[1:]))

    fig, axes = plt.subplots(2, len(bands), figsize=(3.6 * len(bands), 7.6))
    print(f"{'band':>10s} {'n':>6s} {'silhouette by outcome':>22s}")
    for bi, (lo, hi) in enumerate(bands):
        idx = []
        for s in (0, 1):
            c = np.flatnonzero((split == s) & (z["ply"] >= lo) & (z["ply"] < hi))
            idx.append(rng.choice(c, min(args.n, len(c)), replace=False))
        take = np.sort(np.concatenate(idx))
        src, yl = split[take], z["y"][take]
        with torch.no_grad():
            e = net.phi(torch.from_numpy(np.asarray(mm[z["local_row"][take]],
                                                    dtype=np.float32)).to(args.device))
            N = len(e); D = np.empty((N, N), np.float64)
            for i in range(0, N, args.chunk):
                D[i:i + args.chunk] = net.iqe.pairwise(e[i:i + args.chunk], e).cpu().numpy()
        S = (D + D.T) / 2; np.fill_diagonal(S, 0.0)
        Y = umap.UMAP(n_neighbors=25, min_dist=0.12, metric="precomputed",
                      random_state=0).fit_transform(S)
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(Y, yl) if len(np.unique(yl)) > 1 else float("nan")
        print(f"  {lo:>3d}-{hi:<5d} {N:>6d} {sil:>22.4f}")
        cols = np.array([COLOR_WIN, COLOR_DRAW, COLOR_LOSS])[yl]
        for r, (sn, m) in enumerate([("human", src == 0), ("SF-vs-SF", src == 1)]):
            ax = axes[r, bi]; o = rng.permutation(int(m.sum()))
            ax.scatter(Y[m][o, 0], Y[m][o, 1], s=4, c=cols[m][o], alpha=.55, linewidths=0)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            if r == 0:
                ax.set_title(f"ply {lo}-{hi}", color=INK)
            if bi == 0:
                ax.set_ylabel(sn, color=INK)
    fig.suptitle("UMAP WITHIN each ply band, on the field's own (symmetrised) quasimetric\n"
                 "ply is nearly constant inside a band, so it cannot dominate the layout -- "
                 "colour is the eventual outcome")
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}.png", dpi=140)
    print(f"wrote {args.out_prefix}.png [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
