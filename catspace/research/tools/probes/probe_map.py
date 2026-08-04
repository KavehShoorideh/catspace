#!/usr/bin/env python
"""catspace/research/tools/probes/probe_map.py -- 2D map of a representation file colored by a label
column (the paper's "t-SNE of checkpoint embeddings coloured by atom"). Uses
openTSNE when available (supports out-of-sample transform) else sklearn.

Caption discipline (per the paper): t-SNE preserves NEIGHBOURHOODS, not global
distances -- the caption says so, and so should you.

Categorical labels use the validated fixed-order palette (>4 classes: largest 4
kept, rest folded into a muted 'other' -- never a generated 5th hue); continuous
labels use a single-hue sequential ramp.

Usage: catspace/research/tools/probes/probe_map.py rep.npz --color region --fig map.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from catspace.research.tools.figures import figlib                                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep")
    ap.add_argument("--color", required=True)
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--fig", default="map.png")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    d = np.load(args.rep, allow_pickle=True)
    X = d["emb"].astype(np.float64); c = d[args.color]
    if len(X) > args.sample:
        idx = np.sort(rng.choice(len(X), args.sample, replace=False))
        X, c = X[idx], c[idx]
    try:
        from openTSNE import TSNE
        Z = TSNE(perplexity=args.perplexity, random_state=0).fit(X)
        impl = "openTSNE (out-of-sample transform available)"
    except ImportError:
        from sklearn.manifold import TSNE
        Z = TSNE(perplexity=args.perplexity, random_state=0, init="pca").fit_transform(X)
        impl = "sklearn TSNE"
    fig, ax = figlib.new_fig(1, w=5.2, h=4.6)
    uniq = np.unique(c)
    if len(uniq) <= 12 and c.dtype.kind in "biuU":
        counts = {u: (c == u).sum() for u in uniq}
        top = sorted(counts, key=counts.get, reverse=True)[:4]
        for i, u in enumerate(top):
            m = c == u
            ax.scatter(Z[m, 0], Z[m, 1], s=6, color=figlib.CAT[i], label=str(u),
                       edgecolors="none")
        rest = ~np.isin(c, top)
        if rest.any():
            ax.scatter(Z[rest, 0], Z[rest, 1], s=5, color=figlib.MUTED,
                       label="other", alpha=0.5, edgecolors="none")
        ax.legend(frameon=False, markerscale=2, fontsize=8)
    else:
        sc = ax.scatter(Z[:, 0], Z[:, 1], s=6, c=c.astype(float),
                        cmap=figlib.SEQ_CMAP, edgecolors="none")
        fig.colorbar(sc, ax=ax, label=args.color, shrink=0.8)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_xlabel(f"{impl} — preserves neighbourhoods, not global distances",
                  fontsize=8, color=figlib.MUTED)
    figlib.save(fig, args.fig, f"Embedding map — colored by {args.color}")
    print(f"VERDICT map: {len(X)} points | color={args.color} "
          f"({'categorical' if len(uniq) <= 12 else 'continuous'}) | {impl}")


if __name__ == "__main__":
    main()
