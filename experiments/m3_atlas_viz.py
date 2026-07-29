#!/usr/bin/env python
"""experiments/m3_atlas_viz.py -- the M3 atlas artifact: the subgoal map a planner navigates.

2-D UMAP of the 1024 phi-region centroids; panels: SF-refereed crossing flux per Elo band
(where THEY err -- the exploitable ridges), mean committor (outcome phase), and the top-decile
composite subgoal cells (region x contested-band) that passed the 3.3x/3.0x enrichment gate.
Output: artifacts/experiments/m3_atlas.png (+ docs/figures/ copy for the repo record).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    t = np.load("data/derived/reach/region_table_v2.npz", allow_pickle=True)
    bank = t["regions"]; rate = t["crossing_rate"]; qual = t["committor_mean"]
    pcond = t["pcond"]; G, CB, B = pcond.shape
    # region-level flux per band = contested-cell rate weighted by P(contested | region)
    flux_b = np.stack([rate[:, b].reshape(G, CB)[:, 1] for b in range(B)], 1)   # contested band
    qual_r = qual[:, 0].reshape(G, CB).mean(1)

    import umap
    xy = umap.UMAP(n_neighbors=15, min_dist=0.15, random_state=0).fit_transform(bank)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for b, ax, name in ((0, axes[0, 0], "<1500"), (1, axes[0, 1], ">=1500")):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=flux_b[:, b], cmap="inferno", s=14,
                        vmin=0, vmax=np.percentile(flux_b, 98))
        ax.set_title(f"crossing flux (contested cells), band {name}")
        plt.colorbar(sc, ax=ax, fraction=0.046)
    sc = axes[1, 0].scatter(xy[:, 0], xy[:, 1], c=qual_r, cmap="RdBu", s=14, vmin=0, vmax=1)
    axes[1, 0].set_title("mean committor (outcome phase; red=losing, blue=winning)")
    plt.colorbar(sc, ax=axes[1, 0], fraction=0.046)
    top = np.argsort(-(flux_b[:, 0] + flux_b[:, 1]))[: G // 10]
    axes[1, 1].scatter(xy[:, 0], xy[:, 1], c="lightgray", s=10)
    axes[1, 1].scatter(xy[top, 0], xy[top, 1], c="crimson", s=22)
    axes[1, 1].set_title("top-decile subgoal regions (both bands pooled)")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("catspace M3 atlas — 1024 φ-regions × committor bands "
                 "(SF-refereed flux, even-game table)", fontsize=13)
    fig.tight_layout()
    out = Path("artifacts/experiments/m3_atlas.png")
    fig.savefig(out, dpi=130)
    Path("docs/figures").mkdir(exist_ok=True, parents=True)
    fig.savefig("docs/figures/m3_atlas.png", dpi=130)
    print(f"wrote {out} + docs/figures/m3_atlas.png")


if __name__ == "__main__":
    main()
