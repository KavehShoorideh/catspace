#!/usr/bin/env python
"""experiments/visualize_clusters.py — visualize field structure before/after the
cluster fine-tune (Kaveh 2026-07-19: "visualize the field to see the clusters").
Embeds endgame positions with the incumbent vs clustered field, UMAP-projects
each to 2D, and colors by DTM and by material. Structureless incumbent => a blob;
clustered field => DTM gradient / material groups."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt

dev = "cpu"
om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
dz = np.load("data/derived/dtm_endgame.npz")
rng = np.random.default_rng(0)
# balanced sample across materials, capped
idx = np.concatenate([rng.choice(np.flatnonzero(dz["material"] == m), 600, replace=False)
                      for m in (0, 1, 2)])
rng.shuffle(idx)
packed, meta, dtm, mat = dz["packed"][idx], dz["meta"][idx], dz["dtm"][idx], dz["material"][idx]
planes = torch.from_numpy(feature_planes(packed, meta))
omt = torch.from_numpy(np.tile(om, (len(idx), 1)))
names = {0: "KRRvKBP", 1: "KRRvK", 2: "KRvK"}
matcol = np.array(["#e45756", "#4c78a8", "#54a24b"])[mat]

fields = [("incumbent (cert_base_full)", "data/derived/sep/cert_base_full.pt"),
          ("clustered (cert_base_cluster)", "data/derived/sep/cert_base_cluster.pt")]
fig, axes = plt.subplots(2, 2, figsize=(13, 12))
for col, (title, ckpt) in enumerate(fields):
    fb, _ = load_ckpt(Path(ckpt), dev); fb.eval()
    with torch.no_grad():
        F = fb.embed_F(planes, omt).cpu().numpy()
    xy = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                   random_state=0, n_components=2).fit_transform(F)
    sc = axes[0, col].scatter(xy[:, 0], xy[:, 1], c=dtm, cmap="viridis", s=6, alpha=0.7)
    axes[0, col].set_title(f"{title}\ncolored by DTM (plies to mate)")
    plt.colorbar(sc, ax=axes[0, col], shrink=0.8, label="DTM")
    for m in (0, 1, 2):
        mk = mat == m
        axes[1, col].scatter(xy[mk, 0], xy[mk, 1], c=matcol[mk][0], s=6, alpha=0.7, label=names[m])
    axes[1, col].set_title("colored by material")
    axes[1, col].legend(markerscale=2, fontsize=9)
    for r in (0, 1):
        axes[r, col].set_xticks([]); axes[r, col].set_yticks([])

fig.suptitle("Field structure: incumbent (structureless) vs cluster-fine-tuned\n"
             "symmetry-invariance 0.91->2.37, DTM-clustering 1.02->1.35", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "artifacts/experiments/field_clusters.png"
fig.savefig(out, dpi=110)
print(f"VERDICT CLUSTER_VIZ saved {out} (n={len(idx)}, 600/material)")
