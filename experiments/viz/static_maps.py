#!/usr/bin/env python
"""
experiments/viz/static_maps.py — static region-map figures via the pluggable
Projection2D/FittedMap seam (consolidates atlas.py/region_map.py/krkn_map.py/
krkn's tsne_maps.py/tsne_cones.py's shared fit-project-plot pattern into one
driver, parameterized by --which and --projection instead of one script per
domain x projection combination).

--which krkn: KRkn token-territory + win/draw region map (the domain with a
              trained field from experiments/train_krkn.py).
--which krk:  KRk box-area/DTM atlas panel (on-the-fly exact-dynamics field).

--projection {tsne,pca[,umap]} swaps the 2D map with no other code change --
the concrete demonstration that the viz layer no longer hardcodes t-SNE.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latentchess.chain import exact_P
from latentchess.concepts import KMeansVQ
from latentchess.cone.tabular import TabularFB
from latentchess.domains import krk, krkn
from latentchess.io.paths import generated_dir, load_array
from latentchess.viz.plots import style
from latentchess.viz.projection import fit_map


def render_krkn(projection: str, out_dir):
    chain = krkn.build_chain(verbose=False)
    dtm = load_array("dtm_krkn")
    F = load_array("krkn_F")

    n2 = chain.strata["KRkn"].stop
    live_dtm = dtm[:chain.n_live]
    won = np.isfinite(live_dtm)
    extra_pool = np.arange(n2, chain.n_live)   # the KRk sub-stratum

    fmap = fit_map(F[:chain.n_live], live_dtm, won, kind=projection, extra_pool=extra_pool, seed=0)
    tokens = KMeansVQ(n_tokens=16, seed=0).fit(F[:n2]).tokens(F[:n2])

    P_bg = fmap.fit_points()
    fit_idx = fmap.fit_idx
    P_tok = fmap.project(F, np.arange(n2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), facecolor="#10151C")
    ax = axes[0]
    cls = np.where(fit_idx >= n2, 2, np.where(won[np.clip(fit_idx, 0, n2 - 1)], 0, 1))
    colors = np.array(["#4EC9B0", "#6B7280", "#C97B4E"])[cls]
    ax.scatter(P_bg[:, 0], P_bg[:, 1], s=2, c=colors, alpha=0.4, linewidths=0)
    style(ax, f"KRkn win/draw/KRk-stratum ({projection})")

    ax = axes[1]
    ax.scatter(P_tok[:, 0], P_tok[:, 1], s=2, c=tokens, cmap="tab20", alpha=0.5, linewidths=0)
    style(ax, f"KRkn token territories, K=16 ({projection})")

    plt.tight_layout()
    out = out_dir / f"krkn_region_map_{projection}.png"
    plt.savefig(out, dpi=130, facecolor="#10151C")
    return out


def render_krk(projection: str, out_dir):
    chain = krk.build_chain()
    W, B = krk.enumerate_states()
    dtm_w, _ = krk.compute_dtm(W, B)
    feats = krk.concept_features(W, dtm_w)

    P_exact = exact_P(chain)
    emb = TabularFB.fit(P_exact, gamma=0.92, d=64, seed=0)
    F = emb.F[:chain.n_live]
    won = np.isfinite(dtm_w)

    fmap = fit_map(F, dtm_w, won, kind=projection, seed=0)
    P_live = fmap.project(F, np.arange(chain.n_live))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), facecolor="#10151C")
    ax = axes[0]
    sc = ax.scatter(P_live[:, 0], P_live[:, 1], s=3, c=dtm_w, cmap="viridis", alpha=0.6, linewidths=0)
    plt.colorbar(sc, ax=ax, label="DTM")
    style(ax, f"KRk colored by DTM ({projection})")

    ax = axes[1]
    sc = ax.scatter(P_live[:, 0], P_live[:, 1], s=3, c=feats["box_area"], cmap="magma", alpha=0.6, linewidths=0)
    plt.colorbar(sc, ax=ax, label="box area")
    style(ax, f"KRk colored by box area ({projection})")

    plt.tight_layout()
    out = out_dir / f"krk_atlas_{projection}.png"
    plt.savefig(out, dpi=130, facecolor="#10151C")
    return out


RENDERERS = {"krkn": render_krkn, "krk": render_krk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=list(RENDERERS), default="krkn")
    ap.add_argument("--projection", choices=["pca", "tsne", "umap"], default="pca")
    args = ap.parse_args()

    out_dir = generated_dir()
    out = RENDERERS[args.which](args.projection, out_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
