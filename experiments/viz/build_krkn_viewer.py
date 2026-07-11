#!/usr/bin/env python
"""
experiments/viz/build_krkn_viewer.py — build the interactive KRkn linked
viewer, with a pluggable 2D projection (--projection {tsne,pca,umap}).

Requires krkn_F.npy/krkn_B.npy/krkn_scores.npy and dtm_krkn.npy in
data/derived/ (produced by experiments/train_krkn.py and
latentchess/domains/krkn.py's __main__).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from latentchess.domains import krkn
from latentchess.opponents import optimal_reply_table
from latentchess.viz.projection import fit_map, FittedMap, PROJECTIONS
from latentchess.viz.payload import (KrknViewerBuilder, build_games, attach_cones,
                                      finalize_with_xy, build_background, json_default)
from latentchess.viz.build_html import build_html
from latentchess.io.paths import derived_dir, generated_dir, load_array


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projection", choices=list(PROJECTIONS), default="tsne")
    ap.add_argument("--out-name", default=None, help="defaults to krkn-linked-viewer-<projection>.html")
    args = ap.parse_args()

    chain = krkn.build_chain(verbose=False)
    dtm = load_array("dtm_krkn") if (derived_dir() / "dtm_krkn.npy").exists() else krkn.compute_dtm(chain)
    F = load_array("krkn_F")
    B = load_array("krkn_B")
    scores = load_array("krkn_scores")
    won = np.isfinite(dtm[: chain.strata["KRkn"].stop])
    n2 = chain.strata["KRkn"].stop
    b_opt = optimal_reply_table(chain, dtm)

    fmap = fit_map(F, dtm[:n2], won, kind=args.projection,
                   extra_pool=n2 + np.arange(chain.n_live - n2))

    builder = KrknViewerBuilder(chain, dtm, scores, b_opt)
    bands = [(15, 19), (13, 15), (11, 13), (9, 11), (7, 9)]
    games = build_games(builder, bands)
    attach_cones(games, builder)
    games = finalize_with_xy(games, F, fmap)
    bg = build_background(F, fmap, won, n2)

    data = dict(N=5, games=games, bg=bg,
                meta=dict(domain="KRkn on 5x5", opponent="optimal defense (max-DTM, capture-aware)",
                          planner="learned cone, 1-ply minimax readout",
                          oracle="DTM-perfect play (tablebase)",
                          cone="MC futures, black eps=0.25, colored by ply depth",
                          projection=args.projection))

    out_name = args.out_name or f"krkn-linked-viewer-{args.projection}.html"
    template = Path(__file__).resolve().parents[2] / "latentchess/viz/templates/krkn_viewer.html"
    out = generated_dir() / out_name
    build_html(template, data, out)
    print(f"wrote {out} ({len(json.dumps(data, default=json_default)) // 1024} KB)")


if __name__ == "__main__":
    main()
