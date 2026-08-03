#!/usr/bin/env python
"""catspace/research/components/planner/approaches/reach_field/experiments/build_aug_feats.py -- human-axis features for the AUGMENTED codebook (the fired
incremental-eyes gate, JOURNAL 2026-07-29/30): per position, from the ALREADY-CACHED Maia-2
candidates -- no new inference. Features: policy entropy (top-16, renormalized), top-prob,
top-gap, win_prob. Saved row-aligned to the cache; the codebook clusters in
[zs(phi) ⊕ w·zs(feats)] with the scaler stored for query-time assignment.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from catspace.research.components.planner.approaches.opponent_model.src.style_dataio import load_cache                     # noqa: E402
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=paths.derived("m2b/cache_v2"))
    ap.add_argument("--out", default=paths.reach("aug_feats_v2.npz"))
    args = ap.parse_args()
    c = load_cache(args.cache)
    lp = c["cand_logp"].astype(np.float32)                       # (N,17) log p (top16 + played)
    p = np.exp(lp)
    p /= np.maximum(p.sum(1, keepdims=True), 1e-9)               # renormalized over candidates
    ps = np.sort(p, axis=1)[:, ::-1]
    ent = -(p * np.log(np.maximum(p, 1e-12))).sum(1)
    feats = np.stack([ent, ps[:, 0], ps[:, 0] - ps[:, 1],
                      c["win_prob"].astype(np.float32)], 1)
    print(f"AUDIT feats {feats.shape}: entropy med {np.median(ent):.3f} | top-p med "
          f"{np.median(ps[:,0]):.3f} | win_prob med {np.median(feats[:,3]):.3f} | "
          f"nan {np.isnan(feats).sum()}")
    assert not np.isnan(feats).any()
    np.savez_compressed(args.out, feats=feats, meta_cache=str(args.cache))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
