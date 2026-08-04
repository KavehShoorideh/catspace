#!/usr/bin/env python
"""catspace/research/components/planner/approaches/endgame_groundtruth/experiments/eval_dtm_alignment.py — STABLE alignment readout for a field
checkpoint (the per-batch training rank_corr is noisy on 128 samples). Measures
how well the field's distance-to-mate orders held-out tablebase positions by
their TRUE DTM, for both readouts:

  centroid:  d(F(s), MATE_W centroid)                       (the old target)
  composed:  min_g[ d(F(s), B(g)) + dtm(g) ]  over a bank    (the surface target)

Reports overall + per-material Spearman on a large held-out sample. Run on each
ladder checkpoint to track whether the composed alignment is climbing.

Usage:
  .venv/bin/python catspace/research/components/planner/approaches/endgame_groundtruth/experiments/eval_dtm_alignment.py --ckpt data/derived/sep/qrl_dtm_surf_step10000.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


from catspace.research.components.memory.approaches.vector_store_retrieval.src.retrieval import composed_distance, dtm_waypoint_bank
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=paths.sep("qrl_dtm_surf.pt"))
    ap.add_argument("--dtm-npz", default=paths.derived("dtm_endgame.npz"))
    ap.add_argument("--n", type=int, default=2000, help="held-out eval positions")
    ap.add_argument("--bank", type=int, default=256)
    args = ap.parse_args()
    dev = "cpu"
    fb, pay = load_ckpt(Path(args.ckpt), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    zW = pay["zgoals"]["MATE_W"].to(dev).float()
    dz = np.load(args.dtm_npz)

    # bank (seed 0, same construction as the hinge) and a held-out eval set disjoint
    bank = dtm_waypoint_bank(args.dtm_npz, args.bank, seed=0).refresh(fb, dev)
    rng = np.random.default_rng(123)
    all_idx = np.setdiff1d(np.arange(len(dz["dtm"])), np.arange(len(dz["dtm"]))[:args.bank])
    idx = rng.choice(all_idx, min(args.n, len(all_idx)), replace=False)
    pk, mt, dtm, mat = dz["packed"][idx], dz["meta"][idx], dz["dtm"][idx], dz["material"][idx]
    with torch.no_grad():
        F = fb.embed_F(torch.from_numpy(feature_planes(pk, mt)),
                       torch.from_numpy(np.tile(om, (len(idx), 1))))
        dC = fb.distance_matrix(F, zW[None, :])[:, 0].cpu().numpy()      # centroid
        dK = composed_distance(fb, F, bank).cpu().numpy()               # composed surface

    names = {0: "krrkbp", 1: "krrvk", 2: "krvk"}
    sC = spearmanr(dC, dtm).correlation
    sK = spearmanr(dK, dtm).correlation
    print(f"ckpt={Path(args.ckpt).name} step={pay.get('step','?')}  n={len(idx)}")
    print(f"  CENTROID  spearman(d, DTM) = {sC:+.3f}")
    print(f"  COMPOSED  spearman(d, DTM) = {sK:+.3f}   (surface; higher=better aligned)")
    for m in np.unique(mat):
        msk = mat == m
        print(f"     within {names[int(m)]:7s} n={int(msk.sum())}: "
              f"centroid={spearmanr(dC[msk], dtm[msk]).correlation:+.3f} "
              f"composed={spearmanr(dK[msk], dtm[msk]).correlation:+.3f}")
    print(f"VERDICT DTM_ALIGN step={pay.get('step','?')} composed={sK:+.3f} centroid={sC:+.3f}")


if __name__ == "__main__":
    main()
