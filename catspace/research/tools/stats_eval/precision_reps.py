#!/usr/bin/env python
"""catspace/research/tools/stats_eval/precision_reps.py -- does representing a position by its DISTANCE/DIRECTION to the B-goal
space make concepts more PRECISE than the raw F embedding? (Kaposi 2026-07-21.) For each named concept,
train a supervised CAV (linear probe) on several representations and compare held-out precision@K:

  * F            -- raw source embedding (what we used; connected_rooks capped ~60%).
  * B            -- raw goal embedding of the position.
  * dist-to-B    -- [ d(F(s), B_anchor_k) ]_k : how far the position is from each B-cluster (goal region).
  * dir-to-B     -- unit directions (B_anchor_k - B(s)) flattened: which way to each goal region.

If a concept is really "which goals am I near / heading to", the B-relative reps should beat F.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.components.encoder.approaches.concept_quantization.experiments.concept_features import features as named_features
from catspace.io import paths


def prec_at(scores, label, ks=(5, 20, 100)):
    order = np.argsort(-scores)
    return {k: float(label[order[:k]].mean()) for k in ks}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default=paths.sep("lichess_gn_iqeqrl_full.pt"))
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=14000)
    ap.add_argument("--anchors", type=int, default=64)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)

    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    nz = np.load(args.shard)
    P, M, ply = np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["ply"]).astype(int)
    idx = np.flatnonzero(ply >= args.min_ply); idx = idx[rng.permutation(len(idx))[:args.n]]
    Pk, Mk = P[idx], M[idx]
    boards = [board_from_packed(Pk[i], Mk[i]) for i in range(len(Pk))]
    with torch.no_grad():
        t = torch.from_numpy(feature_planes(Pk, Mk)).to(dev)
        F = fb.embed_F(t, torch.from_numpy(np.tile(om, (len(Pk), 1))).to(dev))
        B = fb.embed_B(t)
        Bc = torch.from_numpy(KMeans(args.anchors, n_init=4, random_state=args.seed)
                              .fit(B.cpu().numpy()).cluster_centers_).float().to(dev)
        dist_to_B = fb.distance_matrix(F, Bc).cpu().numpy()        # (N, anchors)
        Bn = B.cpu().numpy(); Bc_np = Bc.cpu().numpy()
        # direction to each anchor in B-space, unit-normalized, flattened
        dir_to_B = (Bc_np[None, :, :] - Bn[:, None, :])
        dir_to_B = (dir_to_B / (np.linalg.norm(dir_to_B, axis=2, keepdims=True) + 1e-9)).reshape(len(Bn), -1)
        F = F.cpu().numpy()
    reps = {"F": F, "B": Bn, "dist-to-B": dist_to_B, "dir-to-B": dir_to_B}
    reps = {k: (v - v.mean(0)) / (v.std(0) + 1e-8) for k, v in reps.items()}

    feats = [named_features(b) for b in boards]
    fnames = [n for n in feats[0] if not n.endswith("_ctrl") and feats[0][n][1] == "bin"]
    Fmat = np.array([[float(f[n][0]) for n in fnames] for f in feats])
    tr_i, te_i = train_test_split(np.arange(len(Pk)), test_size=0.4, random_state=args.seed)
    print(f"[stage] {len(Pk)} positions, {args.anchors} B-anchors ({time.time()-t0:.0f}s)")
    print(f"VERDICT PRECISION_REPS field={Path(args.field).stem} n={len(Pk)}  (CAV precision@5 held-out, per rep)")
    print(f"  {'concept':16s} {'base':>5s} | " + " ".join(f"{k:>10s}" for k in reps))
    for j, nm in enumerate(fnames):
        y = Fmat[:, j]
        if not (0.03 < y.mean() < 0.97):
            continue
        cells = []
        for k, X in reps.items():
            clf = LogisticRegression(max_iter=300).fit(X[tr_i], y[tr_i])
            p = prec_at(clf.decision_function(X[te_i]), y[te_i])
            cells.append(f"{p[5]:>4.0%}/{p[100]:>3.0%}")
        print(f"  {nm.replace('_w',''):16s} {y.mean():>4.0%} | " + " ".join(f"{c:>10s}" for c in cells))
    print("  (each cell = precision@5 / precision@100)")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
