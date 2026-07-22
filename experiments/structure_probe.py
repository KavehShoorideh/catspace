#!/usr/bin/env python
"""experiments/structure_probe.py -- is STRUCTURE present in F / B, or discarded? (Kaposi 2026-07-21).

The value field's B failed to *cluster* by structure -- but clustering only sees the dominant variance
axis (value). Kaposi's point: "where did we come from" (B) should still ENCODE structure, possibly in
a subspace the clustering ignores. So test it the sensitive way -- a SUPERVISED linear probe:

  can a linear readout of F / B recover the material signature (a clean structure label, orthogonal to
  value) and the outcome (value)? Compare against the raw input planes (structure ceiling) and chance.

Reading: probe_acc(B -> material) near the raw ceiling => structure is fully there, just off-axis =>
Kaposi is right, cluster the structure SUBSPACE. Near chance => structure really was discarded.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="data/derived/sep/lichess_gn_iqeqrl_full.pt")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=9000)
    ap.add_argument("--top-materials", type=int, default=25, help="probe the K most common material classes")
    ap.add_argument("--min-ply", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)

    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    nz = np.load(args.shard)
    P, M = np.asarray(nz["packed"]), np.asarray(nz["meta"])
    ply = np.asarray(nz["ply"]).astype(int)
    res = np.asarray(nz["result"]).astype(int) if "result" in nz else np.zeros(len(P), int)
    idx = np.flatnonzero(ply >= args.min_ply)
    idx = idx[rng.permutation(len(idx))[:args.n]]
    Pk, Mk, resk = P[idx], M[idx], res[idx]

    # structure label = material signature (piece multiset) -- pure structure, independent of value
    mat = np.array(["".join(sorted(p.symbol() for p in board_from_packed(Pk[i], Mk[i]).piece_map().values()))
                    for i in range(len(Pk))])
    common = [m for m, _ in Counter(mat).most_common(args.top_materials)]
    keep = np.isin(mat, common)
    Pk, Mk, resk, mat = Pk[keep], Mk[keep], resk[keep], mat[keep]
    ymat = np.array([common.index(m) for m in mat])
    print(f"[stage] {len(Pk)} positions over top-{args.top_materials} materials "
          f"(cover {100*keep.mean():.0f}% of ply>={args.min_ply}); value labels W/D/L="
          f"{int((resk==1).sum())}/{int((resk==0).sum())}/{int((resk==-1).sum())} ({time.time()-t0:.0f}s)", flush=True)

    # representations: raw planes (structure ceiling), F, B
    planes = feature_planes(Pk, Mk).reshape(len(Pk), -1)
    with torch.no_grad():
        t = torch.from_numpy(feature_planes(Pk, Mk)).to(dev)
        F = fb.embed_F(t, torch.from_numpy(np.tile(om, (len(Pk), 1))).to(dev)).cpu().numpy()
        B = fb.embed_B(t).cpu().numpy()
    reps = {"raw-planes(ceiling)": planes, "F": F, "B": B}

    from sklearn.neural_network import MLPClassifier

    def probe(X, y, kind="linear"):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=args.seed)
        clf = (LogisticRegression(max_iter=200, C=1.0) if kind == "linear"
               else MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=120, random_state=args.seed))
        clf.fit(Xtr, ytr)
        return float(clf.score(Xte, yte))

    mat_major = Counter(ymat).most_common(1)[0][1] / len(ymat)
    val_major = Counter(resk).most_common(1)[0][1] / len(resk)
    print(f"VERDICT STRUCTURE_PROBE field={Path(args.field).stem} n={len(Pk)}")
    print(f"  STRUCTURE (material, {args.top_materials}-way, chance {1/args.top_materials:.3f} / majority {mat_major:.3f}):")
    for name, X in reps.items():
        print(f"      {name:20s} -> linear {probe(X, ymat, 'linear'):.3f} | MLP {probe(X, ymat, 'mlp'):.3f}")
    print(f"  VALUE (outcome W/D/L, majority {val_major:.3f}):")
    for name, X in reps.items():
        print(f"      {name:20s} -> linear {probe(X, resk, 'linear'):.3f} | MLP {probe(X, resk, 'mlp'):.3f}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
