#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/concept_quantization/experiments/native_concepts.py -- discover concept DIRECTIONS in the field UNSUPERVISED (Kaposi
2026-07-21: "find the clusters/directions natively, without the hand-coded features as a crutch").

We confirmed structure is there as directions (connected rooks / king safety decode beyond phase). Now
find them without naming anything:

  1. embed F, PROJECT OUT phase (piece_count) -- the dominant axis that otherwise drowns the structure;
  2. run unsupervised direction-finding on the residual: PCA (variance axes) and ICA (independent axes);
  3. each discovered axis is a candidate CONCEPT.

The named features are used ONLY as a post-hoc MIRROR to check our work -- does an axis re-discover
"connected rooks" on its own? -- never to find the axes. Axes that match no named feature are NOVEL
concepts; each is characterized natively by the shared piece-placement of its extreme positions (a
concept heatmap), so we can read new concepts straight off the field.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.components.encoder.approaches.concept_quantization.experiments.concept_features import features as named_features   # VALIDATION MIRROR ONLY
from catspace.io import paths


def residualize(X, z):
    """remove the linear component of z (phase) from every column of X."""
    z = (z - z.mean()) / (z.std() + 1e-9)
    beta = (X * z[:, None]).mean(0) / (z * z).mean()
    return X - np.outer(z, beta)


def concept_heatmap(Pk, Mk, order, top=40):
    """native characterization of an axis: mean piece-occupancy over its most-extreme positions,
    minus the global mean -> which squares/pieces light up. Returns a short human string."""
    sym = {}
    for i in order[:top]:
        for sq, p in board_from_packed(Pk[i], Mk[i]).piece_map().items():
            sym[(p.symbol(), sq)] = sym.get((p.symbol(), sq), 0) + 1
    hot = sorted(sym.items(), key=lambda kv: -kv[1])[:6]
    return " ".join(f"{s}{chess.square_name(sq)}:{100*c//top}%" for (s, sq), c in hot)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default=paths.sep("lichess_gn_iqeqrl_full.pt"))
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=9000)
    ap.add_argument("--k", type=int, default=16, help="axes to discover")
    ap.add_argument("--tower", choices=["F", "B"], default="F")
    ap.add_argument("--min-ply", type=int, default=16)
    ap.add_argument("--match-thresh", type=float, default=0.25, help="|corr| to call an axis a known concept")
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

    with torch.no_grad():
        t = torch.from_numpy(feature_planes(Pk, Mk)).to(dev)
        emb = (fb.embed_F(t, torch.from_numpy(np.tile(om, (len(Pk), 1))).to(dev)) if args.tower == "F"
               else fb.embed_B(t)).cpu().numpy()

    # named features -- VALIDATION MIRROR ONLY (never used to find the axes)
    feats = [named_features(board_from_packed(Pk[i], Mk[i])) for i in range(len(Pk))]
    fnames = [n for n in feats[0] if not n.endswith("_ctrl")]
    Fmat = np.array([[float(f[n][0]) for n in fnames] for f in feats])
    phase = Fmat[:, fnames.index("piece_count")]

    Xr = StandardScaler().fit_transform(residualize(emb, phase))   # phase-projected-out residual
    print(f"[stage] {len(Pk)} positions, tower={args.tower}, phase projected out, discovering {args.k} axes "
          f"({time.time()-t0:.0f}s)", flush=True)

    for method, model in [("PCA", PCA(n_components=args.k, random_state=args.seed)),
                          ("ICA", FastICA(n_components=args.k, random_state=args.seed, max_iter=400))]:
        S = model.fit_transform(Xr)                                # (n, k) axis scores
        S = (S - S.mean(0)) / (S.std(0) + 1e-9)
        C = np.array([[abs(np.corrcoef(S[:, a], Fmat[:, j])[0, 1]) for j in range(len(fnames))]
                      for a in range(args.k)])                     # |corr| axis x named-feature
        print(f"VERDICT NATIVE_CONCEPTS method={method} tower={args.tower} k={args.k}")
        # (1) did the KNOWN concepts re-emerge natively?
        print("  known concepts re-discovered natively (best-matching axis |corr|):")
        for j, nm in enumerate(fnames):
            a = int(C[:, j].argmax())
            flag = "OK" if C[a, j] >= args.match_thresh else "--"
            print(f"      {nm:18s} -> axis {a:2d}  |corr| {C[a, j]:.2f}  [{flag}]")
        # (2) NOVEL axes: strong, but match no named feature -> new concepts, characterized natively
        novel = [a for a in range(args.k) if C[a].max() < args.match_thresh]
        print(f"  NOVEL axes (match no named feature, |corr|<{args.match_thresh}): {len(novel)}/{args.k}")
        for a in novel[:5]:
            order = np.argsort(-S[:, a])
            print(f"      axis {a:2d}: +extreme {concept_heatmap(Pk, Mk, order)}")
        print()
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
