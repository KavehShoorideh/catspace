#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/concept_quantization/experiments/concept_cones.py -- concepts as CONES / density regions where PATH-density is high
(Kaposi 2026-07-21: "not exact directions but continuous sections of the space where density of paths
is high"; cf. Concept Cones, arXiv 2512.07355). SAE atoms are point-directions; a concept is really a
continuous fan of related directions -- a cone -- and the ones that matter are where many game-PATHS
concentrate.

  embed positions from REAL games (keep game_id) -> density-cluster the direction space (HDBSCAN, so
  regions have arbitrary shape and sparse space is left as non-concept) -> each dense region is a
  concept CONE: a center direction + angular spread, weighted by PATH-density (distinct games that
  traverse it). Characterized natively (piece-placement heatmap) + a post-hoc named-feature mirror.

Path-density (fraction of all games passing through the cone) = your density prior: the cones a lot of
play funnels through are the load-bearing concepts.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.components.encoder.approaches.concept_quantization.experiments.concept_features import features as named_features   # MIRROR ONLY
from catspace.io import paths

try:
    from sklearn.cluster import HDBSCAN
except Exception:
    HDBSCAN = None


def heatmap(Pk, Mk, members, top=50):
    sym = {}
    for i in members[:top]:
        for sq, p in board_from_packed(Pk[i], Mk[i]).piece_map().items():
            sym[(p.symbol(), sq)] = sym.get((p.symbol(), sq), 0) + 1
    n = min(len(members), top)
    return " ".join(f"{s}{chess.square_name(sq)}:{100*c//n}%"
                    for (s, sq), c in sorted(sym.items(), key=lambda kv: -kv[1])[:6])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default=paths.sep("lichess_gn_iqeqrl_full.pt"))
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--tower", choices=["F", "B"], default="F")
    ap.add_argument("--min-cluster", type=int, default=60, help="HDBSCAN min_cluster_size (cone granularity)")
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if HDBSCAN is None:
        raise SystemExit("sklearn.cluster.HDBSCAN unavailable (needs sklearn>=1.3)")
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)

    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    nz = np.load(args.shard)
    P, M, ply = np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["ply"]).astype(int)
    gid = np.asarray(nz["game_id"])
    cand = np.flatnonzero(ply >= args.min_ply)
    idx = cand[rng.permutation(len(cand))[:args.n]]
    Pk, Mk, gk = P[idx], M[idx], gid[idx]
    with torch.no_grad():
        t = torch.from_numpy(feature_planes(Pk, Mk)).to(dev)
        emb = (fb.embed_F(t, torch.from_numpy(np.tile(om, (len(Pk), 1))).to(dev)) if args.tower == "F"
               else fb.embed_B(t)).cpu().numpy()
    D = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)   # unit directions -> cones on the sphere
    n_games = len(np.unique(gk))

    lab = HDBSCAN(min_cluster_size=args.min_cluster, metric="euclidean").fit_predict(D)
    cones = [c for c in np.unique(lab) if c >= 0]
    print(f"[stage] {len(Pk)} positions from {n_games} games, tower={args.tower} -> {len(cones)} density cones "
          f"({100*(lab<0).mean():.0f}% left as non-concept) ({time.time()-t0:.0f}s)", flush=True)

    feats = [named_features(board_from_packed(Pk[i], Mk[i])) for i in range(len(Pk))]
    fnames = [n for n in feats[0] if not n.endswith("_ctrl")]
    Fmat = np.array([[float(f[n][0]) for n in fnames] for f in feats])

    rows = []
    for c in cones:
        mem = np.flatnonzero(lab == c)
        center = D[mem].mean(0); center /= (np.linalg.norm(center) + 1e-9)
        halfangle = float(np.degrees(np.arccos(np.clip(D[mem] @ center, -1, 1)).mean()))  # cone half-width
        pathdens = len(np.unique(gk[mem])) / n_games                                     # PATH density
        cors = [abs(np.corrcoef((lab == c).astype(float), Fmat[:, j])[0, 1]) for j in range(len(fnames))]
        jm = int(np.argmax(cors)); named = f"{fnames[jm].replace('_w','')}({cors[jm]:.2f})" if cors[jm] > 0.30 else "novel"
        rows.append((c, len(mem), pathdens, halfangle, named, mem))

    rows.sort(key=lambda r: -r[2])                                  # rank by PATH density
    print(f"VERDICT CONCEPT_CONES field={Path(args.field).stem} tower={args.tower} cones={len(cones)}")
    print(f"  {'cone':>4s} {'size':>5s} {'path%':>6s} {'cone_deg':>8s}  {'mirror':16s} native-heatmap")
    for c, sz, pd, ha, named, mem in rows[:14]:
        order = mem[np.argsort(-(D[mem] @ (D[mem].mean(0) / (np.linalg.norm(D[mem].mean(0)) + 1e-9))))]
        print(f"  {c:>4d} {sz:>5d} {100*pd:>5.1f}% {ha:>7.1f}   {named:16s} {heatmap(Pk, Mk, order)}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
