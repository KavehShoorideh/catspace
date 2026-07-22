#!/usr/bin/env python
"""experiments/goal_clusters.py -- B-clusters as GOALS/SUBGOALS, and does F predict arrival in them?
(Kaposi 2026-07-21). The FB field factors occupancy as score(s,g) = -d(F(s), B(g)): F is the source
tower ("where I'm going"), B is the goal tower ("what I am as a destination"). So:

  * cluster the B embeddings -> destination zones = the GOAL / SUBGOAL alphabet (deep clusters = endgame
    goals, intermediate = midgame subgoals);
  * a position's FIELD-NATIVE arrival prediction is argmin_k d(F(s), B_k) -- no new model, just the
    field read out;
  * VALIDATION: take real games, and for each midgame position s ask whether that field prediction
    matches the B-cluster of where the game ACTUALLY is ~horizon plies later. High accuracy => F
    predicts where you end up => B-clusters are real convergence-concepts. This is the falsifiable core.

Also fits a logistic F->arrival-cluster classifier (interpretable: which F-directions predict each goal).
Field-free of any labels -- the goals are discovered from B, the arrivals are ground truth from games.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device


def embed(fb, dev, P, M, om, which, block=2048):
    out = []
    with torch.no_grad():
        for s in range(0, len(P), block):
            e = min(len(P), s + block)
            t = torch.from_numpy(feature_planes(P[s:e], M[s:e])).to(dev)   # lichess field: FULL planes
            if which == "F":
                out.append(fb.embed_F(t, torch.from_numpy(np.tile(om, (e - s, 1))).to(dev)))
            else:
                out.append(fb.embed_B(t))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="data/derived/sep/lichess_gn_iqeqrl_full.pt")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=6000, help="source midgame positions")
    ap.add_argument("--k-goals", type=int, default=20, help="B-clusters (goals/subgoals)")
    ap.add_argument("--min-ply", type=int, default=16)
    ap.add_argument("--max-ply", type=int, default=40, help="source positions in [min,max] ply")
    ap.add_argument("--horizon", type=int, default=40, help="plies from source to the 'arrival' position")
    ap.add_argument("--out", default="artifacts/experiments/goal_clusters")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device)
    rng = np.random.default_rng(args.seed)

    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    nz = np.load(args.shard)
    P, M = np.asarray(nz["packed"]), np.asarray(nz["meta"])
    ply, gid = np.asarray(nz["ply"]).astype(int), np.asarray(nz["game_id"])
    n = len(gid)
    # last row of each game (rows are game-grouped, ply-ordered)
    change = np.flatnonzero(np.diff(gid)) + 1
    last_of = np.repeat(np.concatenate([change, [n]]) - 1, np.diff(np.concatenate([[0], change, [n]])))

    # source rows = midgame positions that still have >= a bit of future
    cand = np.flatnonzero((ply >= args.min_ply) & (ply <= args.max_ply) & (last_of - np.arange(n) >= 8))
    src = cand[rng.permutation(len(cand))[:args.n]]
    arr = np.minimum(src + args.horizon, last_of[src])                 # arrival ~horizon plies later (capped at game end)
    print(f"[stage] {len(src)} source positions (ply {args.min_ply}-{args.max_ply}), horizon {args.horizon} "
          f"({time.time()-t0:.0f}s)", flush=True)

    Fsrc = embed(fb, dev, P[src], M[src], om, "F")
    Barr = embed(fb, dev, P[arr], M[arr], om, "B")                     # B of the ARRIVAL positions
    Bsrc = embed(fb, dev, P[src], M[src], om, "B")

    # cluster the goal/subgoal space from B of arrivals+sources (broad -> goals + subgoals)
    Ball = torch.nn.functional.normalize(torch.cat([Barr, Bsrc]), dim=1).cpu().numpy()
    km = KMeans(n_clusters=args.k_goals, n_init=6, random_state=args.seed).fit(Ball)
    arr_cluster = km.labels_[:len(src)]                               # actual arrival cluster (ground truth)

    # FIELD-NATIVE prediction: nearest goal-cluster centroid under the quasimetric d(F(s), B_k)
    B_centroids = torch.from_numpy(km.cluster_centers_).float().to(dev)
    with torch.no_grad():
        Dfk = fb.distance_matrix(Fsrc, B_centroids).cpu().numpy()      # (n, k) F->goal-centroid
    pred_field = Dfk.argmin(1)
    acc_field = float((pred_field == arr_cluster).mean())

    # learned classifier F -> arrival cluster (interpretable routing)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    X = Fsrc.cpu().numpy(); y = arr_cluster
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=args.seed)
    clf = LogisticRegression(max_iter=300, C=1.0).fit(Xtr, ytr)
    acc_learned = float(clf.score(Xte, yte))

    sizes = np.bincount(arr_cluster, minlength=args.k_goals)
    base_major = float(sizes.max() / sizes.sum())
    print(f"VERDICT GOAL_CLUSTERS field={Path(args.field).stem} n={len(src)} k_goals={args.k_goals} "
          f"horizon={args.horizon}")
    print(f"  F->arrival-cluster accuracy: field-native(argmin d) {acc_field:.3f} | "
          f"learned(logreg on F) {acc_learned:.3f}  vs baselines: chance {1/args.k_goals:.3f} / "
          f"majority {base_major:.3f}")
    print(f"  goal-cluster sizes (arrivals): {sizes.tolist()}")
    # what ARE the goal clusters? dominant material of the arrival positions in each (structural or value-generic?)
    from collections import Counter

    def matsig(pk, mt):
        return "".join(sorted(p.symbol() for p in board_from_packed(pk, mt).piece_map().values()))
    print("  goal-cluster identity (dominant material of arrivals):")
    for c in np.argsort(-sizes)[:8]:
        idx = np.flatnonzero(arr_cluster == c)
        mats = [matsig(P[arr[i]], M[arr[i]]) for i in idx[:250]]
        top = Counter(mats).most_common(1)[0]
        print(f"    C{c:2d} (n={sizes[c]:4d}): dom-material '{top[0]}' {100*top[1]/len(mats):.0f}%  "
              f"(distinct materials {len(set(mats))})")

    _viz(Barr, Fsrc, arr_cluster, pred_field, args)
    print(f"[done] {time.time()-t0:.0f}s")


def _viz(Barr, Fsrc, arr_cluster, pred_field, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    Bn = torch.nn.functional.normalize(Barr, dim=1).cpu().numpy()
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=args.seed).fit_transform(Bn)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    sc0 = ax[0].scatter(emb[:, 0], emb[:, 1], s=7, c=arr_cluster, cmap="tab20", alpha=0.6)
    ax[0].set_title("B(arrival) t-SNE, colored by GOAL cluster (the goal/subgoal alphabet)")
    correct = pred_field == arr_cluster
    ax[1].scatter(emb[correct, 0], emb[correct, 1], s=7, c="tab:green", alpha=0.5, label="F predicted arrival")
    ax[1].scatter(emb[~correct, 0], emb[~correct, 1], s=7, c="tab:red", alpha=0.4, label="F missed")
    ax[1].set_title("does F(source) predict the arrival goal-cluster?"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = Path(args.out).with_suffix(".png"); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"  figure -> {out}")


if __name__ == "__main__":
    main()
