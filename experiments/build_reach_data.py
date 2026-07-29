#!/usr/bin/env python
"""experiments/build_reach_data.py -- v1 dataset for the z-conditioned first-hit reachability head
(REACHABILITY_FOUNDATIONS §4.1 / §6.1).

From the M2b dense cache (frozen-trunk phi already precomputed per decision point), build:
  * a GOAL BANK: k-means centroids of train-split phi (the v1 stand-in for the atlas bank);
  * per (position, goal) FIRST-HIT labels within the game: hit=1 iff some STRICTLY LATER decision
    point of the same game lands inside goal g's eps-ball; plies = ply gap to the first such point,
    -1 when censored (game ends without a hit). Positions already inside g are flagged
    (in_now) and excluded from training pairs (trivial reach).
  * eps rule (pre-registered): eps = median distance from train positions to their NEAREST
    centroid. Audit prints (TRAINING_STANDARDS 15) must show a non-degenerate hit base rate.

Caveat recorded: trajectories here are ONE player's decision points (every 2nd ply), so "first
hit" is observed on that subsampled trajectory -- adequate for v1; the v2 rebuild from full
records adds both sides + clocks (c_t).

Output: data/derived/reach/reach_v1.npz  (hit uint8 [N,G], plies int16 [N,G], in_now uint8 [N,G],
bank float32 [G,64], eps, and the per-position metadata needed by the trainer). Materialized per
TRAINING_STANDARDS 16; DVC-track after build.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.style.dataio import load_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/derived/m2b/cache_dense")
    ap.add_argument("--out", default="data/derived/reach/reach_v1.npz")
    ap.add_argument("--goals", type=int, default=256)
    ap.add_argument("--eps-quantile", type=float, default=0.5,
                    help="eps = this quantile of train nearest-centroid distance (pre-registered 0.5)")
    ap.add_argument("--aug", default="", help="aug_feats npz -> cluster/assign in phi⊕feats space")
    ap.add_argument("--aug-w", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    c = load_cache(args.cache)
    phi = np.ascontiguousarray(c["phi"], dtype=np.float32)          # (N,64)
    N = len(phi)
    split = np.asarray(c["split"]).astype(str)
    train_mask = split == "train"
    print(f"cache: {N} positions | train {train_mask.sum()} | heldout {(~train_mask).sum()}")

    # ---- goal bank: k-means on TRAIN rows only (bank must not peek at heldout) ----
    from sklearn.cluster import KMeans
    if args.aug:
        # AUGMENTED CODEBOOK (fired eyes-gate 2026-07-30): cluster + assign in standardized
        # [phi ⊕ aug_w*feats]; the goal TOWER keeps eating raw phi -> tower bank = per-cluster
        # mean raw phi (decoupling: assignment space != tower space).
        feats = np.load(args.aug)["feats"].astype(np.float32)
        mu_p, sd_p = phi[train_mask].mean(0), phi[train_mask].std(0) + 1e-9
        mu_f, sd_f = feats[train_mask].mean(0), feats[train_mask].std(0) + 1e-9
        A = np.concatenate([(phi - mu_p) / sd_p, args.aug_w * (feats - mu_f) / sd_f], 1)
        km = KMeans(n_clusters=args.goals, n_init=3, random_state=args.seed)
        km.fit(A[train_mask])
        assign_bank = km.cluster_centers_.astype(np.float32)
        lab = km.predict(A)
        bank = np.stack([phi[lab == g].mean(0) if (lab == g).any() else phi[train_mask].mean(0)
                         for g in range(args.goals)]).astype(np.float32)
        phi_assign = A                                               # membership space
    else:
        km = KMeans(n_clusters=args.goals, n_init=3, random_state=args.seed)
        km.fit(phi[train_mask])
        bank = km.cluster_centers_.astype(np.float32)                # (G,64)
        assign_bank, phi_assign = bank, phi
        mu_p = sd_p = mu_f = sd_f = None

    # ---- distances position->goal, TWO-PASS chunked (never hold the (N,G) float matrix:
    # at v2 scale that is >4 GB; pass 1 = nearest-centroid dist for eps, pass 2 = membership) ----
    G = args.goals
    b2 = (assign_bank * assign_bank).sum(1)[None, :]

    def dchunk(i):
        x = phi_assign[i:i + 8192]
        return np.sqrt(np.maximum((x * x).sum(1)[:, None] + b2 - 2.0 * x @ assign_bank.T, 0.0))

    near = np.empty(N, dtype=np.float32)
    for i in range(0, N, 8192):
        near[i:i + 8192] = dchunk(i).min(1)
    eps = float(np.quantile(near[train_mask], args.eps_quantile))
    member = np.empty((N, G), dtype=bool)
    for i in range(0, N, 8192):
        member[i:i + 8192] = dchunk(i) <= eps

    # ---- first-hit labels per game (strictly later decision point) ----
    game = np.asarray(c["game_id"]); ply = np.asarray(c["ply"])
    order = np.lexsort((ply, game))
    hit = np.zeros((N, G), dtype=np.uint8)
    plies = np.full((N, G), -1, dtype=np.int16)
    starts = np.flatnonzero(np.r_[True, game[order][1:] != game[order][:-1]])
    bounds = np.r_[starts, len(order)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        idx = order[s:e]                                             # one game, ply-ascending
        nxt_ply = np.full(G, -1, dtype=np.int32)                     # backward sweep
        for j in range(len(idx) - 1, -1, -1):
            row = idx[j]
            got = nxt_ply >= 0
            hit[row, got] = 1
            plies[row, got] = np.minimum(nxt_ply[got] - ply[row], 32767)
            nxt_ply[member[row]] = ply[row]                          # this point serves LATER ones

    in_now = member.astype(np.uint8)

    # ---- audits (TRAINING_STANDARDS 15): printed, and gate the write ----
    valid = ~member                                                  # trainable pairs exclude in-now
    for name, m in (("train", train_mask), ("heldout", ~train_mask)):
        v = valid[m]; h = hit[m][v]
        pl = plies[m][valid[m] & (hit[m] == 1)]
        print(f"AUDIT [{name}]: pairs {v.sum():,} | hit rate {h.mean():.4f} | "
              f"plies med {np.median(pl):.0f} p90 {np.percentile(pl, 90):.0f}")
    tr_rate = float(hit[train_mask][valid[train_mask]].mean())
    assert 0.005 < tr_rate < 0.5, f"degenerate hit base rate {tr_rate:.4f} -- adjust eps rule"
    per_goal = hit[train_mask].mean(0)
    print(f"AUDIT per-goal hit rate: min {per_goal.min():.4f} med {np.median(per_goal):.4f} "
          f"max {per_goal.max():.4f} | goals with rate<0.001: {(per_goal < 0.001).sum()}/{G}")
    print(f"eps={eps:.4f} (q{args.eps_quantile} of train nearest-centroid dist)")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, hit=hit, plies=plies, in_now=in_now, bank=bank, eps=np.float32(eps),
        **({"assign_bank": assign_bank, "aug_mu_p": mu_p, "aug_sd_p": sd_p,
            "aug_mu_f": mu_f, "aug_sd_f": sd_f, "aug_w": args.aug_w} if args.aug else {}),
        phi=phi, pidx=np.asarray(c["pidx"], dtype=np.int32),
        elo_self=np.asarray(c["elo_self"], dtype=np.float32),
        elo_oppo=np.asarray(c["elo_oppo"], dtype=np.float32),
        split=split, game_id=game.astype(np.int64), ply=ply.astype(np.int32),
        player_id=np.asarray(c["player_id"]).astype(np.uint64),
        **({"result_mover": np.asarray(c["result_mover"], dtype=np.float32)}
           if "result_mover" in c else {}),
        meta_cache=str(args.cache), meta_goals=G, meta_seed=args.seed,
        meta_eps_quantile=args.eps_quantile)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
