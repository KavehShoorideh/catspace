#!/usr/bin/env python
"""
experiments/build_competence_map.py — build the Method-2 competence map
(catspace/competence.py) from a corpus of positions.

For each sampled position: embed it (F) and measure the engine's Method-1
reliability (FBSearchPolicy.reliability -- shallow-vs-deep disagreement). Store
(embedding, reliability) as the map. Then Method 2 can PREDICT reliability at a
new position cheaply (cosine kNN) without running the deep search.

Also reports a held-out validation: does the map's PREDICTED unreliability
correlate with the actual (held-out) Method-1 reliability? That's the honest
check that the competence field generalizes -- if it doesn't, Method 2 is just
memorizing and adds nothing over Method 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catspace.competence import CompetenceMap
from catspace.data.encode import board_from_packed
from catspace.data.shards import sample_shard_rows
from catspace.io.paths import derived_dir, newest_shard_dir


def load_positions(shard_dir, n, seed):
    import chess
    picks = sample_shard_rows(shard_dir, n * 2, seed, holdout_only=True)
    by_shard = {}
    for name, row in picks:
        by_shard.setdefault(name, []).append(row)
    boards = []
    for name, rows in by_shard.items():
        npz = np.load(Path(shard_dir) / name)
        packed, meta = npz["packed"], npz["meta"]
        for r in rows:
            b = board_from_packed(packed[r], meta[r])
            if not b.is_game_over() and len(list(b.legal_moves)) >= 3:
                boards.append(b)
            if len(boards) >= n:
                return boards
    return boards


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--max-nodes", type=int, default=200)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="data/derived/competence_map.npz")
    args = ap.parse_args()

    import torch  # noqa: F401
    from scipy.stats import spearmanr

    from catspace.data.encode import encode_meta, encode_packed
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.features import feature_planes, omega_ids
    from catspace.nn.policy_fb import FBSearchPolicy

    device = pick_device(args.device)
    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()
    fb, payload = load_ckpt(Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt", device)
    z = payload["zgoals"]["MATE_W"]
    pol = FBSearchPolicy(fb, z, max_nodes=args.max_nodes, beam=args.beam, device=device)

    boards = load_positions(shard_dir, args.n, args.seed)
    print(f"{len(boards)} positions; measuring Method-1 reliability + embedding...", flush=True)

    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    embs, rels = [], []
    for i, b in enumerate(boards):
        rels.append(pol.reliability(b))
        planes = torch.from_numpy(feature_planes(encode_packed(b)[None], encode_meta(b)[None])).to(device)
        with torch.no_grad():
            embs.append(fb.embed_F(planes, torch.from_numpy(om[None]).to(device))[0].cpu().numpy())
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(boards)}", flush=True)
    embs = np.stack(embs).astype(np.float32)
    rels = np.array(rels, dtype=np.float32)

    # held-out validation: train map on 80%, predict the 20% held out
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(embs))
    cut = int(0.8 * len(embs))
    tr, te = perm[:cut], perm[cut:]
    cmap = CompetenceMap(embs[tr], rels[tr], k=args.k)
    pred = cmap.query(embs[te])
    rho = spearmanr(pred, rels[te]).statistic
    print(f"HELD-OUT: rho(predicted unreliability, actual Method-1 reliability) = {rho:+.3f} "
          f"(n={len(te)}; >0 means the competence field generalizes)")

    CompetenceMap(embs, rels, k=args.k).save(args.out)
    print(f"saved {args.out}  (n={len(embs)}, k={args.k}, mean_reliability={rels.mean():.3f})")


if __name__ == "__main__":
    main()
