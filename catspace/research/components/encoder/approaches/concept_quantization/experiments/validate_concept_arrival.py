#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/concept_quantization/experiments/validate_concept_arrival.py -- does the value field predict WHERE YOU END UP, in concept
terms? (Kaposi's thesis: F routes to a destination; concepts predict arrival.) Closes the loop:

  * discover concept atoms with the maintained TopK SAE (dictionary_learning) on real positions;
  * for each MIDGAME source position, its ARRIVAL = the game's position ~horizon plies later; its
    arrival concept = the arrival's dominant (argmax) SAE atom;
  * TEST: can a linear readout of the SOURCE's F predict the arrival concept, above baseline?

High accuracy => F genuinely predicts the concept you converge to, not just your current one. A
measurement, no new architecture.
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
from sklearn.model_selection import cross_val_score


from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from dictionary_learning.trainers import TopKTrainer
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default=paths.sep("lichess_gn_iqeqrl_full.pt"))
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--dict", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--min-ply", type=int, default=16)
    ap.add_argument("--max-ply", type=int, default=44)
    ap.add_argument("--top-concepts", type=int, default=12, help="restrict to the most common arrival concepts")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    nz = np.load(args.shard)
    P, M, ply, gid = (np.asarray(nz["packed"]), np.asarray(nz["meta"]),
                      np.asarray(nz["ply"]).astype(int), np.asarray(nz["game_id"]))
    n = len(gid)
    change = np.flatnonzero(np.diff(gid)) + 1
    last_of = np.repeat(np.concatenate([change, [n]]) - 1, np.diff(np.concatenate([[0], change, [n]])))
    cand = np.flatnonzero((ply >= args.min_ply) & (ply <= args.max_ply) & (last_of - np.arange(n) >= 8))
    src = cand[rng.permutation(len(cand))[:args.n]]
    arr = np.minimum(src + args.horizon, last_of[src])

    def embF(rows):
        out = []
        with torch.no_grad():
            for s in range(0, len(rows), 4096):
                e = min(len(rows), s + 4096)
                t = torch.from_numpy(feature_planes(P[rows[s:e]], M[rows[s:e]])).to(dev)
                out.append(fb.embed_F(t, torch.from_numpy(np.tile(om, (e - s, 1))).to(dev)).cpu().numpy())
        return np.concatenate(out)

    Fsrc, Farr = embF(src), embF(arr)
    print(f"[stage] {len(src)} source->arrival pairs (ply {args.min_ply}-{args.max_ply}, +{args.horizon}) "
          f"({time.time()-t0:.0f}s)", flush=True)

    # maintained SAE on all embeddings -> concept codes
    allF = np.concatenate([Fsrc, Farr]); mu, sd = allF.mean(0), allF.std(0) + 1e-8
    X = torch.from_numpy((allF - mu) / sd).float().to(dev)
    tr = TopKTrainer(steps=args.steps, activation_dim=X.shape[1], dict_size=args.dict, k=args.k, layer=0,
                     lm_name="catspace", device=dev, warmup_steps=max(1, args.steps // 10), seed=args.seed)
    for step in range(args.steps):
        tr.update(step, X[torch.from_numpy(rng.integers(0, len(X), size=1024)).to(dev)])
    with torch.no_grad():
        code = tr.ae.encode(X).cpu().numpy()
    arr_code = code[len(Fsrc):]
    arr_concept = arr_code.argmax(1)                              # arrival's dominant concept atom

    # restrict to the most common arrival concepts for a clean readout
    top = [c for c, _ in Counter(arr_concept).most_common(args.top_concepts)]
    keep = np.isin(arr_concept, top)
    y = np.array([top.index(c) for c in arr_concept[keep]])
    Xsrc = Fsrc[keep]                                             # predict from SOURCE F
    Xsrc_concept = code[:len(Fsrc)][keep].argmax(1)              # source's OWN dominant concept (persistence baseline)

    acc_src = cross_val_score(LogisticRegression(max_iter=300), Xsrc, y, cv=4).mean()
    # persistence baseline: predict arrival concept = source concept (does F ADD beyond "stay put"?)
    persist = float(np.mean([top[yy] == sc for yy, sc in zip(y, Xsrc_concept)]))
    majority = Counter(y).most_common(1)[0][1] / len(y)
    print(f"VERDICT CONCEPT_ARRIVAL field={Path(args.field).stem} n={len(y)} concepts={len(top)} horizon={args.horizon}")
    print(f"  predict arrival concept from SOURCE F (logreg CV): {acc_src:.3f}")
    print(f"  baselines: chance {1/len(top):.3f} | majority {majority:.3f} | persistence(src==arr) {persist:.3f}")
    print(f"  -> F {'PREDICTS convergence beyond chance/persistence' if acc_src > max(majority, persist) + 0.03 else 'does not clearly beat baselines'}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
