#!/usr/bin/env python
"""experiments/rank_sensitivity.py -- diagnosis + weight sensitivity for the 1-ply STEP
gradient (Kaveh). Fixes the broken rank loss found in diagnosis:
  * DROP TIES: only pair siblings whose |DTZ| actually differs (39% were ties -> y=0 ->
    margin_ranking_loss returned a constant margin that floored the loss ~0.34, misread as a
    wall). The metric also counted ties as errors (capped ~61%).
  * MARGIN-FREE logistic ordering: loss = softplus(d_closer - d_farther) = -log sigmoid(
    d_farther - d_closer). No arbitrary margin (the old 0.5 was 5x the true ~0.10 log gap).
Then SWEEP the step weight w_rank and report, on DISTINGUISHABLE endgame siblings only:
  rank-acc (does it order 1-ply-apart moves correctly?), d-vs-DTZ, won/INF separation.
"""
from __future__ import annotations

import argparse, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
from experiments.arch_bakeoff import eff_rank, tokens
from experiments.train_mate_field import MateField


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/derived/child_rank_v1.npz")
    ap.add_argument("--w-rank", type=float, nargs="*", default=[0.0, 1.0, 4.0, 16.0])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--rank-pairs", type=int, default=512)
    ap.add_argument("--margin-obj", type=float, default=400.0)
    ap.add_argument("--device", default="auto"); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = pick_device(args.device)
    z = np.load(args.data)
    ids, stm = tokens(z["packed"], z["meta"]); dtz = z["dtz"].astype(np.float32); grp = z["group"]
    Ci = torch.from_numpy(ids.astype(np.int64)); Cs = torch.from_numpy(stm.astype(np.int64))
    won = dtz >= 0; inf = dtz < 0
    idx_won = np.flatnonzero(won); idx_inf = np.flatnonzero(inf)
    tgt = torch.from_numpy(np.where(won, np.log1p(np.clip(dtz,0,None)), 0.0).astype(np.float32)).to(dev)
    logM = float(np.log1p(args.margin_obj))
    # DISTINGUISHABLE sibling pairs only (drop ties)
    g2 = defaultdict(list)
    for i in idx_won: g2[grp[i]].append(i)
    pairs = []                                   # (closer_idx, farther_idx) with dtz_closer < dtz_farther
    for v in g2.values():
        v = np.array(v)
        if len(v) < 2: continue
        for _ in range(min(6, len(v))):          # a few distinguishable pairs per group
            a, b = np.random.default_rng(int(v[0])).integers(0, len(v), 2) if False else (np.random.randint(len(v)), np.random.randint(len(v)))
            if dtz[v[a]] == dtz[v[b]]: continue
            lo, hi = (v[a], v[b]) if dtz[v[a]] < dtz[v[b]] else (v[b], v[a])
            pairs.append((lo, hi))
    pairs = np.array(pairs)
    print(f"[rank-sens] won {len(idx_won)} inf {len(idx_inf)} | distinguishable pairs {len(pairs)}", flush=True)

    def run(w_rank):
        torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
        net = MateField(32, 64, 6, 16).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
        hb = args.batch // 2
        for s in range(args.steps):
            bw = idx_won[rng.integers(0, len(idx_won), hb)]; bi = idx_inf[rng.integers(0, len(idx_inf), hb)]
            bb = np.concatenate([bw, bi]); dm = net.d_to_mate(Ci[bb].to(dev), Cs[bb].to(dev))
            dml = torch.log1p(dm.clamp(min=0)); wm = torch.from_numpy((dtz[bb]>=0).astype(np.float32)).to(dev)
            reg = (F.huber_loss(dml, tgt[bb], reduction="none")*wm).sum()/wm.sum().clamp(min=1)
            hinge = (F.relu(logM-dml)*(1-wm)).sum()/(1-wm).sum().clamp(min=1)
            loss = reg + hinge
            if w_rank > 0:
                pp = pairs[rng.integers(0, len(pairs), args.rank_pairs)]
                dlo = torch.log1p(net.d_to_mate(Ci[pp[:,0]].to(dev), Cs[pp[:,0]].to(dev)).clamp(min=0))
                dhi = torch.log1p(net.d_to_mate(Ci[pp[:,1]].to(dev), Cs[pp[:,1]].to(dev)).clamp(min=0))
                gap = torch.from_numpy((np.log1p(dtz[pp[:,1]]) - np.log1p(dtz[pp[:,0]])).astype(np.float32)).to(dev)
                rank = F.relu(gap - (dhi - dlo)).mean()                          # ANCHORED: enforce true log-gap
                loss = loss + w_rank * rank
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            te = pairs[rng.integers(0, len(pairs), 8000)]
            dlo = net.d_to_mate(Ci[te[:,0]].to(dev), Cs[te[:,0]].to(dev)).cpu().numpy()
            dhi = net.d_to_mate(Ci[te[:,1]].to(dev), Cs[te[:,1]].to(dev)).cpu().numpy()
            racc = float((dlo < dhi).mean())*100                                # correct 1-ply order
            sub = idx_won[rng.integers(0, len(idx_won), 3000)]
            er = eff_rank(net.phi(Ci[sub].to(dev), Cs[sub].to(dev)).cpu().numpy())
            dd = net.d_to_mate(Ci[sub].to(dev), Cs[sub].to(dev)).cpu().numpy()
            si = idx_inf[rng.integers(0, len(idx_inf), 3000)]
            di = net.d_to_mate(Ci[si].to(dev), Cs[si].to(dev)).cpu().numpy()
        from scipy.stats import spearmanr
        return racc, float(spearmanr(dd, dtz[sub]).correlation), er, float(np.median(dd)), float(np.median(di))

    t0 = time.time()
    print("w_rank | 1-ply RANK-ACC(distinct) | d-vs-DTZ | eff_rank | won-d | INF-d", flush=True)
    for w in args.w_rank:
        racc, sp, er, wd, idd = run(w)
        print(f"  {w:5.1f} |   {racc:5.1f}%   |  {sp:+.3f} |  {er:4.1f}  | {wd:5.1f} | {idd:6.1f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    print("DONE rank_sensitivity", flush=True)


if __name__ == "__main__":
    main()
