#!/usr/bin/env python
"""experiments/train_field_full.py -- the CONSOLIDATED best-candidate field (Kaveh: include ALL
corrections that prevent collapse/problems, don't leave anything out). Single-space IQE field phi
trained with EVERY correction we've found:

  1. SINGLE-SPACE IQE (one encoder + one head): composable, triangle inequality structural
     (two-tower had 10.9% violations -> 0.00%).
  2. MULTI-GOAL / triangulation: d(phi(s),phi(g)) -> Delta on pairwise data. Multilateration
     keeps effective rank up (single MATE-goal scalar collapsed it to 1.7; multi-goal kept 6.3).
  3. REPULSION anti-collapse: push random non-adjacent pairs apart (hinge min-distance) -- cure
     for collapse is repulsion, not width (memory: check-representational-collapse).
  4. LOCAL move-resolution: within-sibling |DTZ| pairwise rank loss (fixes the 52.7% coin-flip).
  5. WDL infinite BARRIERS: draw/loss -> hinge d_mate UP to log1p(M) (bounded, not divergent) --
     the stalemate/draw-interface repeller.
  6. MATE collapsed ATTRACTOR: learnable MATE goal, d_mate->0 at mate (sharp region readout).
  7. BOTH-COLOR coverage (pairwise lines + black-to-move children).
Effective-rank is a first-class health gate on the verdict. Tensor-batched; low dim (d=32) until
rank saturates, then raise. Value = d_mate for the search planner (policy = planner, not greedy).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.nn.fb import pick_device
from catspace.nn.iqe import IQE
from experiments.arch_bakeoff import CNNBackbone, eff_rank, tokens


class FullField(nn.Module):
    def __init__(self, d=32, d_bb=64, blocks=6, iqe_components=16):
        super().__init__()
        self.enc = CNNBackbone(d_bb, blocks)
        self.head = nn.Sequential(nn.Linear(d_bb, d_bb), nn.GELU(), nn.Linear(d_bb, d))
        self.iqe = IQE(d, components=iqe_components)
        self.mate = nn.Parameter(torch.randn(d) * 0.1)

    def phi(self, ids, stm):
        _, pooled = self.enc(ids, stm)
        return self.head(pooled)

    def d_pair(self, se, ge):                       # both already embedded
        return self.iqe(se, ge)

    def d_mate(self, ids, stm):
        e = self.phi(ids, stm)
        return self.iqe(e, self.mate.expand_as(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", default="data/derived/pairwise_tb_v1.npz")
    ap.add_argument("--child", default="data/derived/child_rank_v1.npz")
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--d-bb", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--iqe-components", type=int, default=16)
    ap.add_argument("--margin", type=float, default=400.0)
    ap.add_argument("--w-pair", type=float, default=1.0)
    ap.add_argument("--w-mate", type=float, default=1.0)
    ap.add_argument("--w-hinge", type=float, default=1.0)
    ap.add_argument("--w-rank", type=float, default=1.0)
    ap.add_argument("--w-repel", type=float, default=0.3)
    ap.add_argument("--repel-margin", type=float, default=3.0, help="push random pairs >= this")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--rank-pairs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save", default="artifacts/experiments/field_full_v1.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    logM = float(np.log1p(args.margin))

    # ---- pairwise (multi-goal geometry / rank) ----
    zp = np.load(args.pair)
    ps_ids, ps_stm = tokens(zp["s_packed"], zp["s_meta"])
    pg_ids, pg_stm = tokens(zp["g_packed"], zp["g_meta"])
    p_delta = torch.from_numpy(np.log1p(zp["delta"].astype(np.float32))).to(dev)
    PSi = torch.from_numpy(ps_ids.astype(np.int64)); PSs = torch.from_numpy(ps_stm.astype(np.int64))
    PGi = torch.from_numpy(pg_ids.astype(np.int64)); PGs = torch.from_numpy(pg_stm.astype(np.int64))
    nP = len(p_delta)

    # ---- child (mate attractor + WDL barrier + local rank + both colors) ----
    zc = np.load(args.child)
    c_ids, c_stm = tokens(zc["packed"], zc["meta"]); dtz = zc["dtz"].astype(np.float32); grp = zc["group"]
    Ci = torch.from_numpy(c_ids.astype(np.int64)); Cs = torch.from_numpy(c_stm.astype(np.int64))
    won = dtz >= 0; inf = dtz < 0
    c_tgt = torch.from_numpy(np.where(won, np.log1p(np.clip(dtz, 0, None)), 0.0).astype(np.float32)).to(dev)
    idx_won = np.flatnonzero(won); idx_inf = np.flatnonzero(inf)
    g2c = defaultdict(list)
    for i in idx_won:
        g2c[grp[i]].append(i)
    rank_groups = [np.array(v) for v in g2c.values() if len(v) >= 2]
    print(f"[field-full] pairs {nP} | child won {len(idx_won)} inf {len(idx_inf)} | "
          f"rank-groups {len(rank_groups)} | d{args.d} logM {logM:.2f}", flush=True)

    net = FullField(args.d, args.d_bb, args.blocks, args.iqe_components).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    def rank_pairs(k):
        gi = rng.integers(0, len(rank_groups), k); a = np.empty(k, np.int64); b = np.empty(k, np.int64)
        for t, g in enumerate(gi):
            cs = rank_groups[g]; ii = rng.integers(0, len(cs)); jj = rng.integers(0, len(cs))
            while jj == ii: jj = rng.integers(0, len(cs))
            a[t], b[t] = cs[ii], cs[jj]
        return a, b

    hb = args.batch // 2
    for s in range(args.steps):
        # (2) pairwise geometry
        pb = rng.integers(0, nP, args.batch)
        se = net.phi(PSi[pb].to(dev), PSs[pb].to(dev)); ge = net.phi(PGi[pb].to(dev), PGs[pb].to(dev))
        L_pair = F.huber_loss(torch.log1p(net.d_pair(se, ge).clamp(min=0)), p_delta[pb], delta=1.0)
        # (3) repulsion: push random (s, other-g) pairs apart (anti-collapse)
        perm = torch.randperm(args.batch, device=dev)
        L_repel = F.relu(args.repel_margin - torch.log1p(net.d_pair(se, ge[perm]).clamp(min=0))).mean()
        # (6)/(5) mate attractor + WDL hinge (both colors via child data)
        bw = idx_won[rng.integers(0, len(idx_won), hb)]
        bi = idx_inf[rng.integers(0, len(idx_inf), hb)]
        bb = np.concatenate([bw, bi]); dm = net.d_mate(Ci[bb].to(dev), Cs[bb].to(dev))
        dml = torch.log1p(dm.clamp(min=0)); wmask = torch.from_numpy((dtz[bb] >= 0).astype(np.float32)).to(dev)
        L_mate = (F.huber_loss(dml, c_tgt[bb], reduction="none") * wmask).sum() / wmask.sum().clamp(min=1)
        L_hinge = (F.relu(logM - dml) * (1 - wmask)).sum() / (1 - wmask).sum().clamp(min=1)
        # (4) local rank
        pa, pbb = rank_pairs(args.rank_pairs)
        da = net.d_mate(Ci[pa].to(dev), Cs[pa].to(dev)); db = net.d_mate(Ci[pbb].to(dev), Cs[pbb].to(dev))
        y = torch.from_numpy(np.sign(dtz[pbb] - dtz[pa]).astype(np.float32)).to(dev)
        L_rank = F.margin_ranking_loss(db, da, y, margin=0.5)
        loss = (args.w_pair * L_pair + args.w_repel * L_repel + args.w_mate * L_mate
                + args.w_hinge * L_hinge + args.w_rank * L_rank)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 2500 == 0:
            print(f"  step {s} pair {float(L_pair):.3f} repel {float(L_repel):.3f} mate "
                  f"{float(L_mate):.3f} hinge {float(L_hinge):.3f} rank {float(L_rank):.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    net.eval()
    with torch.no_grad():
        sub = idx_won[rng.integers(0, len(idx_won), min(3000, len(idx_won)))]
        er = eff_rank(net.phi(Ci[sub].to(dev), Cs[sub].to(dev)).cpu().numpy())
        dd = net.d_mate(Ci[sub].to(dev), Cs[sub].to(dev)).cpu().numpy()
        si = idx_inf[rng.integers(0, len(idx_inf), min(3000, len(idx_inf)))]
        di = net.d_mate(Ci[si].to(dev), Cs[si].to(dev)).cpu().numpy()
    from scipy.stats import spearmanr
    sp = float(spearmanr(dd, dtz[sub]).correlation)
    pa, pbb = rank_pairs(4000)
    with torch.no_grad():
        da = net.d_mate(Ci[pa].to(dev), Cs[pa].to(dev)).cpu().numpy()
        db = net.d_mate(Ci[pbb].to(dev), Cs[pbb].to(dev)).cpu().numpy()
    racc = float((np.sign(db - da) == np.sign(dtz[pbb] - dtz[pa])).mean()) * 100
    print(f"VERDICT FIELD-FULL d{args.d}: eff_rank {er:.1f} (single-goal was 1.7) | d-vs-DTZ "
          f"{sp:+.3f} | within-sibling RANK-ACC {racc:.1f}% (coin 50) | won-d {np.median(dd):.1f} "
          f"vs INF-d {np.median(di):.1f} | [{time.time()-t0:.0f}s]", flush=True)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "model": "FullField",
                    "cfg": {"d": args.d, "d_bb": args.d_bb, "blocks": args.blocks,
                            "iqe_components": args.iqe_components},
                    "metrics": {"eff_rank": er, "d_vs_dtz": sp, "rank_acc": racc}}, args.save)
        print(f"  saved -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
