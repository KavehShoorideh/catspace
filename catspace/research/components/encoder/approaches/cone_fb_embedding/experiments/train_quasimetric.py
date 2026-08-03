#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/cone_fb_embedding/experiments/train_quasimetric.py -- MULTI-GOAL QUASIMETRIC field MVP (Kaveh 2026-07-26).

The scalar distance-to-mate field collapses to rank ~3 and ties out at far-spearman ~+0.2
(magnitude ok, ORDERING stuck) because a single-landmark (mate) scalar target only pins one
radial coordinate. Fix (Kaveh's triangulation): supervise d(F(s), F(g)) to MANY reachable
goals at mixed ranges, so the geometry is multilaterated -- full-rank, correctly ordered.
Two-tower IQE quasimetric (triangle inequality structural); strong opponent (tablebase
optimal) => labels are genuine shortest-path distances (quasimetric-safe). omega-free.

We HAVE ground-truth pairwise labels here (delta plies on the optimal line), so training is
SUPERVISED regression on log1p(delta) -- QRL's constraint objective is for the label-free
middlegame extension (Phase 2). Everything is tensor-batched.

Decisive validations vs the scalar baseline (bake-off far-spearman ceiling +0.2):
  1. effective rank of F-embeddings (bootstrapped) -- is the collapse cured?
  2. mate-via-min-over-region: d(s, MATE) = min over mate landmarks of d(F(s),F(m)),
     spearman/MAE vs true DTM -- does the region-as-min readout recover distance-to-mate?
  3. held-out pair ordering (spearman of d vs true delta) -- beats the +0.2 ceiling?
  4. triangle-inequality violation rate on sampled triples -- is it quasimetric-safe
     (the option-A-vs-B green light at strong opponent)?
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE
from catspace.research.components.encoder.approaches.reachability_field.experiments.arch_bakeoff import CNNBackbone, eff_rank, tokens
from catspace.io import paths


class TwoTowerIQE(nn.Module):
    """d(F(s), F(g)) = IQE(headF(encF(s)), headB(encB(g))). Asymmetric by construction
    (from-tower vs to-tower + IQE's directed interval union) -- a quasimetric."""
    def __init__(self, d=32, d_bb=64, blocks=6, iqe_components=16, shared=False):
        super().__init__()
        # shared=True => ONE embedding space phi (single encoder AND single head): d(x,y)=
        # IQE(phi(x),phi(y)). IQE itself supplies the asymmetry, so this is still a proper
        # asymmetric quasimetric -- but now the triangle inequality holds structurally (a
        # node b has ONE embedding, so composition a->b->c is valid). Two separate heads
        # (shared=False, or the earlier trunk-only sharing) give F(b)!=B(b) and break it.
        self.shared = shared
        self.encF = CNNBackbone(d_bb, blocks)
        self.encB = self.encF if shared else CNNBackbone(d_bb, blocks)
        self.headF = nn.Sequential(nn.Linear(d_bb, d_bb), nn.GELU(), nn.Linear(d_bb, d))
        self.headB = self.headF if shared else nn.Sequential(
            nn.Linear(d_bb, d_bb), nn.GELU(), nn.Linear(d_bb, d))
        self.iqe = IQE(d, components=iqe_components)

    def embedF(self, ids, stm):
        _, pooled = self.encF(ids, stm)
        return self.headF(pooled)

    def embedB(self, ids, stm):
        _, pooled = self.encB(ids, stm)
        return self.headB(pooled)

    def dist(self, sf, gf):                              # sf: F-embeddings, gf: B-embeddings
        return self.iqe(sf, gf)

    def forward(self, s_ids, s_stm, g_ids, g_stm):
        return self.dist(self.embedF(s_ids, s_stm), self.embedB(g_ids, g_stm))


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=paths.derived("pairwise_tb_v1.npz"))
    ap.add_argument("--dtm-data", default=paths.derived("dtm_endgame_v2.npz"),
                    help="held-out positions w/ true DTM for the mate-via-min readout test")
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--d-bb", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--iqe-components", type=int, default=16)
    ap.add_argument("--shared", action="store_true", help="share the F/B encoder")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save", default=None, help="path to save the trained field + metadata")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    z = np.load(args.data)
    s_ids, s_stm = tokens(z["s_packed"], z["s_meta"])
    g_ids, g_stm = tokens(z["g_packed"], z["g_meta"])
    delta = z["delta"].astype(np.float32)
    is_mate = z["is_mate"].astype(bool)
    n = len(delta)
    heldout = rng.random(n) < 0.1
    tr = np.flatnonzero(~heldout); te = np.flatnonzero(heldout)
    print(f"[quasimetric] {len(tr)} train / {len(te)} test pairs | "
          f"delta med {int(np.median(delta))} max {int(delta.max())} | "
          f"mate-goals {int(is_mate.sum())}", flush=True)

    S_ids = torch.from_numpy(s_ids.astype(np.int64)); S_stm = torch.from_numpy(s_stm.astype(np.int64))
    G_ids = torch.from_numpy(g_ids.astype(np.int64)); G_stm = torch.from_numpy(g_stm.astype(np.int64))
    tgt = torch.from_numpy(np.log1p(delta)).to(dev)

    net = TwoTowerIQE(args.d, args.d_bb, args.blocks, args.iqe_components, args.shared).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    for s in range(args.steps):
        bi = tr[rng.integers(0, len(tr), args.batch)]
        d = net(S_ids[bi].to(dev), S_stm[bi].to(dev), G_ids[bi].to(dev), G_stm[bi].to(dev))
        loss = F.huber_loss(torch.log1p(d.clamp(min=0)), tgt[bi], delta=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 2000 == 0:
            print(f"  step {s} loss {float(loss):.4f} [{time.time()-t0:.0f}s]", flush=True)

    net.eval()

    @torch.no_grad()
    def pred_pairs(idx):
        out = []
        for k in range(0, len(idx), 4096):
            j = idx[k:k + 4096]
            out.append(net(S_ids[j].to(dev), S_stm[j].to(dev),
                           G_ids[j].to(dev), G_stm[j].to(dev)).cpu().numpy())
        return np.concatenate(out)

    # (3) held-out pair ordering
    dp = pred_pairs(te)
    sp_pair = spearman(dp, delta[te])
    mae_pair = float(np.abs(dp - delta[te]).mean())

    # (1) effective rank of F-embeddings on held-out sources (bootstrapped)
    @torch.no_grad()
    def embF(idx):
        out = []
        for k in range(0, len(idx), 4096):
            j = idx[k:k + 4096]
            out.append(net.embedF(S_ids[j].to(dev), S_stm[j].to(dev)).cpu().numpy())
        return np.concatenate(out)
    Fte = embF(te[:3000])
    ranks = [eff_rank(Fte[rng.integers(0, len(Fte), len(Fte))]) for _ in range(8)]
    er_mean, er_sd = float(np.mean(ranks)), float(np.std(ranks))

    # (4) triangle-inequality violation rate on sampled triples a->c vs a->b->c
    @torch.no_grad()
    def dist_ab(a_ids, a_stm, b_ids, b_stm):
        return net(a_ids.to(dev), a_stm.to(dev), b_ids.to(dev), b_stm.to(dev)).cpu().numpy()
    T = 4000
    a = te[rng.integers(0, len(te), T)]; b = te[rng.integers(0, len(te), T)]; c = te[rng.integers(0, len(te), T)]
    # use the SOURCE board of each pair as the node identity
    d_ac = dist_ab(S_ids[a], S_stm[a], S_ids[c], S_stm[c])
    d_ab = dist_ab(S_ids[a], S_stm[a], S_ids[b], S_stm[b])
    d_bc = dist_ab(S_ids[b], S_stm[b], S_ids[c], S_stm[c])
    slack = d_ac - (d_ab + d_bc)                          # <=0 if triangle holds
    viol = float((slack > 1e-3).mean())
    viol_mag = float(np.clip(slack, 0, None).mean())

    # (2) mate-via-min-over-region: d(s, MATE) = min over mate landmarks of d(F(s),F(m))
    mate_idx = np.flatnonzero(is_mate)
    mbank = mate_idx[rng.integers(0, len(mate_idx), min(256, len(mate_idx)))]
    with torch.no_grad():
        mb = net.embedB(G_ids[mbank].to(dev), G_stm[mbank].to(dev))   # (M,d) mate-region embeddings
    zt = np.load(args.dtm_data)
    q_ids, q_stm = tokens(zt["packed"], zt["meta"])
    q_dtm = zt["dtm"].astype(np.float32)
    keep = q_dtm > 0
    q_ids, q_stm, q_dtm = q_ids[keep], q_stm[keep], q_dtm[keep]
    sub = rng.integers(0, len(q_dtm), min(4000, len(q_dtm)))
    q_ids, q_stm, q_dtm = q_ids[sub], q_stm[sub], q_dtm[sub]
    with torch.no_grad():
        d2mate = []
        for k in range(0, len(q_ids), 2048):
            qf = net.embedF(torch.from_numpy(q_ids[k:k+2048].astype(np.int64)).to(dev),
                            torch.from_numpy(q_stm[k:k+2048].astype(np.int64)).to(dev))  # (B,d)
            dd = net.iqe(qf[:, None, :].expand(-1, mb.shape[0], -1).reshape(-1, qf.shape[1]),
                         mb[None].expand(qf.shape[0], -1, -1).reshape(-1, mb.shape[1]))
            dd = dd.reshape(qf.shape[0], mb.shape[0]).min(1).values
            d2mate.append(dd.cpu().numpy())
    d2mate = np.concatenate(d2mate)
    sp_mate = spearman(d2mate, q_dtm)
    mae_mate = float(np.abs(d2mate - q_dtm).mean())

    print(f"VERDICT QUASIMETRIC d{args.d}c{args.iqe_components}"
          f"{'-shared' if args.shared else ''} ({npar/1e6:.2f}M, {time.time()-t0:.0f}s):", flush=True)
    print(f"  (1) eff_rank(F) {er_mean:.1f}±{er_sd:.1f} / {args.d}   [scalar field was ~3-5]", flush=True)
    print(f"  (2) mate-via-min vs true DTM: spearman {sp_mate:+.3f} MAE {mae_mate:.1f}", flush=True)
    print(f"  (3) held-out pair ordering:   spearman {sp_pair:+.3f} MAE {mae_pair:.1f}  "
          f"[scalar far-ceiling +0.2]", flush=True)
    print(f"  (4) triangle violations: {100*viol:.2f}% of triples (mean slack {viol_mag:.3f})", flush=True)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(),
                    "cfg": {"d": args.d, "d_bb": args.d_bb, "blocks": args.blocks,
                            "iqe_components": args.iqe_components, "shared": args.shared},
                    "data": args.data,
                    "metrics": {"eff_rank": er_mean, "mate_min_spearman": sp_mate,
                                "pair_spearman": sp_pair, "triangle_viol": viol}}, args.save)
        print(f"  saved -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
