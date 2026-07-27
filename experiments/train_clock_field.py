#!/usr/bin/env python
"""experiments/train_clock_field.py -- CLOCK-AWARE mate field (Kaveh). Uses the FULL 20-plane
feature stack (feature_planes: 12 pieces + stm + castling + ep + HALFMOVE CLOCK + repetition),
so the field can see & represent the approaching 50-move draw surface. Conv-over-planes encoder
(the toy tokens() dropped the clock). Tested losses from experiments/losses.py.
Verdict includes a DRAW-SURFACE check: for fixed won positions, does the committor DROP as the
halfmove clock rises? (It must, if the surface is represented.)
"""
from __future__ import annotations

import argparse, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.nn.fb import pick_device
from catspace.nn.features import feature_planes
from catspace.nn.iqe import IQE
from experiments.arch_bakeoff import eff_rank
from experiments.losses import (quasimetric_regression, wdl_hinge, anchored_pairwise_rank,
                                categorical_ending_loss, N_ENDINGS)


class ClockField(nn.Module):
    """conv over 20 feature planes -> phi; d(s)=IQE(phi(s), MATE) + CATEGORICAL ending-type head
    (Kaveh: 'what kind of end is approaching'). Sees the halfmove clock."""
    def __init__(self, d=32, ch=64, blocks=5, iqe_components=16, in_planes=20):
        # in_planes: 20 = single-position (endgame). Full board later = 8-history lc0 stack
        # (~112 planes) -> pass in_planes=112; the rest of the net is unchanged. NOT endgame-locked.
        super().__init__()
        self.in_planes = in_planes
        self.stem = nn.Sequential(nn.Conv2d(in_planes, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch)) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv2d(ch, 32, 1), nn.GroupNorm(8, 32), nn.ReLU(),
                                  nn.Flatten(), nn.Linear(32 * 64, d))
        self.iqe = IQE(d, components=iqe_components)
        self.mate = nn.Parameter(torch.randn(d) * 0.1)
        self.cat = nn.Linear(d, N_ENDINGS)                   # categorical ending-type head

    def phi(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = torch.relu(h + b(h))
        return self.head(h)

    def d_mate_and_end(self, x):
        e = self.phi(x)
        return self.iqe(e, self.mate.expand_as(e)), self.cat(e)

    def d_mate(self, x):
        e = self.phi(x)
        return self.iqe(e, self.mate.expand_as(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/derived/clock_child_v1.npz")
    ap.add_argument("--d", type=int, default=32); ap.add_argument("--margin", type=float, default=400.0)
    ap.add_argument("--ch", type=int, default=64); ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--w-rank", type=float, default=1.0); ap.add_argument("--w-cat", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=14000); ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--rank-pairs", type=int, default=384); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto"); ap.add_argument("--save", default="artifacts/experiments/clock_field_v1.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    z = np.load(args.data)
    packed, meta = z["packed"], z["meta"]; dtz = z["dtz"].astype(np.float32); grp = z["group"]
    ending = z["ending"].astype(np.int64); end_t = torch.from_numpy(ending).to(dev)
    hm = np.minimum(meta[:, 6], 100).astype(np.float32)
    won = dtz >= 0; inf = dtz < 0
    idx_won = np.flatnonzero(won); idx_inf = np.flatnonzero(inf)
    tgt = torch.from_numpy(np.where(won, np.log1p(np.clip(dtz, 0, None)), 0.0).astype(np.float32)).to(dev)
    won_m = torch.from_numpy(won.astype(np.float32)).to(dev); logM = float(np.log1p(args.margin))
    g2 = defaultdict(list)
    for i in idx_won: g2[grp[i]].append(i)
    P_lo, P_hi, P_gap = [], [], []
    for v in g2.values():
        v = np.array(v)
        if len(v) < 2: continue
        for _ in range(min(8, len(v))):
            a, b = rng.integers(0, len(v), 2)
            if dtz[v[a]] == dtz[v[b]]: continue
            lo, hi = (v[a], v[b]) if dtz[v[a]] < dtz[v[b]] else (v[b], v[a])
            P_lo.append(lo); P_hi.append(hi); P_gap.append(np.log1p(dtz[hi]) - np.log1p(dtz[lo]))
    P_lo = np.array(P_lo); P_hi = np.array(P_hi); P_gap = np.array(P_gap, np.float32)
    print(f"[clock-field] rows {len(dtz)} won {len(idx_won)} inf {len(idx_inf)} | pairs {len(P_lo)} | "
          f"20-plane feature encoder (sees halfmove clock)", flush=True)

    def fp(idx):
        return torch.from_numpy(feature_planes(packed[idx], meta[idx])).to(dev)

    net = ClockField(args.d, ch=args.ch, blocks=args.blocks).to(dev)
    print(f"  encoder: d={args.d} ch={args.ch} blocks={args.blocks} "
          f"({sum(p.numel() for p in net.parameters())/1e6:.2f}M params)", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    hb = args.batch // 2
    for s in range(args.steps):
        bw = idx_won[rng.integers(0, len(idx_won), hb)]; bi = idx_inf[rng.integers(0, len(idx_inf), hb)]
        bb = np.concatenate([bw, bi]); dm, catlog = net.d_mate_and_end(fp(bb))
        reg = quasimetric_regression(dm[won_m[bb].bool()], tgt[bb][won_m[bb].bool()])
        hin = wdl_hinge(dm, won_m[bb], logM)
        cat = categorical_ending_loss(catlog, end_t[bb])     # categorical ending-type head
        pi = rng.integers(0, len(P_lo), args.rank_pairs)
        dlo = net.d_mate(fp(P_lo[pi])); dhi = net.d_mate(fp(P_hi[pi]))
        rnk = anchored_pairwise_rank(dlo, dhi, torch.from_numpy(P_gap[pi]).to(dev))
        loss = reg + hin + args.w_rank * rnk + args.w_cat * cat
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 2000 == 0:
            print(f"  step {s} reg {float(reg):.3f} hinge {float(hin):.3f} rank {float(rnk):.3f} "
                  f"cat {float(cat):.3f} [{time.time()-t0:.0f}s]", flush=True)

    net.eval()
    with torch.no_grad():
        te = rng.integers(0, len(P_lo), 6000)
        racc = float((net.d_mate(fp(P_lo[te])).cpu().numpy() < net.d_mate(fp(P_hi[te])).cpu().numpy()).mean()) * 100
        sub = idx_won[rng.integers(0, len(idx_won), 3000)]
        er = eff_rank(net.phi(fp(sub)).cpu().numpy())
        # categorical ending-type accuracy on a held sample
        ce = rng.integers(0, len(dtz), 4000)
        cat_pred = net.d_mate_and_end(fp(ce))[1].argmax(1).cpu().numpy()
        cat_acc = float((cat_pred == ending[ce]).mean()) * 100
        # DRAW-SURFACE check: same won positions, sweep halfmove clock -> committor should DROP
        base = idx_won[rng.integers(0, len(idx_won), 400)]
        surf = []
        for h in (0, 40, 80, 96):
            m2 = meta[base].copy(); m2[:, 6] = h
            d = net.d_mate(torch.from_numpy(feature_planes(packed[base], m2)).to(dev)).cpu().numpy()
            surf.append((h, float(np.median(d))))
    print(f"VERDICT CLOCK-FIELD d{args.d}: 1-ply rank-acc {racc:.1f}% | eff_rank {er:.1f} | "
          f"ENDING-acc {cat_acc:.1f}% (6-way) | [{time.time()-t0:.0f}s]", flush=True)
    print(f"  DRAW-SURFACE (median d vs halfmove, should RISE toward the draw as clock->100): "
          + " ".join(f"h{h}:{d:.0f}" for h, d in surf), flush=True)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "model": "ClockField",
                "cfg": {"d": args.d}, "metrics": {"rank_acc": racc, "eff_rank": er, "surface": surf}}, args.save)
    print(f"  saved -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
