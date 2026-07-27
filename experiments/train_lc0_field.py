#!/usr/bin/env python
"""experiments/train_lc0_field.py -- train the field on lc0 (lczerolens) 112-plane, REAL-history
data (Kaveh's newest method). Reuses the input-agnostic machinery: ClockField(in_planes=112),
IQE quasimetric + learnable MATE goal + categorical ending head, TESTED losses (losses.py).
Only the input encoding (lc0 112) + data (real-history trajectories) changed vs the endgame toy.
Draw-surface check modifies the rule50 plane (109) directly.
"""
from __future__ import annotations

import argparse, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.nn.fb import pick_device
from experiments.arch_bakeoff import eff_rank
from experiments.train_clock_field import ClockField
from experiments.losses import (quasimetric_regression, wdl_hinge, anchored_pairwise_rank,
                                categorical_ending_loss)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/derived/traj_lc0_v1.npz")
    ap.add_argument("--d", type=int, default=64); ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8); ap.add_argument("--margin", type=float, default=400.0)
    ap.add_argument("--w-rank", type=float, default=1.0); ap.add_argument("--w-cat", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=14000); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--rank-pairs", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto"); ap.add_argument("--save", default="artifacts/experiments/lc0_field_v1.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    z = np.load(args.data)
    planes = z["planes"]                                    # (N,112,8,8) uint8
    dtz = z["dtz"].astype(np.float32); ending = z["ending"].astype(np.int64); grp = z["group"]
    won = dtz >= 0; inf = dtz < 0
    idx_won = np.flatnonzero(won); idx_inf = np.flatnonzero(inf)
    tgt = torch.from_numpy(np.where(won, np.log1p(np.clip(dtz, 0, None)), 0.0).astype(np.float32)).to(dev)
    won_m = torch.from_numpy(won.astype(np.float32)).to(dev); end_t = torch.from_numpy(ending).to(dev)
    logM = float(np.log1p(args.margin))
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

    net = ClockField(args.d, ch=args.ch, blocks=args.blocks, in_planes=112).to(dev)
    print(f"[lc0-field] rows {len(dtz)} won {len(idx_won)} inf {len(idx_inf)} | pairs {len(P_lo)} | "
          f"112-plane lc0 input | {sum(p.numel() for p in net.parameters())/1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    def fp(idx):
        return torch.from_numpy(planes[idx].astype(np.float32)).to(dev)

    hb = args.batch // 2
    for s in range(args.steps):
        bw = idx_won[rng.integers(0, len(idx_won), hb)]; bi = idx_inf[rng.integers(0, len(idx_inf), hb)]
        bb = np.concatenate([bw, bi]); dm, catlog = net.d_mate_and_end(fp(bb))
        reg = quasimetric_regression(dm[won_m[bb].bool()], tgt[bb][won_m[bb].bool()])
        hin = wdl_hinge(dm, won_m[bb], logM)
        cat = categorical_ending_loss(catlog, end_t[bb])
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
        ce = rng.integers(0, len(dtz), 4000)
        cat_acc = float((net.d_mate_and_end(fp(ce))[1].argmax(1).cpu().numpy() == ending[ce]).mean()) * 100
        # DRAW-SURFACE: modify rule50 plane (109) directly -> committor should rise toward draw
        base = idx_won[rng.integers(0, len(idx_won), 400)]; surf = []
        for h in (0, 40, 80, 96):
            pl = planes[base].astype(np.float32).copy(); pl[:, 109] = float(h)
            d = net.d_mate(torch.from_numpy(pl).to(dev)).cpu().numpy(); surf.append((h, float(np.median(d))))
    print(f"VERDICT LC0-FIELD d{args.d}: 1-ply rank-acc {racc:.1f}% | eff_rank {er:.1f} | "
          f"ENDING-acc {cat_acc:.1f}% | [{time.time()-t0:.0f}s]", flush=True)
    print(f"  DRAW-SURFACE (median d vs rule50, should RISE toward draw): " + " ".join(f"h{h}:{d:.0f}" for h, d in surf), flush=True)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "model": "ClockField",
                "cfg": {"d": args.d, "ch": args.ch, "blocks": args.blocks, "in_planes": 112},
                "metrics": {"rank_acc": racc, "eff_rank": er, "ending_acc": cat_acc, "surface": surf}}, args.save)
    print(f"  saved -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
