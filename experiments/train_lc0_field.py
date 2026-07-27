#!/usr/bin/env python
"""experiments/train_lc0_field.py -- the CORRECT arch (Kaveh): SINGLE-SPACE (shared phi) IQE
quasimetric trained MULTI-GOAL (same-line pairs d(phi(s),phi(g))->Delta = triangulation ->
rank+composability+fine ordering) + REPULSION (anti-collapse) + mate-goal readout d(phi(s),MATE)
+ WDL hinge barriers + categorical ending head. lc0 112-plane REAL-history input. All losses
tested (losses.py). Restores the IQE as a true quasimetric (was a single-goal scalar).
"""
from __future__ import annotations

import argparse, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.nn.fb import pick_device
from experiments.arch_bakeoff import eff_rank
from experiments.train_clock_field import ClockField
from experiments.losses import quasimetric_regression, wdl_hinge, categorical_ending_loss


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/derived/traj_lc0_v2.npz")
    ap.add_argument("--d", type=int, default=64); ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8); ap.add_argument("--margin", type=float, default=400.0)
    ap.add_argument("--w-multi", type=float, default=1.0); ap.add_argument("--w-mate", type=float, default=1.0)
    ap.add_argument("--w-hinge", type=float, default=1.0); ap.add_argument("--w-repel", type=float, default=0.3)
    ap.add_argument("--w-cat", type=float, default=0.5); ap.add_argument("--repel-margin", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=16000); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--pairs", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto"); ap.add_argument("--save", default="artifacts/experiments/lc0_field_correct.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    z = np.load(args.data)
    planes = z["planes"]; dtz = z["dtz"].astype(np.float32); ending = z["ending"].astype(np.int64)
    game = z["game"]; ply = z["ply"]
    won = dtz >= 0; inf = dtz < 0
    idx_won = np.flatnonzero(won); idx_inf = np.flatnonzero(inf)
    tgt = torch.from_numpy(np.where(won, np.log1p(np.clip(dtz, 0, None)), 0.0).astype(np.float32)).to(dev)
    won_m = torch.from_numpy(won.astype(np.float32)).to(dev); end_t = torch.from_numpy(ending).to(dev)
    logM = float(np.log1p(args.margin))
    # MULTI-GOAL same-line pairs (s before g on the line -> d(s->g)=ply gap)
    g2 = defaultdict(list)
    for i in range(len(dtz)): g2[game[i]].append(i)
    MG_s, MG_g, MG_d = [], [], []
    for rows in g2.values():
        rows = sorted(rows, key=lambda i: ply[i])
        if len(rows) < 2: continue
        for _ in range(min(10, len(rows))):
            a, b = sorted(rng.integers(0, len(rows), 2))
            if a == b: continue
            si, gj = rows[a], rows[b]; delta = ply[gj] - ply[si]
            if delta <= 0: continue
            MG_s.append(si); MG_g.append(gj); MG_d.append(np.log1p(delta))
    MG_s = np.array(MG_s); MG_g = np.array(MG_g); MG_d = np.array(MG_d, np.float32)
    print(f"[lc0-correct] rows {len(dtz)} won {len(idx_won)} inf {len(idx_inf)} | multi-goal pairs {len(MG_s)} "
          f"| SINGLE-SPACE phi, multi-goal+repel+mate+wdl+cat", flush=True)

    net = ClockField(args.d, ch=args.ch, blocks=args.blocks, in_planes=112).to(dev)
    print(f"  {sum(p.numel() for p in net.parameters())/1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    def fp(idx):
        return torch.from_numpy(planes[idx].astype(np.float32)).to(dev)

    hb = args.batch // 2
    for s in range(args.steps):
        # multi-goal + repulsion (shared phi)
        pi = rng.integers(0, len(MG_s), args.pairs)
        es = net.phi(fp(MG_s[pi])); eg = net.phi(fp(MG_g[pi]))
        L_multi = quasimetric_regression(net.d_pair_emb(es, eg), torch.from_numpy(MG_d[pi]).to(dev))
        perm = torch.randperm(len(pi), device=dev)
        L_repel = F.relu(args.repel_margin - torch.log1p(net.d_pair_emb(es, eg[perm]).clamp(min=0))).mean()
        # mate-goal readout + WDL hinge + categorical (won/inf balanced batch)
        bw = idx_won[rng.integers(0, len(idx_won), hb)]; bi = idx_inf[rng.integers(0, len(idx_inf), hb)]
        bb = np.concatenate([bw, bi]); dm, catlog = net.d_mate_and_end(fp(bb))
        L_mate = quasimetric_regression(dm[won_m[bb].bool()], tgt[bb][won_m[bb].bool()])
        L_hinge = wdl_hinge(dm, won_m[bb], logM)
        L_cat = categorical_ending_loss(catlog, end_t[bb])
        loss = (args.w_multi * L_multi + args.w_repel * L_repel + args.w_mate * L_mate
                + args.w_hinge * L_hinge + args.w_cat * L_cat)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 2000 == 0:
            print(f"  step {s} multi {float(L_multi):.3f} repel {float(L_repel):.3f} mate {float(L_mate):.3f} "
                  f"hinge {float(L_hinge):.3f} cat {float(L_cat):.3f} [{time.time()-t0:.0f}s]", flush=True)

    net.eval()
    from scipy.stats import spearmanr
    with torch.no_grad():
        te = rng.integers(0, len(MG_s), 6000)
        dp = net.d_pair(fp(MG_s[te]), fp(MG_g[te])).cpu().numpy()
        sp_pair = float(spearmanr(dp, np.expm1(MG_d[te])).correlation)         # multi-goal ordering
        sub = idx_won[rng.integers(0, len(idx_won), 3000)]
        er = eff_rank(net.phi(fp(sub)).cpu().numpy())
        dd = net.d_mate(fp(sub)).cpu().numpy()
        sp_mate = float(spearmanr(dd, dtz[sub]).correlation)
        ce = rng.integers(0, len(dtz), 4000)
        cat_acc = float((net.d_mate_and_end(fp(ce))[1].argmax(1).cpu().numpy() == ending[ce]).mean()) * 100
        base = idx_won[rng.integers(0, len(idx_won), 400)]; surf = []
        for h in (0, 40, 80, 96):
            pl = planes[base].astype(np.float32).copy(); pl[:, 109] = float(h)
            surf.append((h, float(np.median(net.d_mate(torch.from_numpy(pl).to(dev)).cpu().numpy()))))
    print(f"VERDICT LC0-CORRECT d{args.d}: multi-goal pair-order {sp_pair:+.3f} | eff_rank {er:.1f} "
          f"(single-goal was ~3.5) | mate-vs-DTZ {sp_mate:+.3f} | ENDING {cat_acc:.1f}% | [{time.time()-t0:.0f}s]", flush=True)
    print(f"  DRAW-SURFACE (median d vs rule50): " + " ".join(f"h{h}:{d:.0f}" for h, d in surf), flush=True)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "model": "ClockField",
                "cfg": {"d": args.d, "ch": args.ch, "blocks": args.blocks, "in_planes": 112},
                "metrics": {"pair_order": sp_pair, "eff_rank": er, "mate_vs_dtz": sp_mate,
                            "ending_acc": cat_acc, "surface": surf}}, args.save)
    print(f"  saved -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
