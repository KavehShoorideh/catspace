#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/reachability_field/experiments/train_field_v3.py -- corrected mate field using the TESTED loss module
(catspace/research/tools/training_infra/losses.py). Closes the loop: mate attractor + WDL hinge barrier + TIE-SAFE
ANCHORED 1-ply rank (the fix), all imported from losses.py (no re-implemented loss terms).
Goal: a single field with BOTH good 1-ply ordering AND preserved won-vs-draw scale, then
re-run conversion. Save checkpoint for ab_convert/mate_with_search.
"""
from __future__ import annotations

import argparse, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
from catspace.research.components.encoder.approaches.reachability_field.experiments.arch_bakeoff import eff_rank, tokens
from catspace.research.components.encoder.approaches.reachability_field.experiments.train_mate_field import MateField
from catspace.research.tools.training_infra.losses import quasimetric_regression, wdl_hinge, anchored_pairwise_rank
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=paths.derived("child_rank_v1.npz"))
    ap.add_argument("--margin", type=float, default=400.0)
    ap.add_argument("--w-rank", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=14000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--rank-pairs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save", default=paths.experiment("field_v3.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    z = np.load(args.data)
    ids, stm = tokens(z["packed"], z["meta"]); dtz = z["dtz"].astype(np.float32); grp = z["group"]
    Ci = torch.from_numpy(ids.astype(np.int64)); Cs = torch.from_numpy(stm.astype(np.int64))
    won = dtz >= 0; inf = dtz < 0
    idx_won = np.flatnonzero(won); idx_inf = np.flatnonzero(inf)
    tgt = torch.from_numpy(np.where(won, np.log1p(np.clip(dtz, 0, None)), 0.0).astype(np.float32)).to(dev)
    won_m = torch.from_numpy(won.astype(np.float32)).to(dev)
    logM = float(np.log1p(args.margin))
    # DISTINGUISHABLE sibling pairs (drop ties) with true log-gap; closer-first
    g2 = defaultdict(list)
    for i in idx_won: g2[grp[i]].append(i)
    lo_l, hi_l, gap_l = [], [], []
    for v in g2.values():
        v = np.array(v)
        if len(v) < 2: continue
        for _ in range(min(8, len(v))):
            a, b = rng.integers(0, len(v), 2)
            if dtz[v[a]] == dtz[v[b]]: continue
            lo, hi = (v[a], v[b]) if dtz[v[a]] < dtz[v[b]] else (v[b], v[a])
            lo_l.append(lo); hi_l.append(hi); gap_l.append(np.log1p(dtz[hi]) - np.log1p(dtz[lo]))
    P_lo = np.array(lo_l); P_hi = np.array(hi_l); P_gap = np.array(gap_l, np.float32)
    print(f"[v3] won {len(idx_won)} inf {len(idx_inf)} | distinguishable pairs {len(P_lo)} | "
          f"w_rank {args.w_rank} logM {logM:.2f}", flush=True)

    net = MateField(32, 64, 6, 16).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    hb = args.batch // 2
    for s in range(args.steps):
        bw = idx_won[rng.integers(0, len(idx_won), hb)]; bi = idx_inf[rng.integers(0, len(idx_inf), hb)]
        bb = np.concatenate([bw, bi]); dm = net.d_to_mate(Ci[bb].to(dev), Cs[bb].to(dev))
        reg = quasimetric_regression(dm[won_m[bb].bool()], tgt[bb][won_m[bb].bool()])
        hin = wdl_hinge(dm, won_m[bb], logM)
        pi = rng.integers(0, len(P_lo), args.rank_pairs)
        dlo = net.d_to_mate(Ci[P_lo[pi]].to(dev), Cs[P_lo[pi]].to(dev))
        dhi = net.d_to_mate(Ci[P_hi[pi]].to(dev), Cs[P_hi[pi]].to(dev))
        rnk = anchored_pairwise_rank(dlo, dhi, torch.from_numpy(P_gap[pi]).to(dev))
        loss = reg + hin + args.w_rank * rnk
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 2000 == 0:
            print(f"  step {s} reg {float(reg):.3f} hinge {float(hin):.3f} rank {float(rnk):.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    net.eval()
    with torch.no_grad():
        te = rng.integers(0, len(P_lo), 8000)
        dlo = net.d_to_mate(Ci[P_lo[te]].to(dev), Cs[P_lo[te]].to(dev)).cpu().numpy()
        dhi = net.d_to_mate(Ci[P_hi[te]].to(dev), Cs[P_hi[te]].to(dev)).cpu().numpy()
        racc = float((dlo < dhi).mean()) * 100
        sub = idx_won[rng.integers(0, len(idx_won), 3000)]
        er = eff_rank(net.phi(Ci[sub].to(dev), Cs[sub].to(dev)).cpu().numpy())
        dd = net.d_to_mate(Ci[sub].to(dev), Cs[sub].to(dev)).cpu().numpy()
        si = idx_inf[rng.integers(0, len(idx_inf), 3000)]
        di = net.d_to_mate(Ci[si].to(dev), Cs[si].to(dev)).cpu().numpy()
    from scipy.stats import spearmanr
    sp = float(spearmanr(dd, dtz[sub]).correlation)
    print(f"VERDICT FIELD-V3: 1-ply RANK-ACC {racc:.1f}% | d-vs-DTZ {sp:+.3f} | eff_rank {er:.1f} "
          f"| won-d {np.median(dd):.1f} vs INF-d {np.median(di):.1f} (want BOTH: order>=80 AND "
          f"won<<INF) | [{time.time()-t0:.0f}s]", flush=True)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(),
                "cfg": {"d": 32, "d_bb": 64, "blocks": 6, "iqe_components": 16},
                "metrics": {"rank_acc": racc, "d_vs_dtz": sp, "eff_rank": er,
                            "won_d": float(np.median(dd)), "inf_d": float(np.median(di))}}, args.save)
    print(f"  saved -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
