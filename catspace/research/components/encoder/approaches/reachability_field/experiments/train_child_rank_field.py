#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/reachability_field/experiments/train_child_rank_field.py -- S2d: fix conversion by giving the field LOCAL
move-resolution + restoring rank. Same MateField (single-space IQE + learnable collapsed MATE
goal + WDL hinge-to-M barriers), but trained on child_rank data with:
  * regression d(child) -> log1p(|DTZ|) for won children, 0 for mate (attractor);
  * hinge d UP to log1p(M) for children that throw the win (INF barrier / draw interface);
  * WITHIN-GROUP pairwise RANK loss: for two won siblings, d(lower-|DTZ|) < d(higher-|DTZ|).
The rank loss is the move-selection gradient the field lacked (52.7% coin flip) AND, by forcing
the field to distinguish many siblings, it should fight the rank collapse (1.7). Tensor-batched.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
from catspace.research.components.encoder.approaches.reachability_field.experiments.arch_bakeoff import eff_rank, tokens
from catspace.research.components.encoder.approaches.reachability_field.experiments.train_mate_field import MateField
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=paths.derived("child_rank_v1.npz"))
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--d-bb", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--iqe-components", type=int, default=16)
    ap.add_argument("--margin", type=float, default=400.0)
    ap.add_argument("--rank-weight", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--rank-pairs", type=int, default=256, help="within-group pairs per step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save", default=paths.experiment("mate_field_v2.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    z = np.load(args.data)
    ids, stm = tokens(z["packed"], z["meta"])
    dtz = z["dtz"].astype(np.float32); grp = z["group"]
    ids_t = torch.from_numpy(ids.astype(np.int64)); stm_t = torch.from_numpy(stm.astype(np.int64))
    won = dtz >= 0; inf = dtz < 0
    tgt = np.where(won, np.log1p(np.clip(dtz, 0, None)), 0.0).astype(np.float32)
    tgt_t = torch.from_numpy(tgt).to(dev); won_t = torch.from_numpy(won.astype(np.float32)).to(dev)
    logM = float(np.log1p(args.margin))
    idx_won = np.flatnonzero(won); idx_inf = np.flatnonzero(inf)

    # groups with >=2 won children (for pairwise rank loss)
    from collections import defaultdict
    g2c = defaultdict(list)
    for i in idx_won:
        g2c[grp[i]].append(i)
    rank_groups = [np.array(v) for v in g2c.values() if len(v) >= 2]
    print(f"[child-rank] {len(dtz)} rows | won {len(idx_won)} inf {len(idx_inf)} | "
          f"rankable groups {len(rank_groups)} | logM {logM:.2f}", flush=True)

    net = MateField(args.d, args.d_bb, args.blocks, args.iqe_components).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    def sample_rank_pairs(k):
        gi = rng.integers(0, len(rank_groups), k)
        a = np.empty(k, np.int64); b = np.empty(k, np.int64)
        for t, g in enumerate(gi):
            cs = rank_groups[g]; ii, jj = rng.integers(0, len(cs), 2)
            while jj == ii: jj = rng.integers(0, len(cs))
            a[t], b[t] = cs[ii], cs[jj]
        return a, b

    for s in range(args.steps):
        bw = idx_won[rng.integers(0, len(idx_won), args.batch // 2)]
        bi = idx_inf[rng.integers(0, len(idx_inf), args.batch // 2)] if len(idx_inf) else bw
        b = np.concatenate([bw, bi])
        d = net.d_to_mate(ids_t[b].to(dev), stm_t[b].to(dev)); dlog = torch.log1p(d.clamp(min=0))
        w = won_t[b]
        reg = (F.huber_loss(dlog, tgt_t[b], reduction="none") * w).sum() / w.sum().clamp(min=1)
        hinge = (F.relu(logM - dlog) * (1 - w)).sum() / (1 - w).sum().clamp(min=1)
        # within-group pairwise rank: want d(a) < d(b) when dtz[a] < dtz[b]
        pa, pb = sample_rank_pairs(args.rank_pairs)
        da = net.d_to_mate(ids_t[pa].to(dev), stm_t[pa].to(dev))
        db = net.d_to_mate(ids_t[pb].to(dev), stm_t[pb].to(dev))
        y = torch.from_numpy(np.sign(dtz[pb] - dtz[pa]).astype(np.float32)).to(dev)  # +1 if a closer
        rank = F.margin_ranking_loss(db, da, y, margin=0.5)      # push d(farther) - d(closer) >= margin
        loss = reg + hinge + args.rank_weight * rank
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 2000 == 0:
            print(f"  step {s} loss {float(loss):.3f} (reg {float(reg):.3f} hinge {float(hinge):.3f} "
                  f"rank {float(rank):.3f}) [{time.time()-t0:.0f}s]", flush=True)

    net.eval()
    with torch.no_grad():
        sub = idx_won[rng.integers(0, len(idx_won), min(3000, len(idx_won)))]
        e = net.phi(ids_t[sub].to(dev), stm_t[sub].to(dev)).cpu().numpy(); er = eff_rank(e)
        dd = net.d_to_mate(ids_t[sub].to(dev), stm_t[sub].to(dev)).cpu().numpy()
        si = idx_inf[rng.integers(0, len(idx_inf), min(3000, len(idx_inf)))]
        di = net.d_to_mate(ids_t[si].to(dev), stm_t[si].to(dev)).cpu().numpy()
    from scipy.stats import spearmanr
    sp = float(spearmanr(dd, dtz[sub]).correlation)
    # within-group rank accuracy on held pairs
    pa, pb = sample_rank_pairs(4000)
    with torch.no_grad():
        da = net.d_to_mate(ids_t[pa].to(dev), stm_t[pa].to(dev)).cpu().numpy()
        db = net.d_to_mate(ids_t[pb].to(dev), stm_t[pb].to(dev)).cpu().numpy()
    racc = float((np.sign(db - da) == np.sign(dtz[pb] - dtz[pa])).mean()) * 100
    print(f"VERDICT CHILD-RANK d{args.d}: eff_rank {er:.1f} (was 1.7) | d-vs-DTZ {sp:+.3f} | "
          f"within-group RANK-ACC {racc:.1f}% (coin=50) | won-d med {np.median(dd):.1f} vs "
          f"INF-d med {np.median(di):.1f} | [{time.time()-t0:.0f}s]", flush=True)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(),
                    "cfg": {"d": args.d, "d_bb": args.d_bb, "blocks": args.blocks,
                            "iqe_components": args.iqe_components},
                    "metrics": {"eff_rank": er, "d_vs_dtz": sp, "rank_acc": racc}}, args.save)
        print(f"  saved -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
