#!/usr/bin/env python
"""diagnose_asymmetry.py -- WHY is d(rev)/d(fwd) only ~2x? Which end is binding?

A ratio can be small for two completely different reasons and the fix differs:

  FORWARD FLOOR    d(a->b) never gets near zero on observed pairs, so even a large reverse distance
                   divides down to a modest ratio. The lever is the reachability term / IQE scale.
  REVERSE CEILING  d(b->a) barely exceeds d(a->b), because NOTHING in the objective pushes it up --
                   reverse distance is unconstrained, not penalised. The lever is an explicit
                   irreversibility hinge on pairs with no observed repetition evidence.

This prints the absolute levels, not just the ratio, plus the headroom actually available:
what fraction of IQE components are already saturated, and what the reverse distance would be if
every non-dominated component contributed its full interval.

It also splits by whether a reversal was OBSERVED (coverage), which is the only data-grounded
notion of reversibility available -- pairs with observed repetition SHOULD be near-symmetric, and
if they are not, the model is asserting irreversibility it has no evidence for.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.experiments.plot_strata_figures import (
    embed)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T


def q(v, name):
    p = np.percentile(v, [5, 25, 50, 75, 95])
    print(f"    {name:<22} p5={p[0]:8.3f}  p25={p[1]:8.3f}  med={p[2]:8.3f}  "
          f"p75={p[3]:8.3f}  p95={p[4]:8.3f}   mean={v.mean():8.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=paths.experiment("reach_vit_v1_step20000.pt"))
    ap.add_argument("--n-pair", type=int, default=40000)
    ap.add_argument("--max-gap", type=int, default=40)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    net, payload = load_net(args.ckpt, args.device)
    c = payload["cfg"]
    tr = T.build(n_human=c["games"] // 2, n_sf=c["games"] // 2, seed=c["traj_seed"],
                 max_plies=c["max_plies"], verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    test = np.flatnonzero(split == 2)
    game, ply, pc, cov = tr.game_of_row(), tr.ply_of_row(), tr.piece_count(), tr.coverage()
    rows = np.flatnonzero(np.isin(game, test))
    rng = np.random.default_rng(0)
    iqe = net.qhead.iqe if getattr(net, "dual", False) else net.iqe

    i0 = rows[rng.integers(0, len(rows), args.n_pair)]
    g = game[i0]
    end = tr.start[g] + tr.length[g] - 1
    j0 = i0 + 1 + (rng.random(args.n_pair) * np.minimum(args.max_gap, end - i0)).astype(np.int64)
    ok = j0 <= end
    i0, j0 = i0[ok], j0[ok]
    observed_rev = j0 <= cov[i0]                       # repetition evidence: the reversal WAS seen
    drop = pc[i0].astype(int) - pc[j0].astype(int)

    Za, Zb = embed(net, tr, i0, args.device), embed(net, tr, j0, args.device)
    with torch.no_grad():
        A, B = Za.to(args.device), Zb.to(args.device)
        d_f = iqe(A, B).float().cpu().numpy()
        d_r = iqe(B, A).float().cpu().numpy()
        # component-level headroom: how much of the embedding is even ENGAGED in the asymmetry
        za, zb = A, B
        gap_fwd = torch.relu(zb - za)                  # coords where the target exceeds the source
        gap_rev = torch.relu(za - zb)
        eng_f = float((gap_fwd > 1e-4).float().mean())
        eng_r = float((gap_rev > 1e-4).float().mean())

    print(f"\n=== {args.ckpt}  (step {payload.get('step')}) — {len(d_f):,} test pairs ===")
    print(f"\n  ABSOLUTE LEVELS  (is the ratio limited by a forward FLOOR or a reverse CEILING?)")
    q(d_f, "d(a->b) forward"); q(d_r, "d(b->a) reverse")
    zf = float((d_f < 1e-3).mean())
    print(f"    forward distance is < 1e-3 on {zf:6.1%} of observed pairs "
          f"{'(floor is NOT binding)' if zf > .3 else '(FORWARD FLOOR: forward never reaches zero)'}")
    print(f"    median ratio d_rev/d_fwd = {np.median(d_r/np.maximum(d_f,1e-6)):.3f}   "
          f"median DIFFERENCE d_rev - d_fwd = {np.median(d_r-d_f):+.3f}")

    print(f"\n  COMPONENT ENGAGEMENT  (how much of the {Za.shape[1]}-dim embedding carries the order?)")
    print(f"    coords where target exceeds source, forward: {eng_f:6.1%}   reverse: {eng_r:6.1%}")
    print(f"    -> if BOTH are high, the pair is near-symmetric per-coordinate and the architecture")
    print(f"       is not being used to encode an order; more components would not help.")

    print(f"\n  SPLIT BY EVIDENCE  (repetition coverage is the only data-grounded reversibility)")
    for m, lab in ((observed_rev, "reversal OBSERVED (repetition)"),
                   (~observed_rev, "reversal never observed")):
        if m.sum() < 100:
            continue
        r = d_r[m] / np.maximum(d_f[m], 1e-6)
        print(f"    {lab:<32} n={m.sum():>7,}  med ratio {np.median(r):6.3f}  "
              f"med d_fwd {np.median(d_f[m]):7.3f}  med d_rev {np.median(d_r[m]):7.3f}")

    print(f"\n  SPLIT BY MATERIAL DROP  (graded ratchet?)")
    for k in range(0, 5):
        m = (drop == k) & (~observed_rev)
        if m.sum() < 200:
            continue
        r = d_r[m] / np.maximum(d_f[m], 1e-6)
        print(f"    lost {k} pieces  n={m.sum():>7,}  med ratio {np.median(r):6.3f}  "
              f"med d_fwd {np.median(d_f[m]):7.3f}  med d_rev {np.median(d_r[m]):7.3f}")
    print()


if __name__ == "__main__":
    main()
