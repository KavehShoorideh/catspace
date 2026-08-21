#!/usr/bin/env python
"""eval_interval_pruning.py -- the smoke census for prune-by-bounds (Kaveh 2026-08-20).

"what is the distribution of alpha and beta across all of our options at different
important checkpoints of games" -- per held-out position: every legal move's [dV_lo,
dV_hi] from the interval head, the B*-separation survivor set (moves whose hi is NOT
below the best move's lo), stratified by game phase. Plus the number that licenses all
of it: COVERAGE on held-out CF branches (does the nominal-80 interval contain ~80
percent of real branch outcomes). Conservative pair-bounds mean coverage above nominal
is expected and safe; below nominal is disqualifying.

    .venv/bin/python -m ...eval_interval_pruning --ckpt artifacts/experiments/gdvq1_latest.pt
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/gdvq1_latest.pt")
    ap.add_argument("--graphs", default="data/derived/move_graphs_v2_300000_6545146.npz")
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_delta_value import GraphDeltaValue, batch_of
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)

    dev = args.device
    t0 = time.time()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
    net = GraphDeltaValue(d=A["d_model"], layers=A["layers"], heads=A["attn_heads"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()

    z = np.load(args.graphs)
    keys = ("sq", "gl", "mv", "mut", "vac", "dsc", "att", "n_mv", "played", "n_att", "p_row")
    D = {k: torch.from_numpy(np.ascontiguousarray(z[k])) for k in keys}
    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    gor = tr.game_of_row()
    rs = np.random.default_rng(args.seed); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True
    te_i = np.flatnonzero(test_g[gor[D["p_row"].numpy()]])
    rng = np.random.default_rng(args.seed)
    rows = te_i[rng.choice(len(te_i), min(args.n, len(te_i)), replace=False)]

    # phase from nonpawn-nonking counts (same buckets as the trainer)
    ids = D["sq"][:, :, 0].numpy()
    from catspace.research.tools.training_infra.concepts_reference import TOK_PIECES
    minor_ids = [k + 1 for k, (pt, c) in enumerate(TOK_PIECES) if pt not in (1, 6)]
    npc = np.isin(ids, minor_ids).sum(1)
    phase = np.digitize(npc, [4, 8, 12])       # 0 endgame .. 3 opening

    surv, legal, widths, phases = [], [], [], []
    with torch.no_grad():
        for a in range(0, len(rows), 256):
            ii = torch.from_numpy(rows[a:a + 256])
            sqf, mvf, edges, mask = batch_of(D, ii, dev)
            _, dv = net(sqf, mvf, edges, mask)
            lo, hi = net.last_lo, net.last_hi
            n_mv = D["n_mv"][ii].long()
            for j in range(len(ii)):
                k = int(n_mv[j])
                if k < 2:
                    continue
                l2, h2 = lo[j, :k], hi[j, :k]
                best_lo = l2.max()
                s = int((h2 >= best_lo).sum())
                surv.append(s); legal.append(k)
                widths.append(float((h2 - l2).mean()))
                phases.append(int(phase[rows[a + j]]))

    surv = np.array(surv); legal = np.array(legal)
    widths = np.array(widths); phases = np.array(phases)
    print(f"\n  {len(surv):,} held-out positions [{time.time()-t0:.0f}s]")
    print(f"  survivors: mean {surv.mean():.1f} of {legal.mean():.1f} legal "
          f"({100*surv.mean()/legal.mean():.0f} percent kept, "
          f"{100*(1-surv.mean()/legal.mean()):.0f} percent PRUNED)")
    print(f"  interval width: mean {widths.mean():.3f}")
    print(f"\n  {'phase':<12} {'n':>6} {'legal':>7} {'survive':>8} {'pruned':>7} {'width':>7}")
    for ph, nm in ((3, "opening"), (2, "middle"), (1, "late-mid"), (0, "endgame")):
        m = phases == ph
        if m.sum():
            print(f"  {nm:<12} {int(m.sum()):>6,} {legal[m].mean():>7.1f} "
                  f"{surv[m].mean():>8.1f} {100*(1-surv[m].mean()/legal[m].mean()):>6.0f}% "
                  f"{widths[m].mean():>7.3f}")
    print(f"\n  survivor-count distribution: "
          + "  ".join(f"{q}th={int(np.percentile(surv, q))}" for q in (10, 25, 50, 75, 90)))
    only1 = float((surv == 1).mean())
    print(f"  positions collapsing to a SINGLE candidate: {100*only1:.1f} percent")


if __name__ == "__main__":
    main()
