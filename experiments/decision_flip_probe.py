#!/usr/bin/env python
"""
experiments/decision_flip_probe.py — does deep search CHANGE decisions, and
can the coarse search PREDICT where? (The build/no-build gate for
escalate-on-uncertainty allocation, 2026-07-18.)

Context: mcts@200 = 0.490 conv @ 210 rows/move, mcts@800 = 0.600 @ 811 (the
energy Pareto). Escalation (200n everywhere, 800n only where the 200n search
is contested) only beats the fixed ladder if (a) the 800-vs-200 decision-flip
rate is well below 1 (heterogeneity exists) and (b) flips concentrate where
the coarse top-2 visit gap is small (the gate is predictive). The 2026-07-13
beam-era result (adaptive 0.583 vs uniform 0.600 at matched 455 nodes:
homogeneous difficulty defeats targeting) is the prior AGAINST; this measures
the same question on the current committor-MCTS substrate before any build.

Protocol: play each start with the incumbent mcts@800 vs the tb-optimal
defender (the reference trajectory). At every White decision, ALSO probe the
position with a fresh-tree 200n search (shared eval cache, no tree carry) and
record: flip (200n move != 800n move), the coarse gap fraction
(N1-N2)/(N1+N2), and ply. VERDICT: overall flip rate; flip rate by coarse-gap
tercile; fraction of all flips captured by the lowest-gap tercile (targeting
efficiency: escalating that tercile alone costs ~1/3 of the ladder step).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.value_fixed_point import TB, tb_best_move
from catspace.research.components.planner.approaches.subgoal_cascade.src.probe import probe


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--phead", default="data/derived/sep/cert_base_full_phead.pt")
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_test_n200.json")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--deep", type=int, default=800)
    ap.add_argument("--coarse", type=int, default=200)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    args = ap.parse_args()

    import torch
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.eval_head import EvalHead
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.policy_fb import make_search_policy
    dev = pick_device(args.device)
    fb, pay = load_ckpt(Path(args.ckpt), dev)
    hp = torch.load(args.phead, map_location=dev, weights_only=False)
    ph = EvalHead(d_in=hp["d_in"]).to(dev)
    ph.load_state_dict(hp["state"])
    ph.eval()

    class Committor(torch.nn.Module):
        def forward(self, f):
            p = torch.softmax(ph(f), dim=1)
            return -torch.log(p[:, 0].clamp_min(1e-6)).unsqueeze(-1)

    pol = make_search_policy("mcts", fb, pay["zgoals"]["MATE_W"],
                             max_nodes=args.deep, device=dev,
                             committor_head=Committor())
    # coarse probe rides the SAME mcts (shared cache, budget overridden per
    # call, fresh tree each probe since reuse=None)
    starts = json.loads(Path(args.fixed_set).read_text())["fens"][:args.n]
    tb = TB(args.syzygy_dir)
    rows = []
    t0 = time.perf_counter()
    for i, fen in enumerate(starts):
        rng = np.random.default_rng([args.seed, i])
        b = chess.Board(fen)
        seen = set()
        for _ in range(args.max_plies):
            if b.is_game_over(claim_draw=True):
                break
            if b.turn == chess.WHITE:
                r = probe(pol.mcts, b, budget=args.coarse)
                deep_move = pol.move(b, rng)             # the reference play
                n1, n2 = r.visit_top2
                gap = (n1 - n2) / max(n1 + n2, 1)
                rows.append(dict(flip=int(r.best_move != deep_move),
                                 gap=float(gap), ply=b.ply(), start=i))
                m = deep_move
            else:
                m = tb_best_move(b, tb, seen)
                seen.add(b.board_fen())
            if m is None:
                break
            b.push(m)
    tb.close()

    gaps = np.array([r["gap"] for r in rows])
    flips = np.array([r["flip"] for r in rows], dtype=float)
    terc = np.quantile(gaps, [1 / 3, 2 / 3])
    lo_m, mid_m, hi_m = (gaps <= terc[0]), (gaps > terc[0]) & (gaps <= terc[1]), (gaps > terc[1])
    cap = flips[lo_m].sum() / max(flips.sum(), 1)
    print(f"VERDICT FLIP_RATE={flips.mean():.3f} (n={len(rows)} decisions, "
          f"{args.coarse}n vs {args.deep}n, {time.perf_counter()-t0:.0f}s)")
    print(f"VERDICT FLIP_BY_GAP_TERCILE low={flips[lo_m].mean():.3f} "
          f"mid={flips[mid_m].mean():.3f} high={flips[hi_m].mean():.3f} "
          f"(tercile edges {terc[0]:.2f}/{terc[1]:.2f})")
    print(f"VERDICT LOW_TERCILE_CAPTURE={cap:.3f} of all flips "
          f"(escalating 1/3 of moves catches this fraction)")
    out = Path("artifacts/experiments/decision_flip_probe.json")
    out.write_text(json.dumps(dict(rows=rows, coarse=args.coarse, deep=args.deep,
                                   flip_rate=float(flips.mean()),
                                   capture=float(cap))))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
