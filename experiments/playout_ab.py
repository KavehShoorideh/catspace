#!/usr/bin/env python
"""
experiments/playout_ab.py — paired PLAYOUT A/B with a DETERMINISTIC defender.

The lesson from move_ab: endgame play only diverges when each model drives its OWN
trajectory (fixed-position eval can't see it), and SF-conversion is too
high-variance (CI +-0.38 at n=200) because the opponent is stochastic. This plays
each model (White, hop search) against a TABLEBASE-OPTIMAL defender (Black,
deterministic -> zero opponent variance), from a set of winning starts, and scores
mate-within-budget. Because both the model (argmax) and the defender are
deterministic, the per-start result is exact and reproducible -- the paired diff
vs the incumbent has real power (variance only from which starts we sampled).

VERDICT: mate-rate A vs B, paired diff, bootstrap CI over starts, and mean
plies-to-mate among converted (lower = crisper conversion).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.value_fixed_point import TB, tb_best_move


def playout(pol, start, tb, rng, max_plies):
    """White = model (hop search), Black = tablebase-optimal. Return (mated, plies)."""
    b = start.copy(stack=False)
    seen = set()
    for ply in range(max_plies):
        if b.is_game_over(claim_draw=True):
            break
        if b.turn == chess.WHITE:
            m = pol.move(b, rng)
        else:
            m = tb_best_move(b, tb, seen); seen.add(b.board_fen())
        if m is None:
            break
        b.push(m)
    out = b.outcome(claim_draw=True)
    mated = 1.0 if (out and out.winner == chess.WHITE) else 0.0
    return mated, (b.ply() if mated else None)


def mate_vector(ckpt, starts, tb, nodes, beam, max_plies, seed, device, bank_boards=None):
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.policy_fb import FBSearchPolicy
    dev = pick_device(device)
    fb, pay = load_ckpt(Path(ckpt), dev)
    if bank_boards is not None:                          # region goal: soft-min over exemplars
        from catspace.goal_bank import embed_bank
        z = embed_bank(fb, bank_boards, dev)             # (m, d) -> FBSearchPolicy uses soft_min_bank
    else:
        z = pay["zgoals"]["MATE_W"]                      # centroid goal
    pol = FBSearchPolicy(fb, z, max_nodes=nodes, beam=beam, device=dev)
    mated, plies = [], []
    for i, fen in enumerate(starts):
        rng = np.random.default_rng([seed, i])
        m, p = playout(pol, chess.Board(fen), tb, rng, max_plies)
        mated.append(m)
        if p is not None:
            plies.append(p)
    return np.array(mated), (float(np.mean(plies)) if plies else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_test_n200.json")
    ap.add_argument("--n", type=int, default=100, help="number of starts to play")
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--label", default="")
    ap.add_argument("--ckpt-b-goal", choices=("centroid", "bank"), default="centroid",
                    help="goal used by ckpt-b's planner: centroid (zgoals MATE_W) or a soft-min "
                         "BANK of mate exemplars (Kaveh's 'arrive anywhere in the mate region')")
    ap.add_argument("--bank-shards", nargs="+", default=["data/selfplay/krrkbp_sfsf"])
    ap.add_argument("--bank-max-pieces", type=int, default=6)
    ap.add_argument("--bank-size", type=int, default=128)
    args = ap.parse_args()

    import torch  # noqa: F401
    tb = TB(args.syzygy_dir)
    starts = json.loads(Path(args.fixed_set).read_text())["fens"][:args.n]
    bank_boards = None
    if args.ckpt_b_goal == "bank":
        from catspace.goal_bank import harvest_mate_finals
        bank_boards = harvest_mate_finals(args.bank_shards, want_result=1,
                                          max_pieces=args.bank_max_pieces, cap=args.bank_size)
        print(f"goal bank: {len(bank_boards)} white-mate exemplars (<= {args.bank_max_pieces} pieces)")
    a, pa = mate_vector(args.ckpt_a, starts, tb, args.nodes, args.beam, args.max_plies,
                        args.seed, args.device)
    b, pb = mate_vector(args.ckpt_b, starts, tb, args.nodes, args.beam, args.max_plies,
                        args.seed, args.device, bank_boards=bank_boards)
    tb.close()
    n = len(starts)
    diff = float(b.mean() - a.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(args.boot, n))
    boot = b[idx].mean(1) - a[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sig = (lo > 0 or hi < 0)
    print(f"PLAYOUT_AB {args.label} mate-rate A={a.mean():.3f} vs B={b.mean():.3f}  "
          f"diff={diff:+.3f} CI=[{lo:+.3f},{hi:+.3f}]  "
          f"(n={n} starts, deterministic defender; plies-to-mate A={pa:.0f} B={pb:.0f}) "
          f"[{'SIGNIFICANT' if sig else 'ns'}]")


if __name__ == "__main__":
    main()
