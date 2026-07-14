#!/usr/bin/env python
"""
experiments/conversion_compare.py — paired KRRvKBP conversion, checkpoint A vs B.

krrkbp_arena compares two POLICIES within one checkpoint; this compares the SAME
policy (FBSearchPolicy, matched nodes/beam) across TWO checkpoints, to answer the
play-truth question the curvature probe only proxies: did self-play fine-tuning
actually convert more KRRvKBP wins? Reuses krrkbp_arena.run_paired for the
matched-seed pairing + e-value/CI stats (each fixed position played by BOTH
checkpoints vs the same Stockfish with the same seed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catspace.diagnostic_krrkbp import load_fixed_set
from experiments.krrkbp_arena import run_paired
from catspace.uci import UCIBoardPolicy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-a", required=True, help="baseline (e.g. incumbent)")
    ap.add_argument("--ckpt-b", required=True, help="candidate (e.g. self-play R3)")
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_fixed_set_n60.json")
    ap.add_argument("--opponent", default="sf:skill=0")
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=150)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch  # noqa: F401
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.policy_fb import FBSearchPolicy

    device = pick_device(args.device)
    fb_a, pay_a = load_ckpt(Path(args.ckpt_a), device)
    fb_b, pay_b = load_ckpt(Path(args.ckpt_b), device)
    za = pay_a["zgoals"]["MATE_W"]
    zb = pay_b["zgoals"]["MATE_W"]

    positions = load_fixed_set(args.fixed_set)
    print(f"{len(positions)} positions; A={args.ckpt_a}  B={args.ckpt_b}  "
          f"nodes={args.nodes} vs {args.opponent}")

    def make_a():
        return FBSearchPolicy(fb_a, za, max_nodes=args.nodes, beam=args.beam, device=device)

    def make_b():
        return FBSearchPolicy(fb_b, zb, max_nodes=args.nodes, beam=args.beam, device=device)

    arg = args.opponent.split(":", 1)[1] if ":" in args.opponent else args.opponent
    opp = (UCIBoardPolicy(skill=int(arg[6:]), movetime=0.02) if arg.startswith("skill=")
           else UCIBoardPolicy(elo=int(arg), movetime=0.05))
    with opp:
        result = run_paired(make_a, make_b, "A_incumbent", "B_selfplay", positions,
                            opp, args.max_plies, args.seed, args.alpha,
                            early_stop=False, tablebase=None)
    lo, hi = result["diff_ci"]
    print(f"\nVERDICT conversion A={result['a_score_mean']:.3f} vs B={result['b_score_mean']:.3f}  "
          f"(n={result['n_positions']}, mean_diff={result['mean_diff']:+.3f} "
          f"CI=[{lo:+.3f},{hi:+.3f}], e={result['e_value']:.2f})")


if __name__ == "__main__":
    main()
