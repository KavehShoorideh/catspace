#!/usr/bin/env python
"""
experiments/adaptive_vs_uniform.py — the PLAY test of reliability-gated search
(2026-07-13, Kaveh: validate by play, not by a label).

Runs FBAdaptiveSearchPolicy (Method-1 + Method-2 gated) on the KRRvKBP fixed set,
measures its AVERAGE nodes/move, then runs a UNIFORM FBSearchPolicy at that same
average budget on the same positions. If gating beats uniform AT MATCHED COMPUTE,
searching-more-where-unreliable is a real lever (unlike uniform more-search, which
the node sweep showed is not).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catspace.competence import CompetenceMap
from catspace.diagnostic_krrkbp import load_fixed_set
from catspace.io.paths import derived_dir
from catspace.realboard import play_board_game
from catspace.uci import UCIBoardPolicy

SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5, "*": 0.5}


def run_scan(make_policy, positions, opp, max_plies, seed):
    scores, nodes = [], []
    with opp:
        for i, start in enumerate(positions):
            pol = make_policy()
            rng = np.random.default_rng([seed, i])
            per_move_nodes = []
            # wrap move() to record nodes when it's an adaptive policy
            rec = play_board_game(_NodeLogger(pol, per_move_nodes), opp,
                                  start=start.copy(stack=False), opening_plies=0,
                                  max_plies=max_plies, rng=rng)
            scores.append(SCORE[rec.result])
            if per_move_nodes:
                nodes.append(np.mean(per_move_nodes))
    return float(np.mean(scores)), (float(np.mean(nodes)) if nodes else None), scores


class _NodeLogger:
    """Wrap a policy; after each move, log last_nodes_used if present."""
    def __init__(self, pol, sink):
        self.pol = pol
        self.sink = sink

    def move(self, board, rng):
        mv = self.pol.move(board, rng)
        self.sink.append(getattr(self.pol, "last_nodes_used", getattr(self.pol, "max_nodes", 0)))
        return mv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/lichess_fb_4gb_qm_plygap_only.pt")
    ap.add_argument("--competence", default="data/derived/competence_map.npz")
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_fixed_set_n60.json")
    ap.add_argument("--opponent", default="sf:skill=0")
    ap.add_argument("--base-nodes", type=int, default=200)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--sharp-thresh", type=float, default=0.15)
    ap.add_argument("--node-cap", type=int, default=1600)
    ap.add_argument("--max-plies", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch  # noqa: F401
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.policy_fb import FBAdaptiveSearchPolicy, FBSearchPolicy

    device = pick_device(args.device)
    fb, payload = load_ckpt(Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt", device)
    z = payload["zgoals"]["MATE_W"]
    cmap = CompetenceMap.load(args.competence)
    positions = load_fixed_set(args.fixed_set)
    arg = args.opponent[3:]
    opp = (UCIBoardPolicy(skill=int(arg[6:]), movetime=0.02) if arg.startswith("skill=")
           else UCIBoardPolicy(elo=int(arg), movetime=0.05))

    print(f"ADAPTIVE (base={args.base_nodes}, cap={args.node_cap}, thresh={args.sharp_thresh}) "
          f"vs {args.opponent}, n={len(positions)}", flush=True)
    ad_score, ad_nodes, _ = run_scan(
        lambda: FBAdaptiveSearchPolicy(fb, z, base_nodes=args.base_nodes, beam=args.beam,
                                       competence_map=cmap, sharp_thresh=args.sharp_thresh,
                                       node_cap=args.node_cap, device=device),
        positions, opp, args.max_plies, args.seed)
    matched = int(round(ad_nodes)) if ad_nodes else args.base_nodes
    print(f"  adaptive: score={ad_score:.3f}  avg_nodes/move={ad_nodes:.0f}", flush=True)

    print(f"UNIFORM at matched budget max_nodes={matched}", flush=True)
    un_score, _, _ = run_scan(
        lambda: FBSearchPolicy(fb, z, max_nodes=matched, beam=args.beam, device=device),
        positions, opp, args.max_plies, args.seed)
    print(f"  uniform:  score={un_score:.3f}  max_nodes={matched}", flush=True)

    print(f"\nVERDICT adaptive={ad_score:.3f} vs uniform@matched={un_score:.3f}  "
          f"(delta={ad_score - un_score:+.3f}; >0 means gating helps at equal compute)")


if __name__ == "__main__":
    main()
