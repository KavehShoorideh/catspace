#!/usr/bin/env python
"""
experiments/certainty_rollouts.py — per-state P(reach mate) from stochastic rollouts.

Kaveh 2026-07-14: closeness should = CERTAINTY of transition -- "the closest path
is the one where we're more certain"; a messy position with one winning line is
NOT closer to mate than a clearly-winning one slightly farther. The clean object:
d(s, mate) = plies + lambda * (-ln P(reach mate from s)), where -ln P is itself a
quasimetric (P chains multiplicatively -> -log P is subadditive). Forced mate:
P=1 -> pure plies. Messy: P<1 -> inflated distance.

This estimates P-hat by MCTS-style stochastic rollouts on the KRRvKBP toy:
White = incumbent hop-search policy with epsilon noise (the plausible-play
distribution -- ITS unreliability is exactly the uncertainty we want priced in),
Black = deterministic tablebase-optimal defender. Every state visited on any
rollout is keyed by FEN; across rollouts we aggregate: n visits, wins, mean
remaining plies to the mate when won. Output: JSON rows (fen, n, p_hat,
mean_plies_to_mate) for the certainty-distance trainer. Unseen positions are the
NETWORK's job (it generalises the field; the competence head flags where not to
trust it -> search there -> the closed loop).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.value_fixed_point import TB, tb_best_move


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/lichess_fb_4gb_qm_plygap_only.pt")
    ap.add_argument("--starts", default="artifacts/experiments/krrkbp_win_starts.json")
    ap.add_argument("--n-starts", type=int, default=120)
    ap.add_argument("--rollouts", type=int, default=30, help="rollouts per start")
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--white", choices=("model", "tb"), default="model",
                    help="White's base policy: the incumbent model, or tb=tablebase-optimal "
                         "(with epsilon slips) -- certainty then reflects how FORGIVING the "
                         "position itself is under competent-but-fallible play")
    ap.add_argument("--nodes", type=int, default=100)
    ap.add_argument("--max-plies", type=int, default=100)
    ap.add_argument("--min-visits", type=int, default=2, help="keep states with >= this many visits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="artifacts/experiments/certainty_table.json")
    args = ap.parse_args()

    import torch  # noqa: F401
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.policy_fb import FBSearchPolicy

    dev = pick_device(args.device)
    fb, pay = load_ckpt(Path(args.ckpt), dev)
    pol = FBSearchPolicy(fb, pay["zgoals"]["MATE_W"], max_nodes=args.nodes, beam=4, device=dev)
    tb = TB("data/syzygy")
    starts = json.loads(Path(args.starts).read_text())["fens"][:args.n_starts]

    stats = defaultdict(lambda: [0, 0, []])          # fen -> [visits, wins, plies_to_mate list]
    for si, fen in enumerate(starts):
        for r in range(args.rollouts):
            rng = np.random.default_rng([args.seed, si, r])
            b = chess.Board(fen)
            traj = []
            seen = set()
            for _ in range(args.max_plies):
                if b.is_game_over(claim_draw=True):
                    break
                traj.append((b.fen(), b.ply()))
                if b.turn == chess.WHITE:
                    if rng.random() < args.epsilon:
                        moves = list(b.legal_moves)
                        m = moves[int(rng.integers(len(moves)))]
                    elif args.white == "tb":
                        m = tb_best_move(b, tb, seen)
                    else:
                        m = pol.move(b, rng)
                else:
                    m = tb_best_move(b, tb, seen)
                    seen.add(b.board_fen())
                b.push(m)
            out = b.outcome(claim_draw=True)
            won = bool(out and out.winner == chess.WHITE)
            end_ply = b.ply()
            for f, p in traj:
                s = stats[f]
                s[0] += 1
                if won:
                    s[1] += 1
                    s[2].append(end_ply - p)
        if (si + 1) % 20 == 0:
            print(f"  start {si+1}/{len(starts)}: {len(stats)} states tracked", flush=True)
    tb.close()

    rows = [dict(fen=f, n=v[0], p_hat=v[1] / v[0],
                 plies=(float(np.mean(v[2])) if v[2] else None))
            for f, v in stats.items() if v[0] >= args.min_visits]
    p = np.array([r["p_hat"] for r in rows])
    print(f"{len(rows)} states with >= {args.min_visits} visits; "
          f"P-hat mean {p.mean():.2f}, P=1 frac {(p == 1).mean():.2f}, P=0 frac {(p == 0).mean():.2f}")
    Path(args.out).write_text(json.dumps(dict(epsilon=args.epsilon, rollouts=args.rollouts,
                                              rows=rows)))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
