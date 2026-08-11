#!/usr/bin/env python
"""eval_sanity.py -- run an engine over the behavioral sanity suite (gen_sanity_suite.py).

Per category: mean grade of the CHOSEN move (0-10, STS-style) + top-grade hit rate.
  resist    -- does the lost side maximize the mate distance (prolong)?
  save-draw -- does the losing side take the forced draw?
  convert   -- does the winning side shorten the mate?

    .venv/bin/python -m ...eval_sanity --ckpt <ckpt.pt> [--depth 2] [--nav db|ab]
"""
from __future__ import annotations

import argparse
import json

import chess
import numpy as np

from catspace.io import paths
from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", default=None)
    ap.add_argument("--depth", type=int, default=0, help="0 = 1-ply choose(); else search depth")
    ap.add_argument("--nav", default="db", choices=("db", "ab"))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    suite_path = args.suite or paths.experiment("sanity_suite.jsonl")
    rows = [json.loads(l) for l in open(suite_path)]
    eng = KittyChess(args.ckpt, args.device, nav=args.nav)
    stats = {}
    for r in rows:
        b = chess.Board(r["fen"])
        if args.depth > 0:
            out = eng.search(b, depth=args.depth)
            mv = out[0]["mv"] if out else None
        else:
            mv = eng.choose(b)
        if mv is None:
            continue
        g = r["moves"].get(mv.uci(), 0.0)
        top = max(r["moves"].values())
        s = stats.setdefault(r["cat"], {"g": [], "hit": []})
        s["g"].append(g)
        s["hit"].append(float(g >= top - 0.5))
    print(f"[sanity] {args.ckpt}  nav={args.nav} depth={args.depth or '1-ply'}")
    for cat in ("resist", "save-draw", "convert"):
        if cat not in stats:
            continue
        s = stats[cat]
        # random baseline: expected grade of a uniform choice
        rnd = np.mean([np.mean(list(r["moves"].values())) for r in rows if r["cat"] == cat])
        print(f"  {cat:9s} n={len(s['g']):4d}  mean grade {np.mean(s['g']):5.2f}/10 "
              f"(random {rnd:.2f})  top-move rate {np.mean(s['hit']):.1%}")


if __name__ == "__main__":
    main()
