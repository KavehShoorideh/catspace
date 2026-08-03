#!/usr/bin/env python
"""catspace/research/components/planner/approaches/endgame_groundtruth/experiments/conversion_cornered.py -- USE the "exposed-cornered-king -> mate" concept as the goal, no
validation (Kaveh 2026-07-21: "assume the concept exists in the field; I just wanna use it").

Near-mate positions ARE the exposed-cornered-king concept by definition (you cannot mate a safe king). So
the mate goal = the B-cluster of low-DTM positions (the enemy king cornered AND exposed, about to be mated),
NOT the broad "winning-simplification" region. The planner then drives toward that B-cluster:
  * B carries the mate side (the exposed-cornered target region -- Kaveh: rely on B, it generalizes),
  * F does the LOCAL reach (short adversarial search leaf value = distance to the goal cluster),
  * the tablebase grounds the actual mate at the frontier.
That is the F/B split with the mate-half assumed rather than measured.

A/B on KRRvKBP conversion (tablebase-optimal defense), same field + search, differing only in the goal:
  A = base region (winning-simplification, from --data)   B = exposed-cornered-king region (low-DTM cluster)
Default field = the completed-trajectory field (best endgame metric).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch


from catspace.research.components.planner.approaches.subgoal_cascade.experiments.planner_longshort import LongShortPlanner
from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import tb_best_move
from catspace.io import paths


def play(planner, start, ply_cap):
    b = start.copy(stack=False)
    for _ in range(ply_cap):
        if b.is_game_over(claim_draw=True):
            return 1.0 if (b.is_checkmate() and b.turn == chess.BLACK) else 0.0
        m = planner.move(b) if b.turn == chess.WHITE else tb_best_move(b, planner.tb)
        if m is None:
            return 0.0
        b.push(m)
    return 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default=paths.sep("xfer_treat.pt"))
    ap.add_argument("--data", default=paths.derived("stratified_perfect.npz"))
    ap.add_argument("--dtm-npz", default=paths.derived("dtm_endgame.npz"))
    ap.add_argument("--fixed-set", default=paths.experiment("krrkbp_test_n200.json"))
    ap.add_argument("--syzygy", default=str(paths.syzygy_dir()))
    ap.add_argument("--frontier", type=int, default=5)
    ap.add_argument("--short-depth", type=int, default=3)
    ap.add_argument("--qdepth", type=int, default=0)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--ply-cap", type=int, default=80)
    ap.add_argument("--max-dtm", type=int, default=6, help="near-mate cutoff (plies) = exposed-cornered-king region")
    ap.add_argument("--region-n", type=int, default=800)
    ap.add_argument("--mode", default="both", choices=["both", "base", "cornered"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    fens = json.loads(Path(args.fixed_set).read_text())["fens"]
    starts = [chess.Board(f) for f in fens[args.offset:args.offset + args.n]]
    rng = np.random.default_rng(args.seed)

    print(f"VERDICT CONVERSION_CORNERED field={Path(args.field).stem} n={len(starts)} "
          f"max_dtm={args.max_dtm} short_depth={args.short_depth}", flush=True)
    res = {}
    if args.mode in ("both", "base"):
        p = LongShortPlanner(args.field, args.data, args.syzygy, args.frontier,
                             args.short_depth, args.qdepth, device=args.device, seed=args.seed)
        t1 = time.time()
        a = np.array([play(p, s, args.ply_cap) for s in starts])
        res["A_winning_region"] = a.mean()
        print(f"  A  goal=winning-simplification   mate_rate={a.mean():.3f}  ({time.time()-t1:.0f}s)", flush=True)
        p.close()
    if args.mode in ("both", "cornered"):
        p = LongShortPlanner(args.field, args.data, args.syzygy, args.frontier,
                             args.short_depth, args.qdepth, device=args.device, seed=args.seed)
        dz = np.load(args.dtm_npz)
        dtm = np.asarray(dz["dtm"]).astype(float)
        idx = np.flatnonzero((dtm > 0) & (dtm <= args.max_dtm))     # near-mate = exposed cornered king
        idx = idx[rng.permutation(len(idx))[:args.region_n]]
        p.B_goal = p._embB(dz["packed"][idx], dz["meta"][idx])       # REPLACE goal with the cornered cluster
        p._dcache.clear()
        print(f"  [cornered region: {len(idx)} near-mate positions (dtm<= {args.max_dtm}) embedded on B]", flush=True)
        t1 = time.time()
        b = np.array([play(p, s, args.ply_cap) for s in starts])
        res["B_cornered_region"] = b.mean()
        print(f"  B  goal=exposed-cornered-king     mate_rate={b.mean():.3f}  ({time.time()-t1:.0f}s)", flush=True)
        p.close()
    if "A_winning_region" in res and "B_cornered_region" in res:
        print(f"  DELTA cornered - base: {res['B_cornered_region'] - res['A_winning_region']:+.3f}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
