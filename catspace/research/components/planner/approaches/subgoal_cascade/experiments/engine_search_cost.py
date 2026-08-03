#!/usr/bin/env python
"""catspace/research/components/planner/approaches/subgoal_cascade/experiments/engine_search_cost.py -- how much SEARCH does a real UCI engine (Stockfish,
or Leela via --engine lc0 + a weights file) spend to convert our endgames? Drives the
engine as White vs tablebase-optimal defense, summing info["nodes"] over the game -> total
nodes-to-mate. This is the external baseline for the efficiency thesis (DECISIONS.md sec 3,
reach_efficiency.py): how many nodes a strong EVAL needs, versus our uniform-prior /
field-guided search which lacks that eval.

Per-move it also records the single-decision node count at a fixed depth. Engine-agnostic:
--engine stockfish (default, /opt/homebrew/bin/stockfish) or a path to lc0 (needs --uci
'WeightsFile=...' style options are set via --uci-option KEY=VALUE, repeatable).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np


from catspace.research.components.planner.approaches.endgame_groundtruth.experiments.ladder_mate import random_krrvk
from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import TB, tb_best_move
from catspace.io import paths


def convert_cost(engine, start, tb, depth, max_plies):
    """Play White (engine) vs tablebase-optimal Black; sum nodes over White's moves."""
    b = start.copy(stack=False)
    total_nodes = 0; per_move = []; plies = 0
    while not b.is_game_over(claim_draw=True) and plies < max_plies:
        if b.turn == chess.WHITE:
            info = engine.analyse(b, chess.engine.Limit(depth=depth))
            n = int(info.get("nodes", 0)); total_nodes += n; per_move.append(n)
            mv = info["pv"][0] if info.get("pv") else next(iter(b.legal_moves))
            b.push(mv)
        else:
            b.push(tb_best_move(b, tb))
        plies += 1
    out = b.outcome(claim_draw=True)
    mated = bool(out and out.winner == chess.WHITE)
    return mated, total_nodes, plies, per_move


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--uci-option", action="append", default=[], help="KEY=VALUE, repeatable (e.g. Threads=1)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--depth", type=int, default=18, help="per-move search depth (fixed) -> node count")
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--set", choices=["ladder", "toy"], default="ladder",
                    help="ladder=KRRvK two-rook mate; toy=KRRvKBP full conversion")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); tb = TB(str(paths.syzygy_dir()))
    rng = np.random.default_rng(args.seed)

    if args.set == "ladder":
        starts = [s for s in (random_krrvk(rng, central=True) for _ in range(args.n)) if s is not None]
    else:
        import json
        fens = json.loads(Path(paths.experiment("krrkbp_test_n200.json")).read_text())["fens"]
        starts = [chess.Board(f) for f in fens[:args.n]]

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    opts = dict(kv.split("=", 1) for kv in args.uci_option)
    if opts:
        engine.configure({k: (int(v) if v.isdigit() else v) for k, v in opts.items()})
    idn = engine.id.get("name", args.engine)
    print(f"[engine] {idn}  set={args.set}  n={len(starts)}  depth={args.depth}", flush=True)

    rows = [convert_cost(engine, s, tb, args.depth, args.max_plies) for s in starts]
    engine.quit(); tb.close()

    mated = [r for r in rows if r[0]]
    tot = np.array([r[1] for r in mated]) if mated else np.array([0])
    allmoves = np.array([n for r in rows for n in r[3]])
    rate = len(mated) / len(rows)
    print(f"VERDICT ENGINE_SEARCH engine={idn.split()[0]} set={args.set} depth={args.depth}  "
          f"mate_rate={rate:.2f} ({len(mated)}/{len(rows)})  "
          f"nodes_to_mate median={np.median(tot):,.0f} (p10 {np.percentile(tot,10):,.0f} / p90 {np.percentile(tot,90):,.0f})  "
          f"per_move median={np.median(allmoves):,.0f}  "
          f"median_plies={np.median([r[2] for r in mated]) if mated else float('nan'):.0f}  "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
