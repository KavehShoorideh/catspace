#!/usr/bin/env python
"""experiments/bench_engines_krrvk.py -- reference engines on the SAME 48 KRRvK-central
starts as the bootstrap exam (Kaveh: 'stockfish and leela'). White = the engine under test
(NO tablebase access), Black = tb-optimal defense (referee). Strength-per-node context for
the bootstrap engine's 0.96 @ 5000 evals/move.

Configs: sf5000 (nodes=5000) . sf100ms (movetime) . maia1900_5000 (lc0 weights, nodes=5000).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB, tb_best_move
from experiments.mate_ladder_eval import sample_scenarios

CONFIGS = {
    "sf5000": dict(cmd=["stockfish"], limit=chess.engine.Limit(nodes=5000)),
    "sf100ms": dict(cmd=["stockfish"], limit=chess.engine.Limit(time=0.1)),
    "maia1900_5000": dict(cmd=["lc0", "--weights=data/engines/maia/maia-1900.pb.gz"],
                          limit=chess.engine.Limit(nodes=5000)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default="sf5000,sf100ms,maia1900_5000")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); tb = TB()
    starts = dict(sample_scenarios(np.random.default_rng(args.seed), args.n))["KRRvK-central"]

    for cfg in args.configs.split(","):
        spec = CONFIGS[cfg]
        eng = chess.engine.SimpleEngine.popen_uci(spec["cmd"])
        res = []
        for gi, start in enumerate(starts):
            b = start.copy(stack=False)
            plies = 0; nodes = []
            while plies < args.max_plies and not b.is_game_over(claim_draw=True):
                if b.turn == chess.WHITE:
                    r = eng.play(b, spec["limit"], info=chess.engine.INFO_BASIC)
                    nodes.append(r.info.get("nodes", 0))
                    b.push(r.move)
                else:
                    b.push(tb_best_move(b, tb))
                plies += 1
            out = b.outcome(claim_draw=True)
            mated = bool(out and out.winner == chess.WHITE)
            res.append((mated, plies, int(np.median(nodes)) if nodes else 0))
            print(f"  [{cfg}] g{gi:03d} {'mate' if mated else 'FAIL'} plies={plies} "
                  f"med_nodes={res[-1][2]}  [{time.time()-t0:.0f}s]", flush=True)
        eng.quit()
        m = [r for r in res if r[0]]
        print(f"VERDICT BENCH cfg={cfg} mate={len(m)}/{len(res)} ({len(m)/len(res):.2f}) "
              f"med_plies={np.median([r[1] for r in m]) if m else float('nan'):.0f} "
              f"med_nodes/move={np.median([r[2] for r in res]):.0f}  [{time.time()-t0:.0f}s]",
              flush=True)
    tb.close()


if __name__ == "__main__":
    main()
