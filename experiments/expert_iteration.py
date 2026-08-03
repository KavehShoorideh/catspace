#!/usr/bin/env python
"""experiments/expert_iteration.py -- the mate mission's legitimate engine (DECISIONS 4b):
the value net improves from ITS OWN PLAY. Round r: current net + MCTS plays the graded
scenarios vs tablebase-optimal defense (the referee); every White-to-move position in a WON
game gets labeled with the ACTUAL plies-to-mate that followed (experience, not design);
harvest accumulates; the net retrains on base tb-truth + harvest; the exam re-runs.

No hand-coded concepts anywhere. Tablebase appears only as (a) the opponent/exam and
(b) the base DTM ground truth (game-outcome structure, long-established as training data).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB, tb_best_move
from experiments.ladder_mate import make_dtm_value, white_mcts
from experiments.mate_ladder_eval import sample_scenarios

PY = sys.executable


def harvest_round(value_ckpt, rng, tb, n_per_scenario, nodes, max_plies, active_levels):
    """Play the ACTIVE curriculum levels (Kaveh: KRRvK first -- master the easy material,
    then unlock); return (packed, meta, plies_to_mate) from WON games."""
    vfn = make_dtm_value(value_ckpt)
    scenarios = sample_scenarios(rng, n_per_scenario)[:active_levels]
    PK, MT, Y = [], [], []
    stats = {}
    for name, starts in scenarios:
        won = 0
        for s in starts:
            b = s.copy(stack=False)
            trail = []                      # (position, ply) at White's turns
            ply = 0
            for _ in range(max_plies):
                if b.is_game_over(claim_draw=True):
                    break
                if b.turn == chess.WHITE:
                    trail.append((b.copy(stack=False), ply))
                    mv, _ev = white_mcts(b, nodes, vfn, None)
                    b.push(mv)
                else:
                    b.push(tb_best_move(b, tb))
                ply += 1
            out = b.outcome(claim_draw=True)
            if out and out.winner == chess.WHITE:
                won += 1
                T = ply
                for pos, t in trail:
                    PK.append(encode_packed(pos)); MT.append(encode_meta(pos))
                    Y.append(float(T - t))
        stats[name] = f"{won}/{len(starts)}"
    return PK, MT, Y, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--base-ckpt", default="data/derived/sep/dtm_cnn_v2.pt")
    ap.add_argument("--base-data", default="data/derived/dtm_endgame_v2.npz")
    ap.add_argument("--n-per-scenario", type=int, default=40)
    ap.add_argument("--nodes", type=int, default=600)
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--retrain-steps", type=int, default=5000)
    ap.add_argument("--harvest-out", default="data/derived/ei_harvest.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed); tb = TB()
    t0 = time.time()

    all_pk, all_mt, all_y = [], [], []
    ckpt = args.base_ckpt
    active = 1                                     # curriculum: level 1 = KRRvK-central only
    PROMOTE = 0.60                                 # frontier mate-rate to unlock the next level
    for r in range(1, args.rounds + 1):
        print(f"=== ROUND {r}: levels 1..{active}, harvesting with {Path(ckpt).name} ===", flush=True)
        PK, MT, Y, stats = harvest_round(ckpt, rng, tb, args.n_per_scenario,
                                         args.nodes, args.max_plies, active)
        all_pk += PK; all_mt += MT; all_y += Y
        print(f"VERDICT EI_HARVEST round={r} rows=+{len(PK)} (total {len(all_pk)})  "
              f"wins: {stats}  [{time.time()-t0:.0f}s]", flush=True)
        if not all_pk:
            print("no wins to harvest -- stopping (bootstrap failure)", flush=True)
            break
        np.savez_compressed(args.harvest_out, packed=np.stack(all_pk),
                            meta=np.stack(all_mt), dtm=np.array(all_y, np.float32))
        # merge base + harvest into the round's training set
        base = np.load(args.base_data)
        merged = str(Path("/Users/kav/.claude/jobs/20b9956a/tmp") / f"ei_train_r{r}.npz")
        np.savez_compressed(merged,
                            packed=np.concatenate([base["packed"], np.stack(all_pk)]),
                            meta=np.concatenate([base["meta"], np.stack(all_mt)]),
                            dtm=np.concatenate([base["dtm"].astype(np.float32),
                                                np.array(all_y, np.float32)]))
        new_ckpt = f"data/derived/sep/dtm_cnn_ei_r{r}.pt"
        rc = subprocess.run([PY, "experiments/train_dtm_cnn.py", "--dtm-npz", merged,
                             "--steps", str(args.retrain_steps), "--out", new_ckpt],
                            capture_output=True, text=True)
        print(rc.stdout[-500:], flush=True)
        if rc.returncode != 0:
            print(f"retrain FAILED: {rc.stderr[-300:]}", flush=True)
            break
        ckpt = new_ckpt
        # the exam
        rc = subprocess.run([PY, "experiments/mate_ladder_eval.py", "--configs", "dtm",
                             "--dtm-ckpt", ckpt, "--n", "16", "--nodes", str(args.nodes)],
                            capture_output=True, text=True)
        frontier_rate = None
        for ln in rc.stdout.splitlines():
            if "VERDICT" in ln:
                print(ln.replace("MATE_LADDER", f"MATE_LADDER[EI r{r}]"), flush=True)
                if "mate=" in ln:
                    import re as _re
                    scen_names = ["KRRvK-central", "KRRvKB", "KRRvKP", "KRRvKBP", "KRvK-technique"]
                    if active <= len(scen_names) and scen_names[active - 1] in ln:
                        frontier_rate = float(_re.search(r"mate=([0-9.]+)", ln).group(1))
        if frontier_rate is not None and frontier_rate >= PROMOTE and active < 5:
            active += 1
            print(f"[curriculum] frontier mastered ({frontier_rate:.2f} >= {PROMOTE}) -> "
                  f"LEVEL {active} UNLOCKED", flush=True)
    tb.close()
    print(f"VERDICT EXPERT_ITERATION rounds_done final={Path(ckpt).name} "
          f"harvest={len(all_pk)} rows  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
