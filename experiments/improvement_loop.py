#!/usr/bin/env python
"""experiments/improvement_loop.py -- THE UNIFIED IMPROVEMENT LOOP (Kaveh 2026-07-25:
'this is only the nucleus. what about the iqe? the planner RL? I want them in the loop').

Game-bound learners, one loop, batch-of-10 cadence (see small-batch-retraining memory):
  each round:
    1. PLAY   -- 10 fullgames vs a rotating maia rung (+ experience recorded, probe
                 observations at plan decisions, banks growing, WIP-checkpointed)
    2. EXPORT -- experience store -> regime-11 shards
    3. LEARN  -- three learners, as data allows:
         a. IQE field   : fine-tune current field 2k steps w/ the self-play channel
                          -> self_field_r<k>.pt, pointer swap (banks re-embed at load)
         b. planner RL  : refit P(win | obs, plan) on ALL accumulated tuples
                          -> planner_rl_r<k>.pt (engine + assistant auto-load newest)
         c. (nucleus net trains on its own DATA-bound pipeline in parallel)
    4. The assistant hot-swaps everything without restarting; versions in the UI.
Stale-test enforcement + WIP checkpoints make every round killable and resumable."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.components.memory.approaches.experience_store.src.experience import ExperienceStore

PTR = Path("data/derived/sep/self_field_current.txt")
EPTR = Path("data/derived/sep/opponent_energy_current.txt")
MAIA = ["data/engines/maia/maia-1100.pb.gz", "data/engines/maia/maia-1200.pb.gz",
        "data/engines/maia/maia-1400.pb.gz", "data/engines/maia/maia-1600.pb.gz",
        "data/engines/maia/maia-1900.pb.gz"]   # full rung rotation (camping phase 2)


def field_now():
    return PTR.read_text().strip() if PTR.exists() else "data/derived/sep/lichess_mc2.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--nodes", type=int, default=2000)
    ap.add_argument("--field-steps", type=int, default=2000)
    ap.add_argument("--base-shards",
                    default="data/shards/lichess_db_standard_rated_2019-01.prefix4gb")
    ap.add_argument("--self-shards", default="data/shards/self_play_v1")
    args = ap.parse_args()
    t0 = time.time()
    store = ExperienceStore()
    bpfx = "artifacts/experiments/improve_persistent"

    for rnd in range(args.rounds):
        field = field_now()
        opp = MAIA[rnd % len(MAIA)]
        print(f"=== IMPROVE round {rnd} field={Path(field).stem} opp={Path(opp).stem} "
              f"[{time.time()-t0:.0f}s] ===", flush=True)
        while True:
            rc = subprocess.run([sys.executable, "experiments/bootstrap_mate_engine.py",
                                 "--scenario", "fullgame", "--n", str(args.games),
                                 "--j", "2", "--nodes", str(args.nodes),
                                 "--max-plies", "300", "--field", field,
                                 "--opponent-weights", opp,
                                 "--bank-file", f"{bpfx}_bank.fens",
                                 "--loss-bank-file", f"{bpfx}_lossbank.fens",
                                 "--draw-bank-file", f"{bpfx}_drawbank.fens",
                                 "--milestone-file", f"{bpfx}_ms.jsonl",
                                 "--results-file",
                                 f"artifacts/experiments/improve_r{rnd}_results.jsonl"]).returncode
            if rc != 75:
                break
            print("[improve] relaunch on new commit", flush=True)
        n = store.export_shards(args.self_shards, min_games=1)
        print(f"[improve] exported {n} games", flush=True)
        # a. IQE field fine-tune
        k = len(list(Path("data/derived/sep").glob("self_field_r*.pt")))
        fck = f"data/derived/sep/self_field_r{k}.pt"
        subprocess.run(["cp", field, fck], check=True)
        # --steps is ABSOLUTE and the ckpt resumes at its stored step: fine-tune target
        # = resume step + field_steps (else the round no-ops -- the carry-forward lesson)
        import torch as _t
        _base = int(_t.load(fck, map_location="cpu", weights_only=False).get("step", 0))
        rc = subprocess.run([sys.executable, "experiments/train_lichess_fb.py",
                            "--shards", args.base_shards,
                            "--steps", str(_base + args.field_steps), "--batch", "512",
                            "--iqe", "--l2-preset", "iqe-qrl", "--qrl-objective",
                            "--regime-channels", "16", "--regime-relative", "1",
                            "--regime-shards", f"{args.self_shards}:11:0.35",
                            "--ckpt", fck, "--ckpt-every", "1000",
                            "--val-every", "500", "--device", "auto"],
                           stdout=open(f"artifacts/experiments/improve_field_r{rnd}.log", "w"),
                           stderr=subprocess.STDOUT).returncode
        if rc == 0:
            PTR.write_text(fck)
            print(f"[improve] FIELD -> {fck}", flush=True)
        # b. planner RL refit on everything
        pk = len(list(Path("data/derived/sep").glob("planner_rl_r*.pt")))
        rc = subprocess.run([sys.executable, "experiments/train_planner_rl.py",
                            "--out", f"data/derived/sep/planner_rl_r{pk}.pt"],
                           stdout=open(f"artifacts/experiments/improve_rl_r{rnd}.log", "w"),
                           stderr=subprocess.STDOUT).returncode
        for line in open(f"artifacts/experiments/improve_rl_r{rnd}.log"):
            if "VERDICT" in line:
                print("  " + line.strip(), flush=True)
        # c. energy/opponent model: self cohort (elo 2800) + true maia rungs from the
        # experience export; fine-tune from the current model, lichess replay 65/35
        ms_npz = "data/derived/move_selection_self.npz"
        subprocess.run([sys.executable, "experiments/build_move_selection.py",
                        "--shards", args.self_shards, "--n-rows", "200000",
                        "--out", ms_npz],
                       stdout=open(f"artifacts/experiments/improve_energy_r{rnd}.log", "w"),
                       stderr=subprocess.STDOUT)
        import numpy as np
        n_self = len(np.load(ms_npz)["cohort"]) if Path(ms_npz).exists() else 0
        if n_self < 200:
            print(f"[improve] ENERGY skipped (self rows {n_self} < 200)", flush=True)
        else:
            energy = EPTR.read_text().strip() if EPTR.exists() \
                else "data/derived/sep/opponent_energy_v1.pt"
            ek = len(list(Path("data/derived/sep").glob("opponent_energy_r*.pt")))
            eck = f"data/derived/sep/opponent_energy_r{ek}.pt"
            rc = subprocess.run([sys.executable, "experiments/train_opponent_model.py",
                                "--data",
                                f"data/derived/move_selection_full_v1.npz:0.65,{ms_npz}:0.35",
                                "--init", energy, "--steps", "2000",
                                "--out", eck],
                               stdout=open(f"artifacts/experiments/improve_energy_r{rnd}.log", "a"),
                               stderr=subprocess.STDOUT).returncode
            if rc == 0:
                EPTR.write_text(eck)
                print(f"[improve] ENERGY -> {eck}", flush=True)
            for line in open(f"artifacts/experiments/improve_energy_r{rnd}.log"):
                if "VERDICT" in line:
                    print("  " + line.strip(), flush=True)
    print(f"=== IMPROVE done [{time.time()-t0:.0f}s] ===", flush=True)
    store.close()


if __name__ == "__main__":
    main()
