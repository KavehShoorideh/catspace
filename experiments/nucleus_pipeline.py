#!/usr/bin/env python
"""experiments/nucleus_pipeline.py -- PROGRESSIVE GENERAL-NUCLEUS PIPELINE (Kaveh
2026-07-25: 'sample a bit from each class, then progressively train as more data arrives.
pipeline it!... what I want is an engine that can handle the whole board, all states, to
some extent, then improve').

Rounds of breadth-first coverage:
  round k: GENERATE a per-class delta quota across ALL tablebase classes (parallel
           workers, fresh seeds per round)  ||  in PARALLEL, TRAIN the token net on all
           rounds accumulated so far, then FLIP the engine default by commit (the stale
           enforcement self-heals the fleet onto it, WIP checkpoints preserve games).
After round 0 (~small quota x 149 classes) the engine covers every class 'to some
extent'; each round deepens. Gen(k+1) overlaps Train(k): a true pipeline."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DELTAS = [300, 700, 2000, 3000, 6000]      # cumulative per-class: 300/1k/3k/6k/12k


def run(cmd, log):
    with open(log, "a") as f:
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=len(DELTAS))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out-dir", default="data/derived/nucleus")
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    gen_proc = None
    for k in range(args.rounds):
        delta = DELTAS[min(k, len(DELTAS) - 1)]
        shard = out / f"round_{k}.npz"
        if not shard.exists():
            if gen_proc is None:                 # round 0 gen, or resume
                gen_proc = run([sys.executable, "experiments/gen_dtm_data.py",
                                "--materials", "all", "--per", str(delta),
                                "--seed", str(1000 * k), "--workers", str(args.workers),
                                "--out", str(shard)],
                               f"artifacts/experiments/nucleus_gen_r{k}.log")
            print(f"[pipeline] waiting gen round {k} (delta {delta}/class) "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            gen_proc.wait()
            gen_proc = None
        # PIPELINE: launch NEXT round's gen before training this round
        if k + 1 < args.rounds:
            nshard = out / f"round_{k+1}.npz"
            if not nshard.exists():
                ndelta = DELTAS[min(k + 1, len(DELTAS) - 1)]
                gen_proc = run([sys.executable, "experiments/gen_dtm_data.py",
                                "--materials", "all", "--per", str(ndelta),
                                "--seed", str(1000 * (k + 1)), "--workers", str(args.workers),
                                "--out", str(nshard)],
                               f"artifacts/experiments/nucleus_gen_r{k+1}.log")
                print(f"[pipeline] gen round {k+1} launched in parallel", flush=True)
        # TRAIN on all shards so far
        shards = ",".join(str(out / f"round_{i}.npz") for i in range(k + 1))
        ckpt = f"data/derived/sep/dtm_tok_r{k}.pt"
        print(f"[pipeline] training round {k} on {k+1} shard(s) [{time.time()-t0:.0f}s]",
              flush=True)
        rc = subprocess.run([sys.executable, "experiments/train_dtm_tok.py",
                             "--dtm-npz", shards, "--steps", str(4000 + 2000 * k),
                             "--out", ckpt],
                            stdout=open(f"artifacts/experiments/nucleus_train_r{k}.log", "w"),
                            stderr=subprocess.STDOUT).returncode
        if rc != 0:
            print(f"[pipeline] TRAIN round {k} FAILED rc={rc}; continuing gen, skipping flip",
                  flush=True)
            continue
        for line in open(f"artifacts/experiments/nucleus_train_r{k}.log"):
            if "VERDICT" in line:
                print("  " + line.strip(), flush=True)
        # FLIP the engine default -> fleet self-heals
        eng = Path("catspace/approaches/bootstrap_mate/experiments/bootstrap_mate_engine.py")
        src = eng.read_text()
        import re
        new = re.sub(r'(--last-mile-dtm",\s*default=")[^"]*(")',
                     rf'\g<1>{ckpt}\g<2>', src)
        if new != src:
            eng.write_text(new)
            subprocess.run(["git", "add", str(eng)])
            subprocess.run(["git", "commit", "-m",
                            f"nucleus pipeline round {k}: flip last-mile default to {ckpt} "
                            f"(fleet self-heals)", "--quiet"])
            print(f"[pipeline] FLIPPED default -> {ckpt} (committed)", flush=True)
    print(f"[pipeline] complete [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
