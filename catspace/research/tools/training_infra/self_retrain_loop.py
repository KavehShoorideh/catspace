#!/usr/bin/env python
"""catspace/research/tools/training_infra/self_retrain_loop.py -- THE SELF-RETRAINING LOOP (Kaveh 2026-07-25:
'a retraining loop after every N games where we add all our searched positions in those
games to the indexed banks and train our quasimetric on them and use them to their fullest').

Each round:
  1. PLAY  -- bootstrap engine, --games-per-round games (experience store records every
     game + searched roots + provenance: engine commit, field ckpt, timestamps)
  2. EXPORT -- store -> npz shard (regime-rollouts schema, regime=11 self-play):
     train_lichess_fb ingests own-play with ZERO changes
  3. RETRAIN (when new games >= --retrain-every) -- fine-tune the CURRENT field for
     --retrain-steps with the self-play channel mixed in; new ckpt on the ladder
     (self_field_r<k>.pt, never overwritten), MLflow-tracked by the trainer itself
  4. SWAP -- pointer file advances; next round's engine loads the new field; the banks
     persist as FENs and re-embed under the new field automatically (facts survive
     engine change; embeddings are per-field)

Short rounds, fail-fast (short-runs rule); every round journaled by its VERDICT lines."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


from catspace.research.components.memory.approaches.experience_store.src.experience import ExperienceStore
from catspace.io import paths

PTR = Path(paths.sep("self_field_current.txt"))


def current_field(default=paths.sep("lichess_mc2.pt")) -> str:
    return PTR.read_text().strip() if PTR.exists() else default


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="KRRvK-central")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--games-per-round", type=int, default=10)
    ap.add_argument("--retrain-every", type=int, default=10, help="new games per retrain")
    ap.add_argument("--retrain-steps", type=int, default=2000)
    ap.add_argument("--self-frac", type=float, default=0.3,
                    help="self-play channel fraction in the retrain mix")
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--j", type=int, default=4)
    ap.add_argument("--base-shards", default=paths.shards("lichess_db_standard_rated_2019-01.prefix4gb"))
    ap.add_argument("--self-shards", default=paths.shards("self_play_v1"))
    ap.add_argument("--tag", default="selfloop")
    args = ap.parse_args()
    t0 = time.time()
    store = ExperienceStore()
    since_retrain = 0

    for rnd in range(args.rounds):
        field = current_field()
        print(f"=== SELFLOOP round {rnd} field={field} [{time.time()-t0:.0f}s] ===", flush=True)
        # ONE growing bank across all rounds (Kaveh: 'build one bank and keep reusing it
        # without resetting') -- banks are facts, immortal across field swaps (FENs
        # re-embed at load); results file stays per-round for per-round verdicts.
        bpfx = paths.experiment(f"{args.tag}_persistent")
        pfx = paths.experiment(f"{args.tag}_r{rnd}")
        subprocess.run([sys.executable, "catspace/approaches/bootstrap_mate/experiments/bootstrap_mate_engine.py",
                        "--scenario", args.scenario, "--nodes", str(args.nodes),
                        "--n", str(args.games_per_round), "--j", str(args.j),
                        "--max-plies", "120", "--field", field,
                        "--bank-file", f"{bpfx}_bank.fens",
                        "--loss-bank-file", f"{bpfx}_lossbank.fens",
                        "--draw-bank-file", f"{bpfx}_drawbank.fens",
                        "--milestone-file", f"{bpfx}_ms.jsonl",
                        "--results-file", f"{pfx}_results.jsonl"],
                       check=True)
        n = store.export_shards(args.self_shards, min_games=1)
        since_retrain += n
        print(f"[selfloop] exported {n} games -> {args.self_shards} "
              f"({since_retrain}/{args.retrain_every} toward retrain)", flush=True)
        if since_retrain >= args.retrain_every:
            since_retrain = 0
            k = len(list(Path(str(paths.sep_dir())).glob("self_field_r*.pt")))
            new_ckpt = paths.sep(f"self_field_r{k}.pt")
            subprocess.run(["cp", field, new_ckpt], check=True)
            print(f"[selfloop] RETRAIN {field} -> {new_ckpt} ({args.retrain_steps} steps, "
                  f"self-frac {args.self_frac})", flush=True)
            subprocess.run([sys.executable, "catspace/research/components/encoder/approaches/cone_fb_embedding/experiments/train_lichess_fb.py",
                            "--shards", args.base_shards,
                            "--steps", str(args.retrain_steps), "--batch", "512",
                            "--iqe", "--l2-preset", "iqe-qrl", "--qrl-objective",
                            "--regime-channels", "16", "--regime-relative", "1",
                            "--regime-shards", f"{args.self_shards}:11:{args.self_frac}",
                            "--ckpt", new_ckpt, "--ckpt-every", "1000",
                            "--val-every", "500", "--device", "auto"],
                           check=True)
            PTR.write_text(new_ckpt)
            print(f"[selfloop] SWAP -> {new_ckpt}", flush=True)
    print(f"=== SELFLOOP done ({args.rounds} rounds) [{time.time()-t0:.0f}s] ===", flush=True)
    store.close()


if __name__ == "__main__":
    main()
