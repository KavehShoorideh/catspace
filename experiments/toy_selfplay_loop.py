#!/usr/bin/env python
"""
experiments/toy_selfplay_loop.py — the closed loop on ONE toy scenario.

Kaveh, 2026-07-13: "let's do self-play of this toy scenario, and see if the
model improves by self-play of a specific scenario ... what we want to keep
track of is how much curvature starts to appear in the reachability space where
we want it as we proceed in self-play. I want to see the sensitivity."

So: repeatedly (a) generate self-play games launched from the KRRvKBP fixed set
using the CURRENT checkpoint, (b) fine-tune the embedding on the accumulated
self-play (mixed with a little human data to avoid catastrophic forgetting),
(c) measure the reach-field curvature on the fixed KRRvKBP set. The curvature
record after each round stacks into a sensitivity-vs-round trajectory
(artifacts/experiments/reach_curvature.jsonl), which is the whole point:
does a usable gradient appear in the region where the drill-down found the
field flat?

Each round's self-play is generated under the freshest checkpoint, so as the
field sharpens the games should reach mate more often -> stronger signal -> more
curvature: the feedback we want to watch. Replay is CUMULATIVE (every round's
shards are kept) so early good games aren't forgotten.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def sh(cmd, log):
    log.write(f"\n$ {' '.join(str(c) for c in cmd)}\n"); log.flush()
    p = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        log.write(f"\n!! command failed (rc={p.returncode}); aborting loop\n"); log.flush()
        raise SystemExit(p.returncode)


def base_step(ckpt):
    import torch
    return int(torch.load(ckpt, map_location="cpu", weights_only=False).get("step", 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init-ckpt", default="data/derived/lichess_fb_4gb_qm_plygap_only.pt")
    ap.add_argument("--human-shards", default="data/shards/lichess_db_standard_rated_2019-01.prefix4gb")
    ap.add_argument("--start-fens", default="artifacts/experiments/krrkbp_fixed_set_n60.json")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--games", type=int, default=250)
    ap.add_argument("--steps-per-round", type=int, default=5000)
    ap.add_argument("--selfplay-frac", type=float, default=0.7)
    ap.add_argument("--gen-nodes", type=int, default=100)
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--sf-opponent-frac", type=float, default=0.5)
    ap.add_argument("--sf-skill", type=int, default=2)
    ap.add_argument("--replay-dir", default="data/selfplay/krrkbp_loop")
    ap.add_argument("--work-dir", default="data/derived/krrkbp_loop")
    ap.add_argument("--log", default="artifacts/generated/toy_selfplay_loop.log")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    replay = (ROOT / args.replay_dir); replay.mkdir(parents=True, exist_ok=True)
    work = (ROOT / args.work_dir); work.mkdir(parents=True, exist_ok=True)
    logp = (ROOT / args.log); logp.parent.mkdir(parents=True, exist_ok=True)

    base = base_step(ROOT / args.init_ckpt)
    current = ROOT / args.init_ckpt
    shard_ctr = len(list(replay.glob("shard_*.npz")))

    with logp.open("a") as log:
        log.write(f"\n===== toy self-play loop: {args.rounds} rounds, {args.games} games/round, "
                  f"+{args.steps_per_round} steps/round, selfplay-frac={args.selfplay_frac}, "
                  f"base_step={base}, init={args.init_ckpt} =====\n"); log.flush()

        for r in range(1, args.rounds + 1):
            log.write(f"\n########## ROUND {r} ##########\n"); log.flush()

            # (a) generate self-play from KRRvKBP starts with the CURRENT ckpt
            tmp = ROOT / f"data/selfplay/_krrkbp_r{r}_tmp"
            if tmp.exists():
                shutil.rmtree(tmp)
            sh([PY, "experiments/selfplay_generate.py", "--ckpt", str(current),
                "--out-dir", str(tmp), "--start-fens", args.start_fens,
                "--games", str(args.games), "--max-nodes", str(args.gen_nodes),
                "--beam", "4", "--epsilon", str(args.epsilon),
                "--sf-opponent-frac", str(args.sf_opponent_frac),
                "--sf-skill", str(args.sf_skill), "--max-plies", "120",
                "--device", args.device], log)
            # accumulate into the cumulative replay dir with unique names
            for f in sorted(tmp.glob("shard_*.npz")):
                shutil.move(str(f), str(replay / f"shard_{shard_ctr:05d}.npz"))
                shard_ctr += 1
            shutil.rmtree(tmp, ignore_errors=True)

            # (b) fine-tune from the current ckpt on accumulated self-play + human
            round_ckpt = work / f"krrkbp_loop_r{r}.pt"
            shutil.copy(current, round_ckpt)
            total_steps = base + r * args.steps_per_round
            sh([PY, "experiments/train_lichess_fb.py", "--ckpt", str(round_ckpt),
                "--shards", args.human_shards, "--steps", str(total_steps),
                "--quasimetric", "--ply-gap-weight", "0.05",
                "--selfplay-shards", str(replay), "--selfplay-frac", str(args.selfplay_frac),
                "--device", args.device], log)
            current = round_ckpt

            # (c) measure curvature on the fixed KRRvKBP set
            sh([PY, "experiments/reach_curvature.py", "--ckpt", str(current),
                "--round", f"R{r}", "--device", args.device], log)
            log.write(f"\n-- round {r} done: ckpt={round_ckpt} --\n"); log.flush()

        log.write("\n===== loop complete =====\n"); log.flush()


if __name__ == "__main__":
    main()
