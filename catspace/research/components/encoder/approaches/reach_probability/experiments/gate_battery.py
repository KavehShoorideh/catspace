#!/usr/bin/env python
"""gate_battery.py -- ONE command, the full verdict table, readouts chosen from the checkpoint's
own config (2026-08-08, after three stale-instrument incidents: log-basin logits, unconditioned
phase gate, wrong-readout routing -- instruments must be versioned with the model, not with
anyone's memory).

    .venv/bin/python -m ...gate_battery --ckpt <ckpt.pt> [--quick]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

MOD = "catspace.research.components.encoder.approaches.reach_probability.experiments"


def run(mod, args, pat):
    p = subprocess.run([sys.executable, "-m", f"{MOD}.{mod}"] + args,
                       capture_output=True, text=True, timeout=3600)
    out = p.stdout + p.stderr
    hits = [ln.strip() for ln in out.splitlines() if re.search(pat, ln)]
    return hits or [f"(no output matching /{pat}/ -- rc={p.returncode})"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--quick", action="store_true", help="skip the slow TB gates")
    args = ap.parse_args()

    import torch
    cfg = torch.load(args.ckpt, map_location="cpu", weights_only=False)["cfg"]
    split = cfg.get("split_head", False)
    dual = cfg.get("dual", False)
    cond = ["--cond-elo", "3500"] if dual else []

    print(f"\n===== GATE BATTERY: {args.ckpt} =====")
    print(f"cfg: split_head={split} dual={dual} sf_only={cfg.get('sf_only')} "
          f"move_head={cfg.get('move_head', False)} n_piecedown={cfg.get('n_piecedown', 0)}\n")

    if split:
        print("--- axioms (per block) ---")
        for ln in run("eval_split_axioms", ["--ckpt", args.ckpt], r"^\["):
            print(" ", ln)

    print("--- boundary routing (PRIMARY; gate >= 0.60) ---")
    for ln in run("eval_boundary_routing", ["--ckpt", args.ckpt, "--n-pos", "1500"] + cond,
                  r"pairwise|top1"):
        print(" ", ln)

    print("--- distance calibration (walls-consistent: gap-1 + violation rate) ---")
    for ln in run("eval_distance_error", ["--ckpt", args.ckpt], r"^\s+[1-5]\s|odd|even"):
        print(" ", ln)

    if not args.quick:
        print("--- TB diagnostics (instruments, not gates) ---")
        for ln in run("eval_dtz_gate", ["--ckpt", args.ckpt] + cond, r"TB-WDL|TB-win |TB-loss"):
            print(" ", ln)
        print("--- in-TB move ranking (instrument) ---")
        for ln in run("eval_move_ranking", ["--ckpt", args.ckpt] + cond, r"top1|tau"):
            print(" ", ln)
    print("\n===== end battery =====")


if __name__ == "__main__":
    main()
