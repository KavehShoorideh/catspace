#!/usr/bin/env python
"""
experiments/eval_variant.py — one-command evaluation of an embedding variant for
the overnight "hop-search that plays well" search. Runs the two metrics that
matter for the north star (play + hop gradient) and appends a compact record to
artifacts/experiments/overnight_results.jsonl:

  conversion  : paired KRRvKBP conversion vs the incumbent (PRIMARY -- does it
                actually play better with 200-node hop search?). Uses whatever
                goal the checkpoint's zgoals["MATE_W"] is (pole or centroid).
  curvature   : move-ranking-vs-DTZ rho, top1_win, move_spread on the fixed set
                (does the hop field rank the winning move well / preserve top-1?).

Prints one VERDICT line. Run: python experiments/eval_variant.py --ckpt <c> --label <name>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
INCUMBENT = "data/derived/lichess_fb_4gb_qm_plygap_only.pt"


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.stdout + p.stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_test_n200.json",
                    help="held-out A/B test set (bigger = tighter CI; disjoint from training)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="artifacts/experiments/overnight_results.jsonl")
    args = ap.parse_args()

    rec = {"label": args.label, "ckpt": args.ckpt, "note": args.note}

    # curvature (fast)
    cout = run([PY, "experiments/reach_curvature.py", "--ckpt", args.ckpt,
                "--round", args.label, "--device", args.device])
    for key, pat in [("move_spread", r"move_spread.*?:\s*([\-\d.]+)"),
                     ("dtz_rho", r"dtz_rho.*?:\s*([+\-\d.]+)"),
                     ("best_rank", r"best_rank.*?:\s*([\-\d.]+)"),
                     ("top1_win", r"top1_win.*?:\s*([\-\d.]+)")]:
        m = re.search(pat, cout)
        rec[key] = float(m.group(1)) if m else None

    # conversion vs incumbent -- the PAIRED A/B harness (matched-seed diff + CI +
    # anytime-valid e-value). Capture ALL of it, not just the point estimates.
    vout = run([PY, "experiments/conversion_compare.py", "--ckpt-a", INCUMBENT,
                "--ckpt-b", args.ckpt, "--fixed-set", args.fixed_set, "--opponent", "sf:skill=0",
                "--nodes", str(args.nodes), "--device", args.device])
    m = re.search(r"conversion A=([\d.]+) vs B=([\d.]+)\s+\(n=(\d+),\s*"
                  r"mean_diff=([+\-\d.]+)\s+CI=\[([+\-\d.]+),([+\-\d.]+)\],\s*e=([\d.eE+\-]+)\)", vout)
    if m:
        rec["conv_incumbent"] = float(m.group(1))
        rec["conv_variant"] = float(m.group(2))
        rec["n_test"] = int(m.group(3))
        rec["conv_diff"] = float(m.group(4))
        rec["ci_lo"] = float(m.group(5))
        rec["ci_hi"] = float(m.group(6))
        rec["e_value"] = float(m.group(7))
        # anytime-valid decision: e-value > 20 (~alpha 0.05) AND CI excludes 0
        rec["significant"] = bool(rec["e_value"] > 20 and (rec["ci_lo"] > 0 or rec["ci_hi"] < 0))
    rec["fixed_set"] = args.fixed_set

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    sig = "SIGNIFICANT" if rec.get("significant") else "ns"
    print(f"VERDICT [{args.label}] conv {rec.get('conv_variant','?')} vs incumbent "
          f"{rec.get('conv_incumbent','?')} | diff {rec.get('conv_diff','?')} "
          f"CI=[{rec.get('ci_lo','?')},{rec.get('ci_hi','?')}] e={rec.get('e_value','?')} [{sig}] "
          f"(n={rec.get('n_test','?')}) | top1_win {rec.get('top1_win','?')} dtz_rho {rec.get('dtz_rho','?')}")
    print(f"-> appended {out}")


if __name__ == "__main__":
    main()
