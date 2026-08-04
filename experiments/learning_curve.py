#!/usr/bin/env python
"""experiments/learning_curve.py -- Kaveh 2026-08-03: train/holdout error as a function of
TRAINING-SET SIZE (a learning curve), not as a function of steps (a loss curve).

The question it answers, and why it is worth the compute: we cut the training data 4x (2.33M rows
-> a 600k contiguous subset) purely to fit the trunk features in 36GB of RAM. That is a
performance decision made for I/O reasons, and it should not be allowed to quietly cost accuracy.
A learning curve settles it:

  * the two arms converge, both at a high error  -> UNDERFIT / bias-limited. More data will not
    help; capacity or the objective is the binding constraint.
  * train stays low, holdout sits well above it  -> VARIANCE-limited. More data WILL help, and
    the 4x cut is costing us something measurable.

Protocol:
  * the holdout is FIXED and identical at every size -- the val split is drawn first from a shared
    seed, before --train-frac touches anything.
  * splits are BY GAME, never by row. Positions inside one game are heavily correlated, so a
    row-level split would leave near-duplicates of held-out positions in training and every point
    on the curve would flatter itself.
  * every size trains for the SAME number of steps, chosen past the point the full-size run
    plateaued, so each point is at convergence rather than at a fixed compute budget. Smaller
    sizes therefore see more epochs, which is the standard and intended setup.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import sqlite3
import time
from pathlib import Path

import numpy as np


def last_metrics(db, run_name):
    c = sqlite3.connect(db)
    r = c.execute("select run_uuid from runs where name=? order by start_time desc limit 1",
                  (run_name,)).fetchone()
    if not r:
        return {}
    out = {}
    for (k,) in c.execute("select distinct key from metrics where run_uuid=?", (r[0],)):
        v = c.execute("select value from metrics where run_uuid=? and key=? order by step desc "
                      "limit 1", (r[0], k)).fetchone()
        if v:
            out[k] = v[0]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combined", default="data/derived/field_combined_sub600k.npz")
    ap.add_argument("--fracs", default="0.06,0.12,0.25,0.5,1.0")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--python", default="/Users/kav/code/remote/github/catspace/.venv/bin/python")
    ap.add_argument("--out-prefix", default="artifacts/experiments/basin_learning_curve_v1_noisy")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    fracs = [float(f) for f in args.fracs.split(",")]
    z = np.load(args.combined, allow_pickle=True)
    n_games = len(np.unique(z["game"]))
    rows = []
    for f in fracs:
        name = f"lc_frac{int(round(f*1000)):04d}"
        out = f"artifacts/experiments/lc/{name}"
        Path("artifacts/experiments/lc").mkdir(parents=True, exist_ok=True)
        cmd = [args.python, "-u", "experiments/train_iqe_head.py",
               "--combined", args.combined, "--source", "both", "--seed", str(args.seed),
               "--steps", str(args.steps), "--batch", str(args.batch),
               "--eval-every", "500", "--ckpt-every", str(args.steps),
               "--timing-every", "0", "--train-frac", str(f), "--out", out]
        print(f"\n=== frac {f:.2f}  (~{int(n_games*0.9*f):,} train games) ===", flush=True)
        print("  " + " ".join(cmd[2:]), flush=True)
        if args.dry_run:
            continue
        log = Path(f"{out}.log")
        with open(log, "w") as fh:
            subprocess.run(["caffeinate", "-dimsu"] + cmd, stdout=fh, stderr=subprocess.STDOUT,
                           check=False)
        m = last_metrics("mlflow.db", Path(out).name)
        m["frac"] = f
        m["train_games"] = int(n_games * 0.9 * f)
        rows.append(m)
        print(f"  train CE {m.get('tr_basin_ce', float('nan')):.4f} | "
              f"holdout CE {m.get('va_basin_ce', float('nan')):.4f} | "
              f"gap {m.get('va_basin_ce', np.nan) - m.get('tr_basin_ce', np.nan):+.4f} | "
              f"holdout acc {m.get('va_basin_acc', float('nan')):.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    if args.dry_run or not rows:
        return

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(f"{args.out_prefix}.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = [r["train_games"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (a, b, lab) in zip(axes, [("tr_basin_ce", "va_basin_ce", "basin cross-entropy"),
                                      ("tr_basin_acc", "va_basin_acc", "basin accuracy")]):
        ax.plot(n, [r.get(a, np.nan) for r in rows], "o-", color="#2a78d6", lw=2, label="train")
        ax.plot(n, [r.get(b, np.nan) for r in rows], "s-", color="#e34948", lw=2, label="holdout")
        ax.set_xscale("log"); ax.set_xlabel("training games (log)"); ax.set_ylabel(lab)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title(lab, color="#1c1b19")
    fig.suptitle("Learning curve: does the 4x data cut cost anything?\n"
                 "curves meeting high = bias-limited (more data will not help); "
                 "persistent gap = variance-limited (it will)")
    fig.tight_layout(); fig.savefig(f"{args.out_prefix}.png", dpi=140)

    print("\nLEARNING CURVE")
    print(f"  {'games':>9s} {'train CE':>9s} {'hold CE':>9s} {'gap':>8s} "
          f"{'train acc':>10s} {'hold acc':>9s} {'eff_rank':>9s}")
    for r in rows:
        print(f"  {r['train_games']:>9,} {r.get('tr_basin_ce', np.nan):>9.4f} "
              f"{r.get('va_basin_ce', np.nan):>9.4f} "
              f"{r.get('va_basin_ce', np.nan) - r.get('tr_basin_ce', np.nan):>+8.4f} "
              f"{r.get('tr_basin_acc', np.nan):>10.3f} {r.get('va_basin_acc', np.nan):>9.3f} "
              f"{r.get('eff_rank', np.nan):>9.1f}")
    print(f"wrote {args.out_prefix}.png / .json [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
