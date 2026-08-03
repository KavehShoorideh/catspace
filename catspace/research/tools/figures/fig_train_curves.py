#!/usr/bin/env python
"""catspace/research/tools/figures/fig_train_curves.py -- training curves from a run's metrics JSONL
(written by catspace.research.infra RunLogger): per-term losses, effective rank (the collapse
gate's history), throughput, timer splits. Overlay up to 4 runs.

Usage: catspace/research/tools/figures/fig_train_curves.py artifacts/experiments/jepa_pretrain_metrics.jsonl \
           [more_metrics.jsonl ...] --fig train_curves.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from catspace.research.tools.figures import figlib                                                # noqa: E402


def load(path):
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    keys = sorted({k for r in rows for k in r})
    return {k: np.array([r.get(k, np.nan) for r in rows], float) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", nargs="+")
    ap.add_argument("--fig", default="train_curves.png")
    args = ap.parse_args()
    runs = [(Path(p).stem.replace("_metrics", ""), load(p)) for p in args.metrics[:4]]
    fig, ax = figlib.new_fig(3)
    for i, (name, m) in enumerate(runs):
        col = figlib.CAT[i]
        for term, ls in (("l_dyn", "-"), ("l_haz", "--"), ("l_dest", ":")):
            if term in m and np.isfinite(m[term]).any():
                ax[0].plot(m["step"], m[term], ls, color=col, lw=1.6,
                           label=f"{name}:{term.split('_')[1]}" if len(runs) > 1
                           else term.split("_")[1])
        if "eff_rank" in m:
            ax[1].plot(m["step"], m["eff_rank"], color=col, label=name)
        if "steps_per_s" in m:
            ax[2].plot(m["step"], m["steps_per_s"], color=col, label=name)
    ax[0].set_yscale("log"); ax[0].set_title("loss terms")
    ax[0].set_xlabel("step"); ax[0].legend(frameon=False, fontsize=7)
    ax[1].set_title("effective rank (collapse gate)"); ax[1].set_xlabel("step")
    ax[2].set_title("throughput (steps/s)"); ax[2].set_xlabel("step")
    if len(runs) > 1:
        ax[1].legend(frameon=False, fontsize=7)
    figlib.save(fig, args.fig, "Training curves")
    for name, m in runs:
        last = {k: v[np.isfinite(v)][-1] for k, v in m.items()
                if np.isfinite(v).any() and k != "step"}
        print(f"VERDICT train-curves[{name}]: last step {int(m['step'][-1])} | "
              + " ".join(f"{k} {v:.4g}" for k, v in sorted(last.items())
                         if k in ("l_dyn", "l_haz", "l_dest", "eff_rank",
                                  "clamp_acc", "steps_per_s")))


if __name__ == "__main__":
    main()
