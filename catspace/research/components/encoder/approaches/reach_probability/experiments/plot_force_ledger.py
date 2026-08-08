#!/usr/bin/env python
"""plot_force_ledger.py -- the measured force audit over training, from a run's steps.jsonl.

Per audited step: per-row gradient magnitude each force group exerts on phi (the shared trunk
output). Log scale, because the story spans three orders of magnitude.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402

from catspace.io import paths                                              # noqa: E402

COLORS = {"F_walls": "#1a4f7a", "F_gas": "#c8913a", "F_basin": "#c0392b",
          "F_vicreg": "#7a5ea8", "F_regionA": "#2e8b57"}
LABELS = {"F_walls": "walls (floor+ceiling)", "F_gas": "screened gas",
          "F_basin": "basin CE (committor)", "F_vicreg": "VICReg", "F_regionA": "arm A (region)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", required=True, help="path to <run>_steps.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.steps)]
    au = [(i + 1, r) for i, r in enumerate(rows) if "F_walls" in r]
    xs = [s for s, _ in au]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for k in COLORS:
        ax.plot(xs, [max(r.get(k, 0), 1e-5) for _, r in au], "o-", color=COLORS[k],
                label=LABELS[k], lw=1.6, ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("mean per-row gradient magnitude on phi  (measured, not derived)")
    name = args.steps.split("/")[-1].replace("_steps.jsonl", "")
    ax.set_title(f"Force ledger during training -- {name}\n"
                 "walls: loud only while violated · arm A: the sustained shaper · "
                 "basin CE: never in the fight")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both")
    out = args.out or paths.figure(f"force_ledger_{name}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[fig] -> {out}")


if __name__ == "__main__":
    main()
