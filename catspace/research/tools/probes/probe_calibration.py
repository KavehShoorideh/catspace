#!/usr/bin/env python
"""catspace/research/tools/probes/probe_calibration.py -- reliability diagram + ECE for any predicted
probability vs binary outcome (the plot "the whole planner arithmetic rests on";
run it before trusting any probability the planner multiplies).

Input: an npz with a prediction column and an outcome column (any names).
Reports count-weighted ECE, max |gap|, per-bin table; equal-width or equal-mass
(--quantile) binning -- report both when the mass is skewed.

Usage: catspace/research/tools/probes/probe_calibration.py preds.npz --pred p_reach --y hit [--quantile]
"""
from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--quantile", action="store_true", help="equal-mass bins")
    ap.add_argument("--fig", default="", help="write the reliability diagram")
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    p = d[args.pred].astype(float).ravel(); y = d[args.y].astype(float).ravel()
    assert len(p) == len(y)
    edges = (np.quantile(p, np.linspace(0, 1, args.bins + 1)) if args.quantile
             else np.linspace(0, 1, args.bins + 1))
    edges[-1] += 1e-9
    ece = 0.0; mx = 0.0; rows = []
    for i in range(args.bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() == 0:
            continue
        gap = abs(p[m].mean() - y[m].mean())
        ece += m.mean() * gap; mx = max(mx, gap)
        rows.append((p[m].mean(), y[m].mean(), int(m.sum())))
    print(f"VERDICT calibration[{args.pred} vs {args.y}]: ECE {ece:.4f} "
          f"(count-weighted) | max|gap| {mx:.3f} | n {len(p):,} | "
          f"{'equal-mass' if args.quantile else 'equal-width'} {args.bins} bins")
    for pm, ym, n in rows:
        print(f"    pred {pm:.3f} realized {ym:.3f} n {n:,}")
    if args.fig and rows:
        import sys
        from pathlib import Path
        from catspace.research.tools.figures import figlib
        fig, ax = figlib.new_fig(1, w=4.2, h=4.0)
        pm = [r[0] for r in rows]; ym = [r[1] for r in rows]; nn = [r[2] for r in rows]
        ax.plot([0, 1], [0, 1], color=figlib.MUTED, lw=1, ls="--")
        ax.plot(pm, ym, color=figlib.ACCENT, marker="o", ms=5)
        for x0, y0, n in zip(pm, ym, nn):
            ax.annotate(f"{n:,}", (x0, y0), textcoords="offset points",
                        xytext=(4, -9), fontsize=6, color=figlib.MUTED)
        ax.set_xlabel("predicted probability"); ax.set_ylabel("realized frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        figlib.save(fig, args.fig,
                    f"Reliability — {args.pred} (ECE {ece:.4f}, max|gap| {mx:.3f})")


if __name__ == "__main__":
    main()
