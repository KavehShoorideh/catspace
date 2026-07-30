#!/usr/bin/env python
"""tools/probe_calibration.py -- reliability diagram + ECE for any predicted
probability vs binary outcome (the plot "the whole planner arithmetic rests on";
run it before trusting any probability the planner multiplies).

Input: an npz with a prediction column and an outcome column (any names).
Reports count-weighted ECE, max |gap|, per-bin table; equal-width or equal-mass
(--quantile) binning -- report both when the mass is skewed.

Usage: tools/probe_calibration.py preds.npz --pred p_reach --y hit [--quantile]
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


if __name__ == "__main__":
    main()
