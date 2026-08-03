#!/usr/bin/env python
"""catspace/research/tools/probes/probe_field.py -- probe a PROBABILITY FIELD (reach fields, hazard-derived
R_g, P_fall...) under the forecast-verification standards (Gneiting & Raftery
2007): "maximize sharpness subject to calibration", scored by PROPER rules.

Input: npz with a prediction array and an outcome array of the same shape --
vectors (N,) or fields (N, G). Reports an ensemble of proper scores (no single
score is 'best'; they discriminate differently):
  log score  : -[y log p + (1-y) log(1-p)]  vs the base-rate forecast (skill = lift)
  Brier      : mean squared error of the probability, + its decomposition intuition
  sharpness  : entropy of the field rows (mean bits + histogram) -- a field can
               only claim sharpness AFTER calibration passes (probe_calibration)
Figure (--fig): entropy histogram + per-row log-score-lift histogram.

Usage: catspace/research/tools/probes/probe_field.py preds.npz --p p_reach --y hit [--fig field.png]
"""
from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--p", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--fig", default="")
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    p = np.clip(d[args.p].astype(np.float64), 1e-9, 1 - 1e-9)
    y = d[args.y].astype(np.float64)
    assert p.shape == y.shape
    if p.ndim == 1:
        p = p[:, None]; y = y[:, None]
    base = np.clip(y.mean(0, keepdims=True), 1e-9, 1 - 1e-9)   # per-column base rate
    ls = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    ls_base = -(y * np.log(base) + (1 - y) * np.log(1 - base))
    lift = (ls_base - ls).mean(1)                              # nats/row, >0 = skill
    brier = float(((p - y) ** 2).mean())
    brier_base = float(((base - y) ** 2).mean())
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p)).mean(1)  # bits/cell
    print(f"VERDICT field[{args.p} vs {args.y}]: log-score {ls.mean():.5f} vs "
          f"base-rate {ls_base.mean():.5f} | skill {lift.mean():+.5f} nats/row "
          f"({'PASS' if lift.mean() > 0 else 'no skill'}) | rows>{0} skill: "
          f"{np.mean(lift > 0):.1%}")
    print(f"VERDICT field Brier: {brier:.5f} vs base {brier_base:.5f} "
          f"(skill {brier_base - brier:+.5f})")
    print(f"VERDICT field sharpness: mean entropy {ent.mean():.3f} bits/cell "
          f"(p5 {np.percentile(ent, 5):.3f} p95 {np.percentile(ent, 95):.3f}) | "
          f"sharpness claims require calibration to pass first (probe_calibration)")
    if args.fig:
        import sys
        from pathlib import Path
        from catspace.research.tools.figures import figlib
        fig, ax = figlib.new_fig(2)
        ax[0].hist(ent, bins=40, color=figlib.ACCENT, edgecolor="none")
        ax[0].set_xlabel("entropy (bits/cell)"); ax[0].set_title("Sharpness")
        ax[1].hist(lift, bins=40, color=figlib.ACCENT, edgecolor="none")
        ax[1].axvline(0, color=figlib.MUTED, lw=1)
        ax[1].set_xlabel("log-score lift vs base (nats/row)")
        ax[1].set_title("Skill")
        figlib.save(fig, args.fig, f"Probability field — {args.p}")


if __name__ == "__main__":
    main()
