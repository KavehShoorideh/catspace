#!/usr/bin/env python
"""tools/fig_success.py -- success-rate comparison in the robotics-evaluation
convention (NVIDIA SRL / Fox): binomial rates carry 95% CLOPPER-PEARSON exact
intervals; chess SCORES (wins + draws/2) carry game-bootstrap CIs; a breakdown
panel replaces bare binary scores (a score without its failure structure
explains nothing).

Input CSV (header): name,W,D,L  -- one row per system/config.

Prints a verdict per row and renders grouped bars with CI whiskers + a
W/D/L composition panel.

Usage: tools/fig_success.py results.csv --fig success.png [--baseline 0.125]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import figlib                                                # noqa: E402


def clopper_pearson(k, n, alpha=0.05):
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvfile")
    ap.add_argument("--baseline", type=float, default=float("nan"))
    ap.add_argument("--fig", default="success.png")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = list(csv.DictReader(open(args.csvfile)))
    names, scores, lo_s, hi_s, wdl = [], [], [], [], []
    for r in rows:
        W, D, L = int(r["W"]), int(r["D"]), int(r["L"])
        n = W + D + L
        outcomes = np.concatenate([np.ones(W), 0.5 * np.ones(D), np.zeros(L)])
        boots = [rng.choice(outcomes, n).mean() for _ in range(4000)]
        s_lo, s_hi = np.percentile(boots, [2.5, 97.5])
        w_lo, w_hi = clopper_pearson(W, n)
        names.append(r["name"]); scores.append(outcomes.mean())
        lo_s.append(s_lo); hi_s.append(s_hi); wdl.append((W, D, L))
        print(f"VERDICT success[{r['name']}]: score {outcomes.mean():.3f} "
              f"CI95[{s_lo:.3f},{s_hi:.3f}] (game bootstrap) | win-rate "
              f"{W/n:.3f} CP95[{w_lo:.3f},{w_hi:.3f}] | n={n}")
    x = np.arange(len(names))
    fig, ax = figlib.new_fig(2, w=4.6)
    ax[0].bar(x, scores, 0.55, color=figlib.CAT[0])
    ax[0].errorbar(x, scores, yerr=[np.array(scores) - lo_s, np.array(hi_s) - scores],
                   fmt="none", ecolor=figlib.INK, capsize=3, lw=1)
    if not np.isnan(args.baseline):
        ax[0].axhline(args.baseline, color=figlib.MUTED, ls="--", lw=1)
        ax[0].annotate("baseline", (len(names) - 0.4, args.baseline),
                       fontsize=7, color=figlib.MUTED, va="bottom")
    ax[0].set_xticks(x, names, rotation=20, fontsize=8)
    ax[0].set_ylabel("score"); ax[0].set_title("score (95% CI, game bootstrap)")
    bot = np.zeros(len(names))
    for i, part in enumerate(["W", "D", "L"]):
        vals = np.array([w[i] / sum(w) for w in wdl])
        ax[1].bar(x, vals, 0.55, bottom=bot, color=figlib.CAT[i], label=part)
        bot += vals
    ax[1].set_xticks(x, names, rotation=20, fontsize=8)
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("outcome composition")
    figlib.save(fig, args.fig, "System comparison")


if __name__ == "__main__":
    main()
