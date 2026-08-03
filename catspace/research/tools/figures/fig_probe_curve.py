#!/usr/bin/env python
"""catspace/research/tools/figures/fig_probe_curve.py -- THE headline evaluation figure of the JEPA
literature (I-JEPA/V-JEPA/LeJEPA): frozen-feature probe performance vs
pretraining step. Takes several representation files (one per checkpoint, same
rows) tagged with their step, runs linear + kNN probes and RankMe on each, and
plots all three curves. LeJEPA-style bonus: if --loss values are supplied, a
loss-vs-probe scatter (label-free model selection: does the training loss rank
checkpoints the way the probe does?).

Usage:
  catspace/research/tools/figures/fig_probe_curve.py --reps 5000:s5k.npz 10000:s10k.npz 20000:s20k.npz \
      --label region --group gid --fig probe_curve.png [--loss 5000:1.2 ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from catspace.research.tools.figures import figlib                                                # noqa: E402
from catspace.research.tools.probes.probe_rank import rankme                                     # noqa: E402


def probe_one(path, label, group, rng, sample=20000, knn=20):
    d = np.load(path, allow_pickle=True)
    X = d["emb"].astype(np.float64); y = d[label]
    if len(X) > sample:
        idx = np.sort(rng.choice(len(X), sample, replace=False))
        X, y = X[idx], y[idx]
        g = d[group][idx] if group else None
    else:
        g = d[group][:len(X)] if group else None
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    if g is not None:
        gs = np.unique(g)
        te_g = set(rng.choice(gs, max(1, int(len(gs) * 0.2)), replace=False).tolist())
        te = np.array([x in te_g for x in g])
    else:
        te = rng.random(len(X)) < 0.2
    tr = ~te
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    lin = float(clf.score(X[te], y[te]))
    sim = X[te] @ X[tr].T
    nn = np.argsort(-sim, 1)[:, :knn]
    kacc = float(np.mean([np.bincount(v.astype(int)).argmax() == yy
                          for v, yy in zip(y[tr][nn], y[te])]))
    return lin, kacc, rankme(d["emb"][:sample].astype(np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", required=True, help="step:path pairs")
    ap.add_argument("--label", required=True)
    ap.add_argument("--group", default="")
    ap.add_argument("--loss", nargs="*", default=[], help="step:value pairs")
    ap.add_argument("--fig", default="probe_curve.png")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    steps, lins, knns, ranks = [], [], [], []
    for pair in args.reps:
        step, path = pair.split(":", 1)
        lin, kacc, rm = probe_one(path, args.label, args.group, rng)
        steps.append(int(step)); lins.append(lin); knns.append(kacc); ranks.append(rm)
        print(f"VERDICT probe-curve step {step}: linear {lin:.3f} | kNN {kacc:.3f} "
              f"| RankMe {rm:.1f}")
    order = np.argsort(steps)
    steps = np.array(steps)[order]
    lins = np.array(lins)[order]; knns = np.array(knns)[order]
    ranks = np.array(ranks)[order]
    loss = {int(s): float(v) for s, v in (p.split(":") for p in args.loss)}
    ncols = 3 if loss else 2
    fig, ax = figlib.new_fig(ncols)
    ax[0].plot(steps, lins, color=figlib.CAT[0], marker="o", ms=4, label="linear")
    ax[0].plot(steps, knns, color=figlib.CAT[1], marker="s", ms=4, label="kNN")
    ax[0].set_xlabel("pretraining step"); ax[0].set_ylabel(f"probe acc [{args.label}]")
    ax[0].legend(frameon=False); ax[0].set_title("frozen-probe evaluation")
    ax[1].plot(steps, ranks, color=figlib.CAT[2], marker="o", ms=4)
    ax[1].set_xlabel("pretraining step"); ax[1].set_title("RankMe (label-free)")
    if loss:
        lv = [loss.get(int(s), np.nan) for s in steps]
        ax[2].scatter(lv, lins, color=figlib.CAT[0], s=25)
        for s, x0, y0 in zip(steps, lv, lins):
            ax[2].annotate(str(s), (x0, y0), textcoords="offset points",
                           xytext=(4, 3), fontsize=6, color=figlib.MUTED)
        ax[2].set_xlabel("training loss"); ax[2].set_ylabel("linear probe acc")
        ax[2].set_title("loss vs probe (model selection)")
    figlib.save(fig, args.fig, "Probe performance vs pretraining")


if __name__ == "__main__":
    main()
