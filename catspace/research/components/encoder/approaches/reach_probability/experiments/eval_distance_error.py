#!/usr/bin/env python
"""eval_distance_error.py -- RMS distance error vs ply gap, held-out games (Kaveh 2026-08-07).

For observed pairs (a,b) in TEST games with true gap g plies, the field predicts d(a->b), trained
toward log1p(g) -- so raw d is a plies estimate. This plots RMS error per gap bucket, in BOTH
units (plies, and log-space where the loss lives), with the training cap marked: gaps <= 40 are
directly supervised by springs; everything beyond is COMPOSITION -- the extrapolation your
'hops all the way to the end' plan relies on. n per bucket printed on the bars.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402

from catspace.io import paths                                              # noqa: E402
from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (  # noqa: E402
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (  # noqa: E402
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.experiments.plot_strata_figures import (  # noqa: E402
    embed)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-pair", type=int, default=40000)
    ap.add_argument("--max-gap", type=int, default=200)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    net, pay = load_net(args.ckpt, args.device)
    c = pay["cfg"]
    tr = T.build(n_human=c["games"] // 2, n_sf=c["games"] // 2, seed=c["traj_seed"],
                 max_plies=c["max_plies"], n_piecedown=c.get("n_piecedown", 0), verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    test = np.flatnonzero(split == 2)
    game, ply = tr.game_of_row(), tr.ply_of_row()
    # OPTION-B SCOPE (2026-08-07): the walls only assert SF paths' distances -- scoring human
    # pairs here would report design-intended looseness as error. SF-source games only.
    rows = np.flatnonzero(np.isin(game, test) & (tr.source[game] == 0))
    rng = np.random.default_rng(0)

    i0 = rows[rng.integers(0, len(rows), args.n_pair)]
    g = game[i0]
    end = tr.start[g] + tr.length[g] - 1
    j0 = i0 + 1 + (rng.random(args.n_pair) * np.minimum(args.max_gap, end - i0)).astype(np.int64)
    ok = j0 <= end
    i0, j0 = i0[ok], j0[ok]
    gap = (ply[j0] - ply[i0]).astype(np.float64)

    Za, Zb = embed(net, tr, i0, args.device), embed(net, tr, j0, args.device)
    iqe = net.qhead.iqe if getattr(net, "dual", False) else net.iqe
    with torch.no_grad():
        d = iqe(Za.to(args.device), Zb.to(args.device)).float().cpu().detach().numpy().astype(np.float64)

    err_ply = d - gap
    err_log = np.log1p(np.clip(d, 0, None)) - np.log1p(gap)

    edges = [1, 2, 3, 5, 8, 12, 20, 30, 40, 60, 90, 130, 200]
    labs, rms_p, rms_l, ns = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gap >= lo) & (gap < hi)
        if m.sum() < 50:
            continue
        labs.append(f"{lo}-{hi-1}")
        rms_p.append(float(np.sqrt(np.mean(err_ply[m] ** 2))))
        rms_l.append(float(np.sqrt(np.mean(err_log[m] ** 2))))
        ns.append(int(m.sum()))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    xs = np.arange(len(labs))
    cut = sum(1 for l in labs if int(l.split("-")[1]) <= 40)
    for a, vals, ttl in ((ax[0], rms_p, "RMS error (plies)"),
                         (ax[1], rms_l, "RMS error (log1p space -- where the loss lives)")):
        cols = ["#2e5f9e"] * cut + ["#c0392b"] * (len(labs) - cut)
        a.bar(xs, vals, color=cols)
        a.set_xticks(xs, labs, rotation=45, fontsize=8)
        a.set_xlabel("true ply gap"); a.set_title(ttl)
        a.axvline(cut - 0.5, color="k", ls="--", lw=1)
        for x, v, n in zip(xs, vals, ns):
            a.text(x, v, f"n={n}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax[0].text(cut - 0.4, max(rms_p) * 0.95, "training cap (40) ->\ncomposition only",
               fontsize=8, color="#c0392b")
    step = pay.get("step")
    fig.suptitle(f"Held-out distance error vs gap -- {args.ckpt.split('/')[-1]} @ step {step}\n"
                 f"blue = spring-supervised gaps, red = extrapolation via composition", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = args.out or paths.figure(f"dist_err_{args.ckpt.split('/')[-1].replace('.pt','')}.png")
    fig.savefig(out, dpi=140)
    # PER-GAP FORENSICS for the small gaps (Kaveh: "why is ply 1 higher than 2 and 3?") --
    # signed mean separates bias from noise; odd/even at matched magnitude isolates the
    # mover-parity hypothesis (gap 1 flips the side to move, gap 2 restores it).
    print("gap  mean_signed(plies)  RMS(plies)  RMS(log)      n   <- per-gap forensics")
    for gg in range(1, 9):
        m = gap == gg
        if m.sum() < 50: continue
        print(f"{gg:>4} {np.mean(err_ply[m]):>17.2f} {np.sqrt(np.mean(err_ply[m]**2)):>11.2f} "
              f"{np.sqrt(np.mean(err_log[m]**2)):>9.3f} {int(m.sum()):>7,}")
    modd = (gap % 2 == 1) & (gap <= 19); mev = (gap % 2 == 0) & (gap <= 19)
    print(f"odd gaps <=19:  RMS(log) {np.sqrt(np.mean(err_log[modd]**2)):.3f}  n={modd.sum():,}")
    print(f"even gaps <=19: RMS(log) {np.sqrt(np.mean(err_log[mev]**2)):.3f}  n={mev.sum():,}")
    print("gap    RMS(plies)  RMS(log)   n")
    for l, p, q, n in zip(labs, rms_p, rms_l, ns):
        print(f"{l:>7} {p:9.1f} {q:9.3f} {n:7,}")
    print(f"[fig] -> {out}")


if __name__ == "__main__":
    main()
