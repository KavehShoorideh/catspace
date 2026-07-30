#!/usr/bin/env python
"""tools/probe_rank.py -- label-free spectral quality of a representation file.

Reports (per input file, so several can be compared side by side):
  RankMe        : exp(entropy of normalized singular values) -- Garrido et al. 2023;
                  tracks downstream linear-probe accuracy WITHOUT labels
  eff_rank      : participation-ratio effective rank (the repo's collapse gate)
  variance curve: PCs needed for 50/90/99% variance
  spectrum slope: log-log linear fit of the eigenvalue decay (a cliff ~ collapse;
                  healthy JE spectra decay smoothly -- Jing et al. 2022)
Optionally --lda-labels <col>: LiDAR-flavored check -- effective rank of the LDA
scatter matrix under a label column (discounts variance that carries no signal).

Usage: tools/probe_rank.py rep1.npz [rep2.npz ...] [--lda-labels class]
"""
from __future__ import annotations

import argparse

import numpy as np


def rankme(emb):
    s = np.linalg.svd(emb - emb.mean(0), compute_uv=False)
    p = s / s.sum() + 1e-12
    return float(np.exp(-(p * np.log(p)).sum()))


def eff_rank_pr(emb):
    ev = np.linalg.eigvalsh(np.cov(emb.T))
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reps", nargs="+")
    ap.add_argument("--lda-labels", default="")
    ap.add_argument("--sample", type=int, default=20000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for path in args.reps:
        d = np.load(path, allow_pickle=True)
        emb = d["emb"].astype(np.float64)
        if len(emb) > args.sample:
            emb_s = emb[np.sort(rng.choice(len(emb), args.sample, replace=False))]
        else:
            emb_s = emb
        N, D = emb_s.shape
        ev = np.sort(np.clip(np.linalg.eigvalsh(np.cov(emb_s.T)), 1e-12, None))[::-1]
        var = np.cumsum(ev) / ev.sum()
        pcs = [int(np.searchsorted(var, q) + 1) for q in (0.5, 0.9, 0.99)]
        k = max(D // 4, 8)
        slope = np.polyfit(np.log(np.arange(1, k + 1)), np.log(ev[:k]), 1)[0]
        line = (f"{path}: N={N} d={D} | RankMe {rankme(emb_s):.1f} | eff_rank "
                f"{eff_rank_pr(emb_s):.1f} | PCs for 50/90/99% var: "
                f"{pcs[0]}/{pcs[1]}/{pcs[2]} | spectrum slope {slope:.2f}")
        if args.lda_labels and args.lda_labels in d.files:
            y = d[args.lda_labels][:len(emb)]
            if len(emb) > args.sample:
                idx = np.sort(rng.choice(len(emb), args.sample, replace=False))
                Xl, yl = emb[idx], y[idx]
            else:
                Xl, yl = emb, y
            mu = Xl.mean(0)
            Sb = np.zeros((D, D))
            for c in np.unique(yl):
                m = Xl[yl == c].mean(0) - mu
                Sb += (yl == c).mean() * np.outer(m, m)
            evb = np.clip(np.linalg.eigvalsh(Sb), 0, None)
            lidar = float(evb.sum() ** 2 / ((evb ** 2).sum() + 1e-12))
            line += f" | LDA-rank[{args.lda_labels}] {lidar:.1f}"
        print("VERDICT rank: " + line)


if __name__ == "__main__":
    main()
