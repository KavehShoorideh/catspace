#!/usr/bin/env python
"""catspace/research/tools/probes/probe_cka.py -- Centered Kernel Alignment between two representation files
(row-aligned: same positions through two encoders / two checkpoints / two layers).
Linear CKA by default (Kornblith et al. 2019), --rbf for the kernel variant.
High CKA between a trained encoder and its init (or the frozen trunk) = the
training moved little; low CKA across checkpoints = the geometry is still moving.

Usage: catspace/research/tools/probes/probe_cka.py repA.npz repB.npz [--rbf]
"""
from __future__ import annotations

import argparse

import numpy as np


def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic = np.linalg.norm(Y.T @ X, "fro") ** 2
    return float(hsic / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")))


def rbf_cka(X, Y, frac=0.5):
    def gram(Z):
        d2 = ((Z[:, None] - Z[None]) ** 2).sum(-1)
        sigma2 = np.median(d2) * frac + 1e-12
        K = np.exp(-d2 / (2 * sigma2))
        n = len(K); H = np.eye(n) - 1 / n
        return H @ K @ H
    Kx, Ky = gram(X), gram(Y)
    return float((Kx * Ky).sum() / (np.linalg.norm(Kx, "fro") * np.linalg.norm(Ky, "fro")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep_a"); ap.add_argument("rep_b")
    ap.add_argument("--rbf", action="store_true")
    ap.add_argument("--sample", type=int, default=4000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    A = np.load(args.rep_a)["emb"].astype(np.float64)
    B = np.load(args.rep_b)["emb"].astype(np.float64)
    assert len(A) == len(B), "representation files must be row-aligned"
    if len(A) > args.sample:
        idx = np.sort(rng.choice(len(A), args.sample, replace=False))
        A, B = A[idx], B[idx]
    v = rbf_cka(A, B) if args.rbf else linear_cka(A, B)
    kind = "RBF" if args.rbf else "linear"
    print(f"VERDICT CKA({kind}): {v:.4f} | {args.rep_a} vs {args.rep_b} | n={len(A)}")


if __name__ == "__main__":
    main()
