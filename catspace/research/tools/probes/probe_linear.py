#!/usr/bin/env python
"""catspace/research/tools/probes/probe_linear.py -- frozen-feature linear + kNN probes of a representation
file against one of its label columns. The standard SSL evaluation protocol:
encoder frozen, features optionally L2-normalized, LINEAR model (logistic /
ridge) + cosine-kNN, with a GROUP-AWARE split (--group, e.g. gid) so eval rows
never share a game with train rows. Baselines (majority / label-shuffle) are
printed with every score -- a probe number without its chance floor is noise.

Usage: catspace/research/tools/probes/probe_linear.py rep.npz --label wdl [--group gid] [--knn 20]
"""
from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep")
    ap.add_argument("--label", required=True)
    ap.add_argument("--group", default="", help="split by this column (e.g. gid)")
    ap.add_argument("--l2", type=int, default=1)
    ap.add_argument("--knn", type=int, default=20)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--sample", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    d = np.load(args.rep, allow_pickle=True)
    X = d["emb"].astype(np.float64); y = d[args.label]
    if len(X) > args.sample:
        idx = np.sort(rng.choice(len(X), args.sample, replace=False))
        X, y = X[idx], y[idx]
        groups = d[args.group][idx] if args.group else None
    else:
        groups = d[args.group][:len(X)] if args.group else None
    if args.l2:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    if groups is not None:
        gs = np.unique(groups)
        te_g = set(rng.choice(gs, int(len(gs) * args.test_frac), replace=False).tolist())
        te = np.array([g in te_g for g in groups])
    else:
        te = rng.random(len(X)) < args.test_frac
    tr = ~te
    is_cls = len(np.unique(y)) <= 50                    # many-valued ints = regression

    if is_cls:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
        acc = float(clf.score(X[te], y[te]))
        maj = float(np.mean(y[te] == np.bincount(y[tr].astype(int)).argmax()))
        extras = ""
        if len(np.unique(y)) == 2:
            from sklearn.metrics import roc_auc_score
            extras = f" | AUC {roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]):.3f}"
        # cosine kNN
        sim = X[te] @ X[tr].T
        nn = np.argsort(-sim, 1)[:, :args.knn]
        votes = y[tr][nn]
        knn_pred = np.array([np.bincount(v.astype(int)).argmax() for v in votes])
        knn_acc = float(np.mean(knn_pred == y[te]))
        print(f"VERDICT probe[{args.label}]: linear acc {acc:.3f}{extras} | "
              f"kNN@{args.knn} acc {knn_acc:.3f} | majority {maj:.3f} | "
              f"n_tr {tr.sum()} n_te {te.sum()}"
              + (f" | group-split by {args.group}" if args.group else ""))
    else:
        from sklearn.linear_model import Ridge
        from scipy.stats import spearmanr
        reg = Ridge(alpha=1.0).fit(X[tr], y[tr])
        pred = reg.predict(X[te])
        rho = float(spearmanr(pred, y[te]).statistic)
        shuf = float(spearmanr(rng.permutation(pred), y[te]).statistic)
        print(f"VERDICT probe[{args.label}]: ridge spearman {rho:.3f} "
              f"(shuffle {shuf:+.3f}) | n_tr {tr.sum()} n_te {te.sum()}"
              + (f" | group-split by {args.group}" if args.group else ""))


if __name__ == "__main__":
    main()
