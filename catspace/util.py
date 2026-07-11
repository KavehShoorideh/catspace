"""util.py — small numpy-only helpers shared across the package."""
from __future__ import annotations

import numpy as np


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(random positive scores higher than random negative),
    tie-aware via average ranks (scipy.stats.rankdata semantics). Undefined
    (NaN) when either class is empty -- e.g. KRk has no drawn states at all,
    so a WIN/DRAW AUC is meaningless there, not a bug."""
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    # average-rank tie handling
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    rp = ranks[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ridge_r2(X: np.ndarray, y: np.ndarray, folds: int = 5, lam: float = 1.0, seed: int = 0) -> float:
    r = np.random.default_rng(seed)
    n = len(y)
    idx = r.permutation(n)
    fold_ids = np.array_split(idx, folds)
    ss_res, ss_tot = 0.0, 0.0
    for f in range(folds):
        te = fold_ids[f]
        tr = np.concatenate([fold_ids[i] for i in range(folds) if i != f])
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        d = Xtr.shape[1]
        A = Xtr.T @ Xtr + lam * np.eye(d)
        w = np.linalg.solve(A, Xtr.T @ (ytr - ytr.mean()))
        pred = Xte @ w + ytr.mean()
        ss_res += ((yte - pred) ** 2).sum()
        ss_tot += ((yte - y[tr].mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot
