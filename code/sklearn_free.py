"""sklearn_free.py — minimal ridge regression with k-fold CV R^2 (numpy only)."""
import numpy as np

def ridge_r2(X, y, folds=5, lam=1.0, seed=0):
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
