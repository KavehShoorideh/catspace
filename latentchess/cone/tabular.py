"""
cone/tabular.py — tabular Forward-Backward embedding via randomized SVD of the
discounted successor measure ("the cone"):

    M = (1-g) * sum_t g^t P^t  ~=  F @ B.T   (rank-d factorization)

F = U*S ("cone shape per state"), B = V ("goal embedding per state"). This is
the tabular analogue of Forward-Backward representations (Touati & Ollivier
2021), registered as the "fb_svd" quasimetric-construction method.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from latentchess.cone.embedding import GoalSpec, register_embedding


def sm_matvec(P, X, gamma: float, T: int | None = None) -> np.ndarray:
    """(1-g) * sum_{t=0..T} g^t P^t applied to columns of X."""
    if T is None:
        T = int(np.ceil(np.log(1e-6) / np.log(gamma)))
    acc = X.copy().astype(np.float64)
    cur = X.astype(np.float64)
    for _ in range(T):
        cur = gamma * (P @ cur)
        acc += cur
        if np.abs(cur).max() < 1e-9:
            break
    return (1.0 - gamma) * acc


def randomized_svd_sm(P, gamma: float, d: int, n_oversample: int = 10, seed: int = 0):
    """Rank-d SVD of the successor measure M without forming it."""
    r = np.random.default_rng(seed)
    n = P.shape[0]
    k = d + n_oversample
    Omega = r.standard_normal((n, k))
    Y = sm_matvec(P, Omega, gamma)                 # M @ Omega
    Q, _ = np.linalg.qr(Y)
    Z = sm_matvec(P.T.tocsr(), Q, gamma)           # M.T @ Q
    Bsmall = Z.T                                    # Q.T @ M
    Ub, S, Vt = np.linalg.svd(Bsmall, full_matrices=False)
    U = Q @ Ub
    return U[:, :d], S[:d], Vt[:d, :].T            # U, S, V


def fb_from_svd(U, S, V):
    """F = U*S (cone shape per state), B = V (goal embedding per state)."""
    return U * S[None, :], V


def rank_error(P, gamma: float, F, Bm, n_probe: int = 20, seed: int = 3) -> float:
    """Relative error ||M - F B^T||_F / ||M||_F via Hutchinson probes."""
    r = np.random.default_rng(seed)
    n = P.shape[0]
    X = r.standard_normal((n, n_probe))
    MX = sm_matvec(P, X, gamma)
    RX = MX - F @ (Bm.T @ X)
    return np.linalg.norm(RX) / np.linalg.norm(MX)


@register_embedding("fb_svd")
@dataclass
class TabularFB:
    F: np.ndarray
    B: np.ndarray

    @property
    def d(self) -> int:
        return self.F.shape[1]

    @classmethod
    def fit(cls, P, gamma: float, d: int, n_oversample: int = 10, seed: int = 0) -> "TabularFB":
        U, S, V = randomized_svd_sm(P, gamma, d, n_oversample=n_oversample, seed=seed)
        F, B = fb_from_svd(U, S, V)
        return cls(F=F, B=B)

    def F_of(self, idx=None):
        return self.F if idx is None else self.F[idx]

    def B_of(self, idx=None):
        return self.B if idx is None else self.B[idx]

    def reach(self, idx, goal: GoalSpec) -> np.ndarray:
        F = self.F if idx is None else self.F[idx]
        return F @ goal.z
