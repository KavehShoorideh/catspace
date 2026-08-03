"""
concepts.py — pluggable concept quantization/embedding (VQ plan tokens).

Consolidates 7 copy-pasted kmeans implementations and exposes the token count
K as an explicit hyperparameter (previously hardcoded 16/32 at each call
site). Other quantizers (FSQ, SAE, spectral clustering) register into
QUANTIZERS under the same protocol at the full-board milestone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


class ConceptQuantizer(Protocol):
    n_tokens: int

    def fit(self, X: np.ndarray) -> "ConceptQuantizer": ...
    def tokens(self, X: np.ndarray) -> np.ndarray: ...


def usage_perplexity(tokens: np.ndarray, n_tokens: int) -> float:
    """exp(entropy) of the token-usage distribution -- the roadmap's collapse
    metric (perplexity == n_tokens means every code is used equally often)."""
    usage = np.bincount(tokens, minlength=n_tokens) / len(tokens)
    nz = usage[usage > 0]
    return float(np.exp(-(nz * np.log(nz)).sum()))


@dataclass
class KMeansVQ:
    n_tokens: int
    iters: int = 50
    seed: int = 5
    centers: np.ndarray | None = field(default=None, repr=False)

    def fit(self, X: np.ndarray) -> "KMeansVQ":
        r = np.random.default_rng(self.seed)
        C = X[r.choice(len(X), self.n_tokens, replace=False)].copy()
        for _ in range(self.iters):
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            a = d2.argmin(1)
            for k in range(self.n_tokens):
                m = a == k
                if m.any():
                    C[k] = X[m].mean(0)
        self.centers = C
        return self

    def tokens(self, X: np.ndarray) -> np.ndarray:
        d2 = ((X[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        return d2.argmin(1)


QUANTIZERS: dict[str, Callable[..., ConceptQuantizer]] = {"kmeans": KMeansVQ}
