"""
cone/neural.py — neural Forward-Backward embedding for the TOY domains:
F-net/B-net MLPs trained by InfoNCE on geometric-horizon future pairs, the
contrastive estimate of the discounted successor measure (the cone).
Registered as the "fb_neural" quasimetric-construction method.

Internals are PyTorch (import name `torch`; lazy-imported so the numpy-only
core stays importable without it -- construct a NeuralFB and you need
`pip install -e .[nn]`). The original hand-rolled numpy MLP with manual
backprop/Adam lives in git history; project rule: import the framework,
don't reimplement it. The full-board stack is catspace/nn/ (TorchFB).

F-net/B-net: one-hot board encoding -> dh -> dh -> d.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from catspace.cone.embedding import GoalSpec, register_embedding


def one_hot_state(values, nsq: int = 25) -> np.ndarray:
    """One-hot encode a fixed-length tuple of board-square values, e.g. (wk, wr, bk)."""
    v = np.zeros(3 * nsq + 2, dtype=np.float32)
    for i, val in enumerate(values[:3]):
        v[i * nsq + val] = 1
    return v


def absorbing_vec(kind: int, nsq: int = 25) -> np.ndarray:  # kind: 0=MATE, 1=DRAW
    v = np.zeros(3 * nsq + 2, dtype=np.float32)
    v[3 * nsq + kind] = 1
    return v


class NeuralFB:
    """Same public API as the pre-PyTorch version: train_step(Xs, Xg, lr) ->
    float loss (InfoNCE, in-batch negatives, logits = F @ B.T / tau);
    embed_F/embed_B take and return numpy."""

    def __init__(self, d: int = 32, dh: int = 256, seed: int = 0, tau: float = 0.1, din: int = 77):
        import torch
        from torch import nn
        self._torch = torch
        torch.manual_seed(seed)          # sequential construction: F then B draw distinct inits
        def mlp():
            return nn.Sequential(nn.Linear(din, dh), nn.ReLU(),
                                 nn.Linear(dh, dh), nn.ReLU(),
                                 nn.Linear(dh, d))
        self.F = mlp()
        self.B = mlp()
        self.opt = torch.optim.Adam([*self.F.parameters(), *self.B.parameters()])
        self.tau = tau
        self.d = d

    def train_step(self, Xs: np.ndarray, Xg: np.ndarray, lr: float) -> float:
        """InfoNCE with in-batch negatives: rows = anchors F(s), cols = B(g)."""
        torch = self._torch
        for group in self.opt.param_groups:
            group["lr"] = lr
        f = self.F(torch.from_numpy(np.asarray(Xs, dtype=np.float32)))
        b = self.B(torch.from_numpy(np.asarray(Xg, dtype=np.float32)))
        logits = (f @ b.T) / self.tau
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(logits)))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss)

    def _apply(self, net, X: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            return net(torch.from_numpy(np.asarray(X, dtype=np.float32))).numpy()

    def embed_F(self, X: np.ndarray) -> np.ndarray:
        return self._apply(self.F, X)

    def embed_B(self, X: np.ndarray) -> np.ndarray:
        return self._apply(self.B, X)


@register_embedding("fb_neural")
@dataclass
class EncodedNeuralFB:
    """Adapts a trained NeuralFB + a fixed per-state encoding into the
    QuasimetricEmbedding protocol by precomputing F/B once over all states
    (mirrors exp_generalization.py's `Fn = net.embed_F(X_all)` pattern)."""
    net: NeuralFB
    F: np.ndarray
    B: np.ndarray

    @classmethod
    def from_encoded(cls, net: NeuralFB, X_all: np.ndarray) -> "EncodedNeuralFB":
        return cls(net=net, F=net.embed_F(X_all), B=net.embed_B(X_all))

    @property
    def d(self) -> int:
        return self.net.d

    def F_of(self, idx=None):
        return self.F if idx is None else self.F[idx]

    def B_of(self, idx=None):
        return self.B if idx is None else self.B[idx]

    def reach(self, idx, goal: GoalSpec) -> np.ndarray:
        F = self.F if idx is None else self.F[idx]
        return F @ goal.z
