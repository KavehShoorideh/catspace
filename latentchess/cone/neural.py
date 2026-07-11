"""
cone/neural.py — neural Forward-Backward embedding (numpy MLP, manual
backprop/Adam) trained by InfoNCE on geometric-horizon future pairs, the
contrastive estimate of the discounted successor measure (the cone).
Registered as the "fb_neural" quasimetric-construction method.

F-net/B-net: one-hot board encoding -> 256 -> 256 -> d.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from latentchess.cone.embedding import GoalSpec, register_embedding


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


class MLP:
    def __init__(self, din: int, dh: int, dout: int, seed: int):
        r = np.random.default_rng(seed)
        s1, s2, s3 = (2 / din) ** .5, (2 / dh) ** .5, (2 / dh) ** .5
        self.p = dict(
            W1=r.standard_normal((din, dh)).astype(np.float32) * s1, b1=np.zeros(dh, np.float32),
            W2=r.standard_normal((dh, dh)).astype(np.float32) * s2, b2=np.zeros(dh, np.float32),
            W3=r.standard_normal((dh, dout)).astype(np.float32) * s3, b3=np.zeros(dout, np.float32))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, X):
        p = self.p
        z1 = X @ p['W1'] + p['b1']; a1 = np.maximum(z1, 0)
        z2 = a1 @ p['W2'] + p['b2']; a2 = np.maximum(z2, 0)
        out = a2 @ p['W3'] + p['b3']
        self.cache = (X, z1, a1, z2, a2)
        return out

    def backward(self, dout):
        X, z1, a1, z2, a2 = self.cache
        p, g = self.p, {}
        g['W3'] = a2.T @ dout; g['b3'] = dout.sum(0)
        da2 = dout @ p['W3'].T; dz2 = da2 * (z2 > 0)
        g['W2'] = a1.T @ dz2; g['b2'] = dz2.sum(0)
        da1 = dz2 @ p['W2'].T; dz1 = da1 * (z1 > 0)
        g['W1'] = X.T @ dz1; g['b1'] = dz1.sum(0)
        return g

    def adam(self, g, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)


class NeuralFB:
    def __init__(self, d: int = 32, dh: int = 256, seed: int = 0, tau: float = 0.1, din: int = 77):
        self.F = MLP(din, dh, d, seed)
        self.B = MLP(din, dh, d, seed + 1)
        self.tau = tau
        self.d = d

    def train_step(self, Xs, Xg, lr):
        """InfoNCE with in-batch negatives: rows = anchors F(s), cols = B(g)."""
        f = self.F.forward(Xs)                       # (n, d)
        b = self.B.forward(Xg)                       # (n, d)
        logits = (f @ b.T) / self.tau                # (n, n)
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits); probs = e / e.sum(1, keepdims=True)
        n = len(Xs)
        loss = -np.log(probs[np.arange(n), np.arange(n)] + 1e-12).mean()
        dlog = (probs - np.eye(n, dtype=np.float32)) / n / self.tau
        df = dlog @ b                                # (n, d)
        db = dlog.T @ f
        self.F.adam(self.F.backward(df), lr)
        self.B.adam(self.B.backward(db), lr)
        return loss

    def embed_F(self, X): return self.F.forward(X)
    def embed_B(self, X): return self.B.forward(X)


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
