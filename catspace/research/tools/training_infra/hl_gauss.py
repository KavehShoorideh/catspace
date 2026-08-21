#!/usr/bin/env python
"""hl_gauss.py -- histogram-loss Gaussian value binning ("Stop Regressing", 2403.03950).

A scalar target y in [lo, hi] becomes a soft categorical over K bins: the probability
mass a Gaussian N(y, sigma) puts in each bin. Training is cross-entropy against that
soft target; decoding is the expectation of bin centers. ChessBench's recipe for
action-value distillation, adopted per docs/GKO1_PLAN.md.
"""
from __future__ import annotations

import math

import torch


class HLGauss:
    def __init__(self, k=64, lo=0.0, hi=1.0, sigma_ratio=0.75, pad_sigmas=3.0):
        # support PADDED by ~3 sigma beyond [lo, hi]: without it, boundary targets lose
        # half their Gaussian mass to clipping and decode ~1 percent inward (U5 catch)
        self.k, self.lo, self.hi = k, lo, hi
        sigma0 = sigma_ratio * (hi - lo) / k
        pad = pad_sigmas * sigma0
        self.edges = torch.linspace(lo - pad, hi + pad, k + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.sigma = sigma_ratio * (self.edges[-1] - self.edges[0]) / k

    def to(self, dev):
        self.edges = self.edges.to(dev); self.centers = self.centers.to(dev)
        return self

    def encode(self, y):
        """(...,) scalars -> (..., K) soft categorical via Gaussian CDF differences."""
        y = y.clamp(self.lo, self.hi).unsqueeze(-1)
        z = (self.edges - y) / (self.sigma * math.sqrt(2.0))
        cdf = 0.5 * (1.0 + torch.erf(z))
        p = cdf[..., 1:] - cdf[..., :-1]
        return p / p.sum(-1, keepdim=True).clamp(min=1e-12)

    def decode(self, logits):
        return ((torch.softmax(logits, -1) * self.centers).sum(-1)
                ).clamp(self.lo, self.hi)

    def ce(self, logits, y, reduce=True):
        tgt = self.encode(y)
        lp = torch.log_softmax(logits, -1)
        ce = -(tgt * lp).sum(-1)
        return ce.mean() if reduce else ce
