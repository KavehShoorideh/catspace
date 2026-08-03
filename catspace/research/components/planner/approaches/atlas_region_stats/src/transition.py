"""T(phi, ctx) -- transition/crossing-risk estimator (Atlas component; extracted
from experiments/train_transition_estimator.py 2026-07-30)."""
from __future__ import annotations

import torch
import torch.nn as nn


class T(nn.Module):
    def __init__(self, d_phi, d_ctx, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_phi + d_ctx, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, phi, ctx):
        return self.net(torch.cat([phi, ctx], 1)).squeeze(-1)
