"""
nn/encoder.py — BoardEncoder: a small from-scratch ResNet over (19,8,8) input
planes. Deliberately behind a plain constructor so a frozen Maia/lc0 trunk
can replace it later without touching TorchFB (the agreed plug-in seam).
"""
from __future__ import annotations

import torch
from torch import nn


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        h = torch.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return torch.relu(x + h)


class BoardEncoder(nn.Module):
    """(N,in_planes,8,8) -> (N,out_dim). Spatial info is kept through a
    1x1-conv + flatten head (AZ-style) rather than global pooling -- chess is
    not translation-invariant."""

    def __init__(self, in_planes: int = 20, channels: int = 64, blocks: int = 6,
                 out_dim: int = 256, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*[_ResBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 64, out_dim))
        self.out_dim = out_dim

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))
