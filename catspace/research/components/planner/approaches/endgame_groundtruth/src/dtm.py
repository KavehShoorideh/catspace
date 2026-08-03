"""DTMNet -- distance-to-mate CNN (Endgame/conversion component; extracted from
catspace/research/components/planner/approaches/endgame_groundtruth/experiments/train_dtm_cnn.py 2026-07-30)."""
from __future__ import annotations

import torch
import torch.nn as nn


class DTMNet(nn.Module):
    def __init__(self, c=20, w=96):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(),
            nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w)) for _ in range(4)])
        self.head = nn.Sequential(nn.Linear(w, w), nn.ReLU(), nn.Linear(w, 1))

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = torch.relu(h + b(h))
        return self.head(h.mean((2, 3)))[:, 0]
