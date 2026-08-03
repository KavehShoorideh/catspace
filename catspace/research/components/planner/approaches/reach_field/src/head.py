"""ReachHead -- z-conditioned first-hit reachability head (ReachModel component;
extracted from catspace/research/components/encoder/approaches/reachability_field/experiments/train_reach_head.py 2026-07-30)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ReachHead(nn.Module):
    """Shared-trunk two-tower head; separate output projections per quantity (STANDARDS 10)."""

    def __init__(self, d_phi=64, d_z=16, d_emb=64, width=128, d_opp=0):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(d_phi + d_z + 4 + d_opp, width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU())
        self.goal = nn.Sequential(nn.Linear(d_phi, width), nn.ReLU(),
                                  nn.Linear(width, width), nn.ReLU())
        self.s_hit = nn.Linear(width, d_emb); self.s_time = nn.Linear(width, d_emb)
        self.g_hit = nn.Linear(width, d_emb); self.g_time = nn.Linear(width, d_emb)
        self.b_hit = nn.Parameter(torch.tensor(0.0)); self.b_time = nn.Parameter(torch.tensor(0.0))
        self.scale = d_emb ** -0.5

    def state_embs(self, phi, z, elos):
        h = self.state(torch.cat([phi, z, elos], -1))
        return self.s_hit(h), self.s_time(h)

    def goal_embs(self, bank):
        g = self.goal(bank)
        return self.g_hit(g), self.g_time(g)

    def forward(self, phi, z, elos, bank):
        sh, st = self.state_embs(phi, z, elos)
        gh, gt = self.goal_embs(bank)
        logit = sh @ gh.T * self.scale + self.b_hit          # (B,G)
        plies_log = st @ gt.T * self.scale + self.b_time     # (B,G)
        return logit, plies_log
