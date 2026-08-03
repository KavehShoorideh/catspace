"""PlanNet -- the RL PlanSelector's contextual outcome model (Kaveh: 'the planner RL --
I want them in the loop too').

P(win | probe observation at the plan decision, plan taken). make_planner scores plans by
expected outcome and takes the argmax when a checkpoint exists; the deterministic rules
remain the fallback, and the goal machinery is unchanged -- the RL chooses the PLAN TYPE.

Extracted from catspace/research/components/planner/approaches/subgoal_cascade/experiments/train_planner_rl.py in the 2026-08-03 restructure: the engine
loads this at play time, so the architecture and the observation contract are shipping
code. The trainer imports them from here, which also guarantees the featurization used at
train time and at play time cannot drift apart.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

OBS_KEYS = ["n_win", "d_win", "n_loss", "d_loss", "n_draw", "d_draw", "class_density",
            "seen_in_game", "seen_across_games", "prior_entropy", "prior_top1",
            "clock", "clock_headroom", "rep_max_nearby", "n_pieces"]
PLANS = ["direct", "reset", "tradedown"]


def featurize(snap: dict) -> np.ndarray:
    v = []
    for k in OBS_KEYS:
        x = snap.get(k)
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            x = -1.0
        v.append(float(x))
    return np.array(v, np.float32)


class PlanNet(nn.Module):
    def __init__(self, d_obs=len(OBS_KEYS), n_plans=len(PLANS), h=64, seed=0):
        torch.manual_seed(seed)
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_obs + n_plans, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, obs, plan_onehot):
        return self.net(torch.cat([obs, plan_onehot], -1))[:, 0]
