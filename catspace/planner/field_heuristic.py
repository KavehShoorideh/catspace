"""catspace/planner/field_heuristic.py -- turn the quasimetric field into a search
heuristic + goal test for a subgoal region. Board-only geometry (clock/rep zeroed), so
the heuristic matches how iqe_geom was trained. Batched for GPU efficiency."""
from __future__ import annotations

import numpy as np
import torch

from catspace.data.encode import encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.planner.search import Goal

BOARD_ONLY = (18, 19)                     # halfmove clock, repetition -> excluded from geometry


def _planes(boards):
    sp = np.stack([encode_packed(b) for b in boards])
    sm = np.stack([encode_meta(b) for b in boards])
    pl = feature_planes(sp, sm)
    pl[:, BOARD_ONLY] = 0.0
    return pl


def make_field_goal(fb, subgoal_boards, device="cpu", om=None, reach_thresh: float = 2.0,
                    label: str = "") -> Goal:
    """Goal whose heuristic is min_g d(F(board) -> B(subgoal_g)) in plies; reached when that
    field distance <= reach_thresh. `subgoal_boards` is the subgoal region (>=1 positions)."""
    if om is None:
        om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    with torch.no_grad():
        Bsub = fb.embed_B(torch.from_numpy(_planes(list(subgoal_boards))).to(device)).detach()

    def heuristic(boards):
        with torch.no_grad():
            F = fb.embed_F(torch.from_numpy(_planes(list(boards))).to(device),
                           torch.from_numpy(np.tile(om, (len(boards), 1))).to(device))
            return fb.distance_matrix(F, Bsub).min(dim=1).values.cpu().numpy().tolist()

    def reached(board):
        return heuristic([board])[0] <= reach_thresh

    return Goal(heuristic=heuristic, reached=reached, label=label)
