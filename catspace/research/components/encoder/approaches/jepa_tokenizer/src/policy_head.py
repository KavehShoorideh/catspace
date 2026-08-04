"""
nn/policy_head.py — a MOVE-PRIOR head reading ONLY the field F(s) (Kaveh
2026-07-19: "policy head it is, but only using the field").

Purpose: give MCTS all of a node's child priors in ONE forward pass of the
PARENT's F, so expansion costs ~1 eval instead of branching-many child evals.
That is the AlphaZero/Leela design and the fix for value-only expansion making
the node budget count evals-not-simulations (the "all moves same visit count"
symptom). It is also a field diagnostic: top-1 move-prediction accuracy from F
measures how much game structure F carries.

Move space: from_square*64 + to_square = 4096 slots (promotions collapse onto
their from-to slot -- fine for a prior; MCTS still creates the separate
promotion children and shares the slot's prior). Reads F only, never the board.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

N_MOVES = 64 * 64          # from_square * 64 + to_square


def move_index(move) -> int:
    return move.from_square * 64 + move.to_square


class PolicyHead(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, seed: int = 0):
        torch.manual_seed(seed)
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, N_MOVES))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)


def policy_loss(head: PolicyHead, f: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the head's logits vs the played-move from-to index."""
    return nn.functional.cross_entropy(head(f), target_idx)


def legal_priors(logits_row: np.ndarray, board) -> dict:
    """Softmax over the legal moves' from-to slots -> {move: prior}. Illegal
    slots are dropped, so the prior is always a valid distribution over the
    node's actual children."""
    moves = list(board.legal_moves)
    if not moves:
        return {}
    z = np.asarray([logits_row[move_index(m)] for m in moves], dtype=np.float64)
    z -= z.max()
    p = np.exp(z)
    p /= (p.sum() + 1e-12)
    return {m: float(pi) for m, pi in zip(moves, p)}
