"""IQEHead -- thin adapter over frozen trunk features -> phi + IQE quasimetric + mate
anchor (Encoder component; extracted from catspace/research/components/encoder/approaches/cone_fb_embedding/experiments/train_iqe_head.py 2026-07-30)."""
from __future__ import annotations

import torch
import torch.nn as nn

from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE


N_POLES = 3                                                  # win / draw / loss, mover-POV


class IQEHead(nn.Module):
    """thin adapter over frozen trunk features (C,8,8) -> phi (d) + IQE quasimetric + mate anchor
    + three W/D/L basin poles.

    POLES (Kaveh 2026-08-03). Three learned points P_win/P_draw/P_loss act as the vertices of a
    probability simplex: a position's basin probability is a softmax over its negative log
    distances to the three poles (see experiments/losses.py::basin_logits), so the geometry IS
    the committor -- there is no separate WDL head to keep in sync. Terminals are anchored one
    ply from their outcome's pole, which holds the many different mate structures together as
    arrival points on ONE shell instead of letting them drift apart.

    The poles are MOVER-POV, not white/black: lc0 input planes are already side-to-move
    relative, so a single win pole covers both colors by symmetry.

    `mate` is KEPT and is NOT the win pole. They mean different things: `mate` is the tablebase
    DTZ conversion anchor (distance to the <=7p TB-won boundary), the win pole is the eventual
    GAME outcome. Keeping both also means every pre-existing checkpoint keeps its exact
    numerics and `d_mate_emb` semantics -- the planner/MCTS path is untouched by this change.
    Old checkpoints simply lack the pole keys; load them with `load_state_dict(..., strict=False)`
    (see `load_compat`), which leaves the poles at their init and is correct for a field that was
    never trained with them."""

    def __init__(self, in_ch: int = 64, d: int = 64, components: int = 16, adapter_ch: int = 32):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_ch, adapter_ch, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(adapter_ch * 64, d))
        self.iqe = IQE(d, components=components)
        self.mate = nn.Parameter(torch.randn(d) * 0.01)
        self.poles = nn.Parameter(torch.randn(N_POLES, d) * 0.01)
        self.log_T = nn.Parameter(torch.zeros(()))           # basin softmax temperature, T = exp(log_T)

    def phi(self, feats):                                    # (B,C,8,8) -> (B,d)
        return self.adapter(feats)

    def d_pair_emb(self, es, eg):
        return self.iqe(es, eg)

    def d_mate_emb(self, es):
        return self.iqe(es, self.mate.unsqueeze(0).expand(len(es), -1))

    @property
    def temperature(self):
        return torch.exp(self.log_T)

    def d_poles(self, es):
        """(B,d) -> (B,3) forward distances d(phi -> P_k), k = win/draw/loss.
        ONE batched IQE.pairwise call for all three poles -- not three separate calls."""
        return self.iqe.pairwise(es, self.poles)

    def d_from_poles(self, es):
        """(B,d) -> (B,3) REVERSE distances d(P_k -> phi). The absorbing term pushes these UP
        while d_poles is pulled DOWN; that gap is the trained asymmetry of the quasimetric."""
        return self.iqe.pairwise(self.poles, es).transpose(0, 1)

    def d_poles_pairwise(self):
        """(3,3) pairwise pole-to-pole distances, for the non-degeneracy (separation) term."""
        return self.iqe.pairwise(self.poles, self.poles)

    def load_compat(self, state_dict):
        """Load a checkpoint that may predate the poles. Returns the missing keys so callers can
        report (rather than silently accept) a field with untrained poles."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        return list(missing), list(unexpected)
