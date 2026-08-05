"""IQEHead -- thin adapter over frozen trunk features -> phi + IQE quasimetric + mate
anchor (Encoder component; extracted from catspace/research/components/encoder/approaches/cone_fb_embedding/experiments/train_iqe_head.py 2026-07-30)."""
from __future__ import annotations

import torch
import torch.nn as nn

from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE


N_OUTCOME_POLES = 3                                          # win / draw / loss, mover-POV
START = 3                                                    # the time-origin pole
N_POLES = 4                                                  # outcomes + start


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
        # poles[0:3] = win/draw/loss outcomes; poles[3] = START, a TIME ORIGIN, not a basin.
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

        Returns the OUTCOME poles only, deliberately: the basin softmax is a distribution over
        outcomes, and admitting the start pole would make every opening position read as ~25%
        "start" while the three outcome probabilities stopped summing to 1. Keeping the slice here
        also means every pre-existing call site stays correct without modification."""
        return self.iqe.pairwise(es, self.poles[:N_OUTCOME_POLES])

    def d_from_poles(self, es):
        """(B,d) -> (B,3) REVERSE distances d(P_k -> phi). The absorbing term pushes these UP
        while d_poles is pulled DOWN; that gap is the trained asymmetry of the quasimetric."""
        return self.iqe.pairwise(self.poles[:N_OUTCOME_POLES], es).transpose(0, 1)

    def d_from_start(self, es):
        """(B,) d(P_start -> phi): how far the game has travelled. Regressed to log1p(ply+1), so
        this is the field's ABSOLUTE ply coordinate -- the radius of the tent."""
        return self.iqe.pairwise(self.poles[START:START + 1], es).squeeze(0)

    def d_to_start(self, es):
        """(B,) d(phi -> P_start): pushed LARGE. You cannot un-play moves -- chess is genuinely
        irreversible (pawns, captures), so this is the asymmetry encoding a fact about the domain
        rather than a modelling convenience."""
        return self.iqe.pairwise(es, self.poles[START:START + 1]).squeeze(1)

    def d_poles_pairwise(self):
        """(4,4) pairwise pole-to-pole distances, for the non-degeneracy (separation) term.
        Includes START: the time origin should sit far from all three outcomes too."""
        return self.iqe.pairwise(self.poles, self.poles)

    def load_compat(self, state_dict):
        """Load a checkpoint that may predate the poles. Returns the missing keys so callers can
        report (rather than silently accept) a field with untrained poles."""
        # Drop keys whose SHAPE disagrees before loading. strict=False tolerates missing/extra
        # keys but NOT a size mismatch, and pre-start checkpoints carry poles of shape (3,d) --
        # without this they fail to load outright rather than degrading gracefully.
        own = self.state_dict()
        filtered, mismatched, padded = {}, [], []
        for k, v in state_dict.items():
            if k in own and own[k].shape != v.shape:
                # A pre-START checkpoint has poles (3,d) against our (4,d). Dropping it entirely
                # left ALL FOUR poles at random init, so a 3-pole checkpoint silently produced
                # basin probabilities from noise -- which looked like "the poles collapsed" in a
                # layout and cost a wrong diagnosis. Copy the outcome poles into place and leave
                # only the START row uninitialised, which is exactly what a 3-pole field means.
                if k == "poles" and v.ndim == 2 and own[k].ndim == 2 \
                        and v.shape[1] == own[k].shape[1] and v.shape[0] < own[k].shape[0]:
                    merged = own[k].clone()
                    merged[:v.shape[0]] = v
                    filtered[k] = merged
                    padded.append(f"{k}: {v.shape[0]}/{own[k].shape[0]} rows loaded, "
                                  f"rows {v.shape[0]}..{own[k].shape[0]-1} left at init")
                else:
                    mismatched.append(f"{k}{tuple(v.shape)}->{tuple(own[k].shape)}")
            else:
                filtered[k] = v
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        return list(missing) + mismatched + padded, list(unexpected)
