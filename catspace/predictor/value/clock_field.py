"""ClockField -- the M0/analysis committor net (ValueOracle component; extracted
from experiments/train_clock_field.py 2026-07-30)."""
from __future__ import annotations

import torch
import torch.nn as nn

from catspace.nn.iqe import IQE

# Ending-type categories (Kaveh's categorical head: "what kind of end is approaching").
# Order is the label index. Draws in the middle, decisive at the ends. Canonical home
# (component refactor 2026-07-30); experiments/losses.py imports from here.
ENDINGS = ["WIN_MATE", "DRAW_FIFTY", "DRAW_STALEMATE", "DRAW_INSUFFICIENT",
           "DRAW_REPETITION", "LOSS_MATE"]
N_ENDINGS = len(ENDINGS)


class ClockField(nn.Module):
    """conv over 20 feature planes -> phi; d(s)=IQE(phi(s), MATE) + CATEGORICAL ending-type head
    (Kaveh: 'what kind of end is approaching'). Sees the halfmove clock."""
    def __init__(self, d=32, ch=64, blocks=5, iqe_components=16, in_planes=20):
        # in_planes: 20 = single-position (endgame). Full board later = 8-history lc0 stack
        # (~112 planes) -> pass in_planes=112; the rest of the net is unchanged. NOT endgame-locked.
        super().__init__()
        self.in_planes = in_planes
        self.stem = nn.Sequential(nn.Conv2d(in_planes, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch)) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv2d(ch, 32, 1), nn.GroupNorm(8, 32), nn.ReLU(),
                                  nn.Flatten(), nn.Linear(32 * 64, d))
        self.iqe = IQE(d, components=iqe_components)
        self.mate = nn.Parameter(torch.randn(d) * 0.1)
        self.cat = nn.Linear(d, N_ENDINGS)                   # categorical / DISTRIBUTIONAL ending head
        # score per ending (WIN_MATE, DRAW_FIFTY, STALE, INSUF, REP, LOSS_MATE) -> committor readout
        self.register_buffer("outcome_score", torch.tensor([1., .5, .5, .5, .5, 0.]))

    def committor(self, x):
        """TRAINED committor / expected score over ALL outcome types = score-weighted ending
        distribution. Endgame (deterministic) -> ~0/0.5/1; stochastic midgame -> a real distribution.
        Replaces the exp(-d) proxy as the MCTS value; trained by the categorical ending loss."""
        p = torch.softmax(self.cat(self.phi(x)), dim=-1)
        return (p * self.outcome_score).sum(-1)

    def phi(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = torch.relu(h + b(h))
        return self.head(h)

    def d_mate_and_end(self, x):
        e = self.phi(x)
        return self.iqe(e, self.mate.expand_as(e)), self.cat(e)

    def d_mate(self, x):
        e = self.phi(x)
        return self.iqe(e, self.mate.expand_as(e))

    def d_pair(self, xs, xg):                                # SHARED phi -> triangle-safe quasimetric
        return self.iqe(self.phi(xs), self.phi(xg))

    def d_pair_emb(self, es, eg):
        return self.iqe(es, eg)
