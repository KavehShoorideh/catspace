"""IQEHead -- thin adapter over frozen trunk features -> phi + IQE quasimetric + mate
anchor (Encoder component; extracted from experiments/train_iqe_head.py 2026-07-30)."""
from __future__ import annotations

import torch
import torch.nn as nn

from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE


class IQEHead(nn.Module):
    """thin adapter over frozen trunk features (C,8,8) -> phi (d) + IQE quasimetric + mate anchor."""

    def __init__(self, in_ch: int = 64, d: int = 64, components: int = 16, adapter_ch: int = 32):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_ch, adapter_ch, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(adapter_ch * 64, d))
        self.iqe = IQE(d, components=components)
        self.mate = nn.Parameter(torch.randn(d) * 0.01)

    def phi(self, feats):                                    # (B,C,8,8) -> (B,d)
        return self.adapter(feats)

    def d_pair_emb(self, es, eg):
        return self.iqe(es, eg)

    def d_mate_emb(self, es):
        return self.iqe(es, self.mate.unsqueeze(0).expand(len(es), -1))
