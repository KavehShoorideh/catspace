"""DTMTok -- piece-token transformer for the last-mile DTM (Kaveh 2026-07-25:
'shouldn't we train a transformer for this instead of a CNN?').

Endgames are SPARSE (3-7 pieces on 64 squares) and DTM is RELATIONAL (opposition,
cutoffs, king distance): tokens = one per piece (type+color emb + square emb) + a
side-to-move CLS token; a small TransformerEncoder computes pairwise relations in one
hop; CLS head regresses dtm/scale.

Extracted from catspace/research/components/encoder/approaches/jepa_tokenizer/experiments/train_dtm_tok.py in the 2026-08-03 restructure. The engine
loads this checkpoint at play time, so the architecture is shipping code and cannot live
in a training script -- the trainer imports it from here.
"""
from __future__ import annotations

import numpy as np
import torch.nn as nn

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed

MAX_TOKENS = 8          # <=7 pieces in <=6-man + CLS


class DTMTok(nn.Module):
    def __init__(self, d: int = 128, heads: int = 4, layers: int = 3, seed: int = 0):
        import torch
        torch.manual_seed(seed)
        super().__init__()
        self.config = dict(d=d, heads=heads, layers=layers, seed=seed)
        self.emb_piece = nn.Embedding(13, d)     # 0 CLS, 1-6 white, 7-12 black
        self.emb_sq = nn.Embedding(65, d)        # 0 CLS slot, 1-64 squares
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                         batch_first=True, dropout=0.0, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, 1))

    def forward(self, piece_ids, sq_ids, pad):
        tok = self.emb_piece(piece_ids) + self.emb_sq(sq_ids)
        h = self.enc(tok, src_key_padding_mask=pad)
        return self.head(h[:, 0, :])[:, 0]       # CLS


def tokenize(P, M):
    n = len(P)
    pid = np.zeros((n, MAX_TOKENS), np.int64)
    sqi = np.zeros((n, MAX_TOKENS), np.int64)
    pad = np.ones((n, MAX_TOKENS), bool)
    for i in range(n):
        b = board_from_packed(P[i], M[i])
        pad[i, 0] = False                        # CLS
        j = 1
        for sq, p in sorted(b.piece_map().items()):
            pid[i, j] = p.piece_type + (0 if p.color else 6)
            sqi[i, j] = sq + 1
            pad[i, j] = False
            j += 1
        if not b.turn:                            # side-to-move in the CLS square slot
            sqi[i, 0] = 0; pid[i, 0] = 0
        # white-to-move CLS keeps ids 0/0; black-to-move flagged via piece id 12+... keep
        # simple: encode stm by giving CLS sq id 64 when black to move
        sqi[i, 0] = 64 if not b.turn else 0
    return pid, sqi, pad
