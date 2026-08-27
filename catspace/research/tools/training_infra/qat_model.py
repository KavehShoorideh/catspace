"""The quantized-attention readout, importable.

It lived inside qat_train.main() as a closure, so nothing except the trainer could build
one -- which blocks every downstream user (planner, arena, probes). Same architecture,
same tensor layout, same checkpoints; only the scope changed.

    level 1   binary/ternary concepts from the trunk        2528 dims
    level 2   quantized attention over the 64 SQUARE tokens  512 dims  (arm qat/soft)
    move      per-move binary concepts                        32 dims
    readout   ONE Linear(-> 3), factorised into a position term shared by all siblings
              plus a per-move term that carries the entire ranking (dV/dmove)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QATReadout(nn.Module):
    def __init__(self, field, arm="qat", n_q=16, attn_heads=4, top_k=4, out_bits=8,
                 mv_bits=32, n_tok=64):
        super().__init__()
        from vector_quantize_pytorch import LFQ
        from catspace.research.tools.training_infra.quantized_attention import (
            QuantizedAttention)
        self.field, self.arm = field, arm
        self.n_tok, self.d_tok = n_tok, field.sq_width
        self.out_bits = out_bits
        d_read = field.d_state
        if arm in ("qat", "soft"):
            self.attn = QuantizedAttention(self.d_tok, n_q=n_q, n_head=attn_heads,
                                           top_k=top_k, out_bits=out_bits)
            if arm == "soft":
                self.soft = nn.MultiheadAttention(self.d_tok, attn_heads,
                                                  batch_first=True)
                self.soft_out = nn.Linear(self.d_tok, out_bits)
            d_read = field.d_state + n_tok * out_bits
        self.mv_bits = LFQ(dim=mv_bits, codebook_size=2, num_codebooks=mv_bits)
        self.mv_proj = nn.Linear(field.mv_vq_proj.in_features, mv_bits)
        self.d_pos, self.d_mv = d_read, mv_bits
        self.readout = nn.Linear(d_read + mv_bits, 3)
        self.d_read = d_read + mv_bits

    def level2(self, zq):
        if self.arm == "level1":
            return None, {}
        b0 = self.field.g_heads * self.field.code_width
        tok = zq[:, b0:b0 + self.n_tok * self.d_tok].reshape(-1, self.n_tok, self.d_tok)
        if self.arm == "soft":
            h, _ = self.soft(tok, tok, tok)
            return torch.tanh(self.soft_out(h)).reshape(len(zq), -1), {}
        bits, aux = self.attn(tok)
        return bits.reshape(len(zq), -1), aux

    def forward(self, zq, h_mv):
        """-> (B, MV_CAP, 3): a WDL logit for every child move."""
        l2, aux = self.level2(zq)
        z = zq if l2 is None else torch.cat([zq, l2], -1)
        mvb, _, vq2 = self.mv_bits(self.mv_proj(h_mv))
        W = self.readout.weight
        pos = F.linear(z, W[:, :self.d_pos], self.readout.bias)
        mov = F.linear(mvb, W[:, self.d_pos:])
        aux = dict(aux)
        aux["vq_loss"] = aux.get("vq_loss", 0.0) + vq2
        return pos[:, None, :] + mov, z, aux
