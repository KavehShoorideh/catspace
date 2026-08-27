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
                 mv_bits=32, n_tok=64, scope="all"):
        super().__init__()
        from vector_quantize_pytorch import LFQ
        from catspace.research.tools.training_infra.quantized_attention import (
            QuantizedAttention)
        self.field, self.arm, self.scope = field, arm, scope
        self.d_tok = field.sq_width
        self.out_bits = out_bits

        # CONCEPT TOKEN LAYOUT. Kaveh 2026-08-26: "why only limit to one set of concepts?
        # This should be on all concepts." It was reading only the 64 square tokens, so a
        # question whose ANSWER is a piece slot ("who is attacking me?" -> "that knight")
        # was not even representable. All four families are tokens now.
        #
        # They have different widths -- board 3 bits per head, squares 24, pieces 24, the
        # edge histogram 128 -- so each family gets its own projection into a shared token
        # width, plus a learned TYPE embedding so attention can tell a square token from a
        # piece token after projection.
        B0 = field.g_heads * field.code_width                 # board block ends
        S0 = B0 + 64 * field.sq_width                          # squares end
        P0 = field.d_state - field.edge_codes                  # pieces end / hist begins
        self.spans = [("board",  0,  B0, field.g_heads, field.code_width),
                      ("square", B0, S0, 64,            field.sq_width),
                      ("piece",  S0, P0, 32,            max(1, (P0 - S0) // 32)),
                      ("hist",   P0, field.d_state, 1,  field.edge_codes)]
        if scope == "squares":
            self.spans = [t for t in self.spans if t[0] == "square"]
        self.n_tok = sum(t[3] for t in self.spans)
        self.fam_proj = nn.ModuleList([nn.Linear(t[4], self.d_tok) for t in self.spans])
        self.fam_emb = nn.Parameter(torch.zeros(len(self.spans), self.d_tok))
        nn.init.normal_(self.fam_emb, std=0.02)

        d_read = field.d_state
        if arm in ("qat", "soft"):
            self.attn = QuantizedAttention(self.d_tok, n_q=n_q, n_head=attn_heads,
                                           top_k=top_k, out_bits=out_bits)
            if arm == "soft":
                self.soft = nn.MultiheadAttention(self.d_tok, attn_heads,
                                                  batch_first=True)
                self.soft_out = nn.Linear(self.d_tok, out_bits)
            d_read = field.d_state + self.n_tok * out_bits
        self.mv_bits = LFQ(dim=mv_bits, codebook_size=2, num_codebooks=mv_bits)
        self.mv_proj = nn.Linear(field.mv_vq_proj.in_features, mv_bits)
        # DYNAMICS MOVE CODES, separate capacity from the readout's value codes. The
        # future heads backprop through their conditioning; sharing the readout's 32 move
        # bits let "what this move changes" (near-deterministic under White-POV, strong
        # gradient) evict "how good is this move" (hard) -- qat7b: fut 0.54 -> 0.11 while
        # regret collapsed to chance and the WDL loss itself destabilised. Mechanics-bits
        # for rolling, value-bits for ranking; both still bits, both saved with the model
        # (the roller needs these codes at inference).
        self.dyn_mv_bits = LFQ(dim=mv_bits, codebook_size=2, num_codebooks=mv_bits)
        self.dyn_mv_proj = nn.Linear(field.mv_vq_proj.in_features, mv_bits)
        self.d_pos, self.d_mv = d_read, mv_bits
        self.readout = nn.Linear(d_read + mv_bits, 3)
        self.d_read = d_read + mv_bits
        # WHITE-POV support: whose turn it is enters as an explicit BIAS, not a concept,
        # so the concepts are free to describe the board alone. Outside the pruner.
        self.turn = nn.Parameter(torch.zeros(2, 3))

    def tokens(self, zq):
        """(B, d_state) -> (B, n_tok, d_tok) concept tokens, and a validity mask.

        Captured pieces are all-zero blocks, and attending to a piece that is not on the
        board is meaningless, so those tokens are MASKED OUT rather than embedded.
        """
        B = len(zq)
        toks, mask = [], []
        for i, (nm, a, b, cnt, w) in enumerate(self.spans):
            blk = zq[:, a:b].reshape(B, cnt, w)
            toks.append(self.fam_proj[i](blk) + self.fam_emb[i])
            if nm == "piece":
                mask.append(blk.abs().sum(-1) > 0)          # dead slots do not answer
            else:
                mask.append(torch.ones(B, cnt, dtype=torch.bool, device=zq.device))
        return torch.cat(toks, 1), torch.cat(mask, 1)

    def level2(self, zq):
        if self.arm == "level1":
            return None, {}
        tok, m = self.tokens(zq)
        if self.arm == "soft":
            h, _ = self.soft(tok, tok, tok, key_padding_mask=~m)
            return torch.tanh(self.soft_out(h)).reshape(len(zq), -1), {}
        bits, aux = self.attn(tok, tok_mask=m)
        return bits.reshape(len(zq), -1), aux

    def forward(self, zq, h_mv, stm=None):
        """-> (B, MV_CAP, 3): a WDL logit for every child move."""
        l2, aux = self.level2(zq)
        z = zq if l2 is None else torch.cat([zq, l2], -1)
        mvb, _, vq2 = self.mv_bits(self.mv_proj(h_mv))
        W = self.readout.weight
        pos = F.linear(z, W[:, :self.d_pos], self.readout.bias)
        mov = F.linear(mvb, W[:, self.d_pos:])
        if stm is not None:
            # WHITE-POV: move bits mean "good for the MOVER" universally; this sign maps
            # that into White's frame. Without it, white and black rows push W_mv in
            # OPPOSITE directions for the same move geometry and the gradients cancel --
            # observed in qat7: regret loss flat at ~0.25 for 1000 steps, top-1 at the
            # slot-0 baseline, while balanced accuracy learned normally off the pos term.
            pos = pos + self.turn[stm]
            # The color swap acts on (W,D,L) as the PERMUTATION W<->L with D fixed --
            # NOT as negation. Negating all three columns (the previous fix) coincides on
            # the W-L ranking axis, which is why ranking briefly learned, but it forces
            # the D column to be ANTISYMMETRIC under side-to-move -- while a drawish move
            # is drawish for BOTH sides. That unsatisfiable constraint destabilised the
            # whole readout in qat7b/qat7c (draw recall 0.01 -> 0.57 -> 0.01, WDL loss
            # rising, ranking collapsing with it).
            mov = torch.where((stm == 0)[:, None, None],
                              mov[:, :, [2, 1, 0]], mov)
        aux = dict(aux)
        aux["vq_loss"] = aux.get("vq_loss", 0.0) + vq2
        return pos[:, None, :] + mov, z, aux
