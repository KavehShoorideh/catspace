"""QGT graph base with PLUGGABLE MIXERS (Kaveh: right abstraction, no repeated code).

One GraphBase owns the layer loop -- relation message-passing + mixer + FFN, written
once. What mixes information across nodes is an injected strategy:

    NoMixer            pure GNN         (QGT locked spec: quantize AFTER this base,
                                         then stacked QuantizedAttention levels)
    DenseAttnMixer     nn.MultiheadAttention   (the incumbent's shape, for ablation)
    QuantizedAttnMixer QuantizedAttention IN the trunk on continuous tokens
                       (Kaveh's "quantized attention on the GNN base directly" reading)

router_losses() implements the ST-MoE stabilisers once for BOTH quantized placements:
z-loss (1/B)(logsumexp)^2 at c_z=1e-3 on question and attention logits (ST-MoE eq.5 +
footnote 4), Switch balance N*sum(f_i*P_i) at c_B=0.01.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from catspace.research.components.encoder.approaches.reach_probability.experiments \
    .graph_move_policy import Relation

RELS = ("att", "src", "dst", "mut", "vac", "dsc")


def router_losses(qlog, attn_logit, c_z=1e-3, c_b=1e-2):
    """qlog: (B,T,H,R) question logits. attn_logit: (B,H,T,T) pre-softmax."""
    z_q = torch.logsumexp(qlog, -1).pow(2).mean()
    z_a = torch.logsumexp(attn_logit.clamp(min=-30), -1).pow(2).mean()
    R = qlog.shape[-1]
    P = F.softmax(qlog, -1).mean(dim=(0, 1))            # (H,R) mean router prob
    f = F.one_hot(qlog.argmax(-1), R).float().mean(dim=(0, 1))   # (H,R) route fraction
    balance = (R * (f * P).sum(-1)).mean()
    return c_z * (z_q + z_a) + c_b * balance


class NoMixer(nn.Module):
    def forward(self, h, pad):
        return torch.zeros_like(h), {}


class DenseAttnMixer(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, h, pad):
        a, _ = self.mha(h, h, h, key_padding_mask=pad, need_weights=False)
        return a, {}


class QuantizedAttnMixer(nn.Module):
    def __init__(self, d, n_q=16, heads=4, top_k=4, out_bits=8):
        super().__init__()
        from catspace.research.tools.training_infra.quantized_attention import (
            QuantizedAttention)
        self.qa = QuantizedAttention(d, n_q=n_q, n_head=heads, top_k=top_k,
                                     out_bits=out_bits)
        self.proj = nn.Linear(out_bits, d)

    def forward(self, h, pad):
        bits, aux = self.qa(h, tok_mask=~pad)
        loss = aux.get("vq_loss", torch.zeros((), device=h.device))
        if "qlog" in aux and "attn_logit" in aux:
            loss = loss + router_losses(aux["qlog"], aux["attn_logit"])
        return self.proj(bits), {"mixer_loss": loss}


class GraphBase(nn.Module):
    """Relation MPNN + injected mixer + FFN, GPS-parallel pre-LN form. Written once."""

    def __init__(self, d=128, layers=3, mixer_factory=None, rels=RELS,
                 n_nodes=None, d_sq_in=28, d_mv_in=19):
        super().__init__()
        from catspace.research.components.encoder.approaches.reach_probability \
            .experiments.graph_move_policy import MV_CAP
        self.d, self.rels = d, rels
        self.n_nodes = n_nodes or (64 + MV_CAP)
        self.sq_in = nn.Linear(d_sq_in, d)
        self.mv_in = nn.Linear(d_mv_in, d)
        mf = mixer_factory or (lambda: NoMixer())
        self.rel = nn.ModuleList([nn.ModuleDict({r: Relation(d) for r in rels})
                                  for _ in range(layers)])
        self.mix = nn.ModuleList([mf() for _ in range(layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                                nn.Linear(2 * d, d))
                                  for _ in range(layers)])
        self.ln_l = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ln_g = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ln_f = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])

    def forward(self, sqf, mvf, edges, mv_mask):
        B = sqf.shape[0]
        h = torch.cat([self.sq_in(sqf), self.mv_in(mvf)], 1)
        pad = torch.cat([torch.zeros(B, 64, dtype=torch.bool, device=sqf.device),
                         ~mv_mask], 1)
        aux_total, per_layer = torch.zeros((), device=sqf.device), []
        for i in range(len(self.rel)):
            xf = self.ln_l[i](h).reshape(B * self.n_nodes, self.d)
            agg = 0.0
            for r in self.rels:
                sd = edges.get(r)
                if sd is not None and len(sd[0]):
                    agg = agg + self.rel[i][r](xf, sd[0], sd[1], B * self.n_nodes)
            a_l = (agg.reshape(B, self.n_nodes, self.d) if torch.is_tensor(agg)
                   else torch.zeros_like(h))
            a_g, aux = self.mix[i](self.ln_g[i](h), pad)
            if "mixer_loss" in aux:
                aux_total = aux_total + aux["mixer_loss"]
            h = h + a_l + a_g
            h = h + self.ffn[i](self.ln_f[i](h))
            per_layer.append(h)
        return h, {"mixer_loss": aux_total, "per_layer": per_layer}


def _test():
    torch.manual_seed(0)
    from catspace.research.components.encoder.approaches.reach_probability \
        .experiments.graph_move_policy import MV_CAP
    B, NN = 3, 64 + MV_CAP
    sqf, mvf = torch.randn(B, 64, 28), torch.randn(B, MV_CAP, 19)
    mask = torch.zeros(B, MV_CAP, dtype=torch.bool); mask[:, :20] = True
    edges = {"att": (torch.randint(0, B * NN, (200,)), torch.randint(0, B * NN, (200,)))}
    outs = {}
    for name, mf in (("none", None), ("dense", lambda: DenseAttnMixer(64)),
                     ("quant", lambda: QuantizedAttnMixer(64))):
        gb = GraphBase(d=64, layers=2, mixer_factory=mf)
        h, aux = gb(sqf, mvf, edges, mask)
        assert h.shape == (B, NN, 64), (name, h.shape)
        assert torch.isfinite(h).all() and torch.isfinite(aux["mixer_loss"])
        outs[name] = gb
    n_mha = sum(1 for m in outs["none"].modules() if isinstance(m, nn.MultiheadAttention))
    assert n_mha == 0, "NoMixer base contains dense attention"
    # router_losses sanity: peaked routing must cost more balance than uniform
    ql_u = torch.zeros(4, 10, 2, 8); ql_p = torch.full((4, 10, 2, 8), -9.0); ql_p[..., 0] = 9.0
    al = torch.randn(4, 2, 10, 10)
    assert router_losses(ql_p, al) > router_losses(ql_u, al), "balance loss inverted"
    h, aux = outs["quant"](sqf, mvf, edges, mask)
    assert float(aux["mixer_loss"]) != 0.0, "router losses silently off (qlog not exposed)"
    (h.sum() + aux["mixer_loss"]).backward()
    g = outs["quant"].rel[0]["att"].w.weight.grad
    assert g is not None and float(g.abs().sum()) > 0
    print(f"  three mixers, one loop: shapes OK, no MHA under NoMixer, balance(peaked) > "
          f"balance(uniform), gradients flow through relations+mixer")
    print("  QGT BASE TESTS PASS")


if __name__ == "__main__":
    _test()
