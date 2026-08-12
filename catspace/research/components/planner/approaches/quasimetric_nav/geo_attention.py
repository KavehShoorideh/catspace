#!/usr/bin/env python
"""geo_attention.py -- attention whose similarity IS the quasimetric (Kaveh 2026-08-12:
"how much each token attends to another ... instead of that [QK matrix], be the distance --
distance A and distance B -- and the value being a learned parameter to say it helps,
synergizes, makes it easier or harder").

No W_Q / W_K. The attention logit between subgoal tokens a and b is a per-head learned mix
over their GEOMETRIC relation -- the directed dA/dB between their anchors plus live state
coordinates (the race feature) -- computed fresh each position, never a stored interaction
matrix. Precedents for logits-from-structure: Graphormer's shortest-path attention bias,
AlphaFold's Evoformer pair bias. The quasimetric's asymmetry maps onto attention's inherent
directionality (a->b != b->a) with nothing wasted.

Legibility is a design goal: each head's mixing vector w_h states which geometric relation
that head reads. head_report() prints it.

  G[a,b] (F=7): [log1p dA(a->b), log1p dA(b->a), log1p dB(a->b), log1p dB(b->a),
                 log1p dA(s->b), P(activate b|s), race(a,b)]
  logit_h(a->b) = w_h . g(G[a,b])          g = shared 2-layer MLP
  out_h(a) = sum_b softmax_b(logit_h) * (W_V e_b + U_V g(G[a,b]))
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_GEO = 7
GEO_NAMES = ("dA(a->b)", "dA(b->a)", "dB(a->b)", "dB(b->a)",
             "dA(s->b)", "P(act b)", "race(a,b)")


def pairwise_geometry(dA_fn, dB_fn, Z, dA_s, p_act, race):
    """build G (T,T,F) from anchor embeddings Z (T,d) + per-token state features.
    dA_fn/dB_fn: (n,d),(n,d) -> (n,) directed distances (the IQE rulers)."""
    T = len(Z)
    ii = torch.arange(T).repeat_interleave(T)
    jj = torch.arange(T).repeat(T)
    a2b_A = torch.log1p(dA_fn(Z[ii], Z[jj]).clamp(min=0)).view(T, T)
    a2b_B = torch.log1p(dB_fn(Z[ii], Z[jj]).clamp(min=0)).view(T, T)
    G = torch.stack([a2b_A, a2b_A.T, a2b_B, a2b_B.T,
                     torch.log1p(dA_s.clamp(min=0)).expand(T, T),
                     p_act.expand(T, T),
                     race.expand(T, T)], -1)
    return G


class GeoAttention(nn.Module):
    def __init__(self, d_tok=64, heads=4, d_geo_hidden=32):
        super().__init__()
        self.heads, self.d_tok = heads, d_tok
        assert d_tok % heads == 0
        self.g = nn.Sequential(nn.Linear(N_GEO, d_geo_hidden), nn.GELU(),
                               nn.Linear(d_geo_hidden, d_geo_hidden))
        self.w = nn.Parameter(torch.randn(heads, d_geo_hidden) * 0.2)   # per-head relation mix
        self.v_tok = nn.Linear(d_tok, d_tok)                # value from the concept embedding
        self.v_geo = nn.Linear(d_geo_hidden, d_tok)         # value from the interaction itself
        self.out = nn.Linear(d_tok, d_tok)
        self.ln = nn.LayerNorm(d_tok)

    def forward(self, E, G, return_attn=False):
        """E (T,d_tok) token embeddings, G (T,T,N_GEO) pairwise geometry -> (T,d_tok)."""
        T = len(E)
        H = self.heads
        gg = self.g(G)                                      # (T,T,dg)
        logits = torch.einsum("abd,hd->hab", gg, self.w)    # (H,T,T)
        attn = torch.softmax(logits, dim=-1)
        V = (self.v_tok(E).view(T, H, -1).permute(1, 0, 2)  # (H,T,dv)
             + torch.einsum("hab,abd->had", attn, self.v_geo(gg))
               .view(H, T, H, -1)[torch.arange(H), :, torch.arange(H)])
        mixed = torch.einsum("hab,hbd->had", attn, V)       # (H,T,dv)
        y = self.out(mixed.permute(1, 0, 2).reshape(T, -1))
        y = self.ln(E + y)
        return (y, attn) if return_attn else y

    def head_report(self):
        """which geometric relations each head reads: gradient of the logit wrt each raw
        geometry feature at the origin -- the legibility readout."""
        x = torch.zeros(1, N_GEO, requires_grad=True)
        rep = []
        for h in range(self.heads):
            s = (self.g(x) * self.w[h]).sum()
            g, = torch.autograd.grad(s, x, retain_graph=True)
            rep.append({n: round(float(v), 3) for n, v in zip(GEO_NAMES, g[0])})
        return rep


def _tests():
    torch.manual_seed(0)
    T, d = 6, 64
    E = torch.randn(T, d)
    m = GeoAttention(d_tok=d, heads=4)

    def mk_G(block_pair=None):
        G = torch.rand(T, T, N_GEO) * 2
        if block_pair is not None:                          # plant: token b is CLOSE to a on dA
            a, b = block_pair
            G[a, b, 0] = 0.01
        return G

    ok = True
    y, attn = m(E, mk_G(), return_attn=True)
    ok &= y.shape == (T, d) and torch.allclose(attn.sum(-1), torch.ones(4, T), atol=1e-5)
    # directionality: transposing the directed features must change the logits
    G = mk_G()
    Gt = G.clone()
    Gt[..., 0], Gt[..., 1] = G[..., 1], G[..., 0]
    _, a1 = m(E, G, return_attn=True)
    _, a2 = m(E, Gt, return_attn=True)
    ok &= not torch.allclose(a1, a2)
    # trainability: a planted blocker must become attended-to under a supervised push
    m2 = GeoAttention(d_tok=d, heads=2)
    opt = torch.optim.Adam(m2.parameters(), lr=3e-3)
    Gp = mk_G(block_pair=(0, 3))
    for _ in range(300):
        _, at = m2(E, Gp, return_attn=True)
        loss = -torch.log(at[:, 0, 3] + 1e-9).mean()        # "head must find the blocker"
        opt.zero_grad(); loss.backward(); opt.step()
    _, at = m2(E, Gp, return_attn=True)
    ok &= float(at[:, 0, 3].mean()) > 0.8
    print(f"[geo-attn] shapes/softmax OK | directional: {not torch.allclose(a1, a2)} | "
          f"planted blocker attention {float(at[:, 0, 3].mean()):.2f} (>0.8)")
    print("[geo-attn] head report (relation weights):")
    for h, r in enumerate(m2.head_report()):
        print(f"  head {h}: {r}")
    print("ALL GEO-ATTENTION TESTS PASSED" if ok else "TESTS FAILED")


if __name__ == "__main__":
    _tests()
