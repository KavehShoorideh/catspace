"""
nn/iqe.py — Interval Quasimetric Embedding (Wang & Isola 2022), the
valid-and-universal quasimetric head the merged paper (adversarial_reachability)
adopts in place of the MRN score.

Construction (union-of-intervals form, provably a quasimetric):
  reshape each latent to (C components, K per component). For an ordered pair
  (u, v) and component c, form the directed intervals {[v_ck, u_ck] : u_ck >
  v_ck} and take the Lebesgue measure of their UNION as the per-component
  distance d_c(u->v). Combine components by a learned mix of max and mean
  (both preserve the quasimetric axioms):
      d(u->v) = alpha * max_c d_c  +  (1-alpha) * mean_c d_c,   alpha in [0,1].

Why this is a quasimetric BY CONSTRUCTION (axiom-tested below, not assumed):
  * d(u,u) = 0     -- u_ck > u_ck is never true, so every interval is empty.
  * asymmetric     -- the intervals for u->v (where u exceeds v) differ from
                      v->u (where v exceeds u).
  * triangle ineq. -- per coordinate k, the u->w interval [w_k,u_k] is COVERED
                      by the union of the u->v and v->w intervals (case check
                      over the orderings of u_k,v_k,w_k), and the union measure
                      is subadditive; max and mean of quasimetrics are
                      quasimetrics.
Universality (any quasimetric on a finite set is approximable to arbitrary
accuracy by an IQE of sufficient width) is Wang & Isola's theorem.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _union_length(l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Lebesgue measure of the union of intervals [l_k, r_k] along the last
    dim. Empty intervals must have l_k == r_k. (..., K) -> (...). Exact via a
    sort + sequential sweep (K is the small per-component dim)."""
    order = torch.argsort(l, dim=-1)
    ls = torch.gather(l, -1, order)
    rs = torch.gather(r, -1, order)
    K = ls.shape[-1]
    total = torch.zeros(ls.shape[:-1], dtype=ls.dtype, device=ls.device)
    cur_r = torch.full(ls.shape[:-1], float("-inf"), dtype=ls.dtype, device=ls.device)
    for k in range(K):
        lk, rk = ls[..., k], rs[..., k]
        # new coverage beyond what's already covered up to cur_r
        start = torch.maximum(lk, cur_r)
        total = total + torch.clamp(rk - start, min=0.0)
        cur_r = torch.maximum(cur_r, rk)
    return total


class IQE(nn.Module):
    """Interval quasimetric on (N,d) latents. d = components * dim_per_comp."""

    def __init__(self, d: int, components: int = 8):
        super().__init__()
        assert d % components == 0, "d must divide into equal components"
        self.components = components
        self.k = d // components
        self.alpha_logit = nn.Parameter(torch.zeros(()))   # sigmoid -> mix in [0,1]

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """(N,d) x (N,d) -> (N,) directed distance d(u->v). For all-pairs use
        pairwise()."""
        U = u.reshape(*u.shape[:-1], self.components, self.k)
        V = v.reshape(*v.shape[:-1], self.components, self.k)
        # directed intervals [V, U] where U > V; empty ones set l==r so they
        # contribute nothing to the union
        lo = torch.minimum(U, V)
        hi = U
        empty = U <= V
        lo = torch.where(empty, hi, lo)                    # collapse empties to points
        dc = _union_length(lo, hi)                          # (N, components)
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * dc.amax(dim=-1) + (1 - alpha) * dc.mean(dim=-1)

    def pairwise(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """(N,d) x (M,d) -> (N,M) directed distances d(u_i -> v_j)."""
        n, m = u.shape[0], v.shape[0]
        U = u.reshape(n, 1, self.components, self.k).expand(n, m, self.components, self.k)
        V = v.reshape(1, m, self.components, self.k).expand(n, m, self.components, self.k)
        lo = torch.minimum(U, V)
        hi = U
        empty = U <= V
        lo = torch.where(empty, hi, lo)
        dc = _union_length(lo, hi)                          # (N, M, components)
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * dc.amax(dim=-1) + (1 - alpha) * dc.mean(dim=-1)
