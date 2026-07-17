"""
nn/iqe.py — Interval Quasimetric Embedding (Wang & Isola 2022), the
valid-and-universal quasimetric head the merged paper (adversarial_reachability)
adopts in place of the MRN score.

Construction (union-of-intervals form, provably a quasimetric) -- paper
convention (Wang & Isola 2022), verified by the direction test:
  reshape each latent to (C components, K per component). For an ordered pair
  (u, v) and component c, form the directed intervals {[u_ck, v_ck] : v_ck >
  u_ck} and take the Lebesgue measure of their UNION as the per-component
  distance d_c(u->v) -- i.e. length accumulates where v EXCEEDS u, so d(u->v)
  is ~0 when u dominates v ("already reached") and LARGE when v exceeds u
  ("must climb"). Combine components by a learned mix of max and mean (both
  preserve the quasimetric axioms):
      d(u->v) = alpha * max_c d_c  +  (1-alpha) * mean_c d_c,   alpha in [0,1].

Why this is a quasimetric BY CONSTRUCTION (axiom-tested below, not assumed):
  * d(u,u) = 0     -- v_ck > u_ck is never true when v==u, so every interval
                      is empty.
  * asymmetric     -- the intervals for u->v (where v exceeds u) differ from
                      v->u (where u exceeds v).
  * triangle ineq. -- per coordinate k, the u->w interval [u_k,w_k] is COVERED
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
    dim. Empty intervals must have l_k == r_k. (..., K) -> (...).

    Vectorized (no per-k loop): sort by left endpoint; the running max-right
    BEFORE interval i is cummax(r_sorted) shifted by one, so each interval's
    fresh coverage is clamp(r_i - max(l_i, prev_max_r), 0) and the union
    length is their sum. One sort + one cummax + elementwise -- fast enough
    for the (N,M,components,K) tensors InfoNCE builds every step."""
    order = torch.argsort(l, dim=-1)
    ls = torch.gather(l, -1, order)
    rs = torch.gather(r, -1, order)
    cummax_r = torch.cummax(rs, dim=-1).values
    prev_max_r = torch.cat([torch.full_like(cummax_r[..., :1], float("-inf")),
                            cummax_r[..., :-1]], dim=-1)
    start = torch.maximum(ls, prev_max_r)
    return torch.clamp(rs - start, min=0.0).sum(dim=-1)


class IQE(nn.Module):
    """Interval quasimetric on (N,d) latents. d = components * dim_per_comp."""

    def __init__(self, d: int, components: int = 8):
        super().__init__()
        assert d % components == 0, "d must divide into equal components"
        self.components = components
        self.k = d // components
        self.alpha_logit = nn.Parameter(torch.zeros(()))   # sigmoid -> mix in [0,1]
        # learnable output scale: lets absolute calibration (ply-gap) adjust the
        # distance SCALE without shrinking the embeddings back into the
        # degenerate small-coordinate regime (diagnosed 2026-07-17)
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """(N,d) x (N,d) -> (N,) directed distance d(u->v). For all-pairs use
        pairwise()."""
        U = u.reshape(*u.shape[:-1], self.components, self.k)
        V = v.reshape(*v.shape[:-1], self.components, self.k)
        # paper convention (Wang-Isola 2022): interval [U, max(U,V)], nonempty
        # (length V-U) exactly where V exceeds U -> d(u->v). (Was flipped once to
        # [V,U] = d(v->u), the reverse direction, which made InfoNCE fight time's
        # arrow -- guarded now by test_invariants.test_iqe_direction_semantics.)
        lo = U
        hi = torch.maximum(U, V)
        dc = _union_length(lo, hi)                          # (N, components)
        alpha = torch.sigmoid(self.alpha_logit)
        d = alpha * dc.amax(dim=-1) + (1 - alpha) * dc.mean(dim=-1)
        return torch.exp(self.log_scale) * d

    def pairwise(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """(N,d) x (M,d) -> (N,M) directed distances d(u_i -> v_j)."""
        n, m = u.shape[0], v.shape[0]
        U = u.reshape(n, 1, self.components, self.k).expand(n, m, self.components, self.k)
        V = v.reshape(1, m, self.components, self.k).expand(n, m, self.components, self.k)
        lo = U
        hi = torch.maximum(U, V)                            # [U, max(U,V)]: d(u->v)
        dc = _union_length(lo, hi)                          # (N, M, components)
        alpha = torch.sigmoid(self.alpha_logit)
        d = alpha * dc.amax(dim=-1) + (1 - alpha) * dc.mean(dim=-1)
        return torch.exp(self.log_scale) * d
