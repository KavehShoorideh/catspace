"""ReachModel component: region-level first-hit probabilities and plies from a
trained reach head, at a fixed play-time context (population z, fixed Elos)."""
from __future__ import annotations

import numpy as np
import torch


class RegionReach:
    """Wraps a SubgoalRanker's field + goal bank: phis -> (P(hit), E[plies|hit])."""

    def __init__(self, rk, elo_self: float, elo_oppo: float):
        self.rk = rk
        self.elo_self = elo_self; self.elo_oppo = elo_oppo
        self._bank_np = rk.bank.cpu().numpy()

    @property
    def bank(self):
        return self._bank_np                                # (G, 64) phi centroids

    @torch.no_grad()
    def heads(self, phis):
        """(B,64) phi -> (p_hit (B,G), plies (B,G)) under the population-z context."""
        B = len(phis)
        m = self.rk.model
        f = torch.as_tensor(np.asarray(phis, np.float32), device=self.rk.dev)
        zs = torch.zeros(B, 16, device=self.rk.dev)
        ctx = [torch.tensor([[(self.elo_self - 1500) / 400, (self.elo_oppo - 1500) / 400,
                              1.0, 1.0]], dtype=torch.float32, device=self.rk.dev).expand(B, -1)]
        if m.state[0].in_features > 84:                     # two-z field: z_opp cold start
            ctx.append(torch.cat([torch.zeros(B, 16, device=self.rk.dev),
                                  torch.zeros(B, 1, device=self.rk.dev)], 1))
        sh, st = m.state_embs(f, zs, torch.cat(ctx, 1))
        p = torch.sigmoid(sh @ self.rk.gh.T * m.scale + m.b_hit)
        plies = torch.expm1((st @ self.rk.gt.T * m.scale + m.b_time).clamp(0, 8))
        return p.cpu().numpy(), plies.cpu().numpy()

    def region_of(self, phi):
        """the navigator's hit rule: phi-space nearest centroid (argmin, no eps)."""
        return int(((self._bank_np - phi) ** 2).sum(1).argmin())
