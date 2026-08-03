"""catspace/style/model.py -- M2b player STYLE-RESIDUAL z (Matilda-style; see memory
matilda_residual_style_embedding + style_z_allows_strength). A frozen Maia-2 rating base supplies the
per-move log-prob l0(m|s,Elo); on top we add an ADDITIVE, player-specific residual:

    logit_P(m|s) = l0(m) + z_P . U(s,m),   p_P = softmax over the candidate set (Maia top-K u {played}).

U(s,m) in R^dz is a SHARED, player-independent "style-axis" head over [phi(s) (frozen ReachabilityField),
a learned move embedding, the base log-prob, the base rank]. z_P in R^dz is the player coefficient. The
logit is LINEAR in z_P given U, so recovering z for a held-out / unseen / mid-game opponent is a CONVEX
MAP fit (catspace/style/recover.py) -> a Laplace posterior (the M2c in-game-tightening hook).

Prior p(z|Elo) (locked decision 4): z_P = mu(Elo_P) + Delta_P. TRAIN individual players own a free Delta
row; PROVISIONAL and HELD-OUT players carry Delta=0 so z = mu(Elo) -- so the provisional pool's moves
ESTIMATE mu(Elo) (Kaveh 2026-07-27). Gaussian prior lam*||Delta||^2. z=0 (no style) reproduces Maia-2
exactly (identity gate). Per Kaveh, z is ALLOWED to carry strength / structure-competence -- no purge.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 1858                                              # lc0 / Maia move vocabulary size


def elo_norm(elo):
    return (elo.float() - 1500.0) / 400.0


class StyleResidual(nn.Module):
    def __init__(self, n_individual: int, d_phi: int = 64, d_move: int = 32, d_z: int = 16,
                 hidden: int = 128, lam_prior: float = 1.0, learn_mu: bool = False):
        super().__init__()
        self.d_z = d_z; self.lam_prior = lam_prior; self.learn_mu = learn_mu
        self.move_emb = nn.Embedding(VOCAB + 1, d_move, padding_idx=VOCAB)   # last idx = pad
        self.U = nn.Sequential(nn.Linear(d_phi + d_move + 2, hidden), nn.ReLU(),
                               nn.Linear(hidden, hidden), nn.ReLU(),
                               nn.Linear(hidden, d_z))
        # prior mean p(z|Elo): DEFAULT mu=0 -- raw Maia IS the universal rating prior (trained on the
        # whole population), and a casual-pool-fit mu on active players is a population mismatch that
        # also blows up via the mu/Delta gauge. See memory opponent_base_rate_and_mu_zero.
        self.mu = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, d_z)) if learn_mu else None
        self.delta = nn.Embedding(max(n_individual, 1), d_z)                 # per-train-player deviation
        nn.init.zeros_(self.delta.weight)                                    # identity-at-init

    # --- style axes U(s,m): (B,K,d_z) ---
    def U_of(self, phi, cand_idx, cand_logp, rank):
        B, K = cand_idx.shape
        me = self.move_emb(cand_idx)                                         # (B,K,d_move)
        phe = phi.unsqueeze(1).expand(-1, K, -1)                             # (B,K,d_phi)
        x = torch.cat([phe, me, cand_logp.unsqueeze(-1), rank.unsqueeze(-1)], -1)
        return self.U(x)                                                     # (B,K,d_z)

    def mu_of(self, elo):
        """prior mean mu(Elo). DEFAULT 0 (raw Maia is the rating prior); learned only if learn_mu."""
        if self.mu is None:
            return elo.new_zeros(elo.shape[0], self.d_z)
        return self.mu(elo_norm(elo).unsqueeze(-1))

    # --- z_P = mu(Elo) + Delta (Delta only for train players; pidx<0 -> prior mean) ---
    def z_of(self, pidx, elo):
        mu = self.mu_of(elo)                                                # (B,d_z), 0 by default
        z = mu
        ind = pidx >= 0
        if ind.any():
            d = torch.zeros_like(mu)
            d[ind] = self.delta(pidx.clamp(min=0))[ind]
            z = mu + d
        return z, mu

    def logits(self, z, U, cand_logp, cand_mask):
        style = (U * z.unsqueeze(1)).sum(-1)                                # (B,K)
        logit = cand_logp + style                                          # additive residual in logit space
        return logit.masked_fill(~cand_mask, -1e9)

    def nll(self, logit, played_slot):
        logp = F.log_softmax(logit, dim=-1)                                # over candidate set
        return -logp.gather(1, played_slot.view(-1, 1)).squeeze(1)         # (B,)

    def forward(self, phi, cand_idx, cand_logp, cand_mask, rank, played_slot, pidx, elo):
        """returns (per-position NLL (B,), prior scalar). Prior penalises only train-player Deltas."""
        U = self.U_of(phi, cand_idx, cand_logp, rank)
        z, _ = self.z_of(pidx, elo)
        nll = self.nll(self.logits(z, U, cand_logp, cand_mask), played_slot)
        ind = pidx >= 0
        prior = (self.delta(pidx[ind].clamp(min=0)) ** 2).sum(-1).mean() if ind.any() else phi.new_zeros(())
        return nll, prior

    @torch.no_grad()
    def base_nll(self, cand_logp, cand_mask, played_slot):
        """A0 baseline: raw Maia-2 (z=0 -> logit = base log-prob), NLL over the same candidate set."""
        logit = cand_logp.masked_fill(~cand_mask, -1e9)
        logp = F.log_softmax(logit, dim=-1)
        return -logp.gather(1, played_slot.view(-1, 1)).squeeze(1)
