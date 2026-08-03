"""catspace/style/recover.py -- CONVEX recovery of a player's style residual z from their moves.

Because logit_P(m|s) = l0(m) + z_P . U(s,m) is LINEAR in z given the frozen U (catspace/style/model.py),
the penalised move-NLL is CONVEX in the per-player deviation Delta (z = mu(Elo) + Delta). We solve the
MAP fit with LBFGS from cached features -- no Maia / field calls -- in milliseconds, and (optionally)
return the Laplace posterior covariance H^-1 (H = Hessian at the optimum): the M2c in-game-tightening
object (cold start Delta=0 => z=mu(Elo) = the prior; each observed move tightens the posterior).

The SAME code recovers z for train players (re-inferred, no procedural asymmetry), held-out players, and
unseen / mid-game opponents (Matilda's post-hoc property).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _precompute_U(model, feats, device):
    """U(s,m) for a player's positions: (n,K,d_z), plus the fixed mu(Elo) means (n,d_z)."""
    phi = feats["phi"].to(device); cand_idx = feats["cand_idx"].to(device)
    cand_logp = feats["cand_logp"].to(device); rank = feats["rank"].to(device)
    with torch.no_grad():
        U = model.U_of(phi, cand_idx, cand_logp, rank)                      # (n,K,d_z)
        mu = model.mu_of(feats["elo"].to(device).float())                  # (n,d_z), 0 by default
    return U, mu, cand_logp, feats["cand_mask"].to(device), feats["played_slot"].to(device)


def recover_delta(model, feats, lam: float = 1.0, steps: int = 60, laplace: bool = False, device="cpu",
                  weights=None):
    """Return (delta (d_z,), optional H^-1 (d_z,d_z)). feats: dict of support-position tensors
    (phi, cand_idx, cand_logp, cand_mask, rank, played_slot, elo). `weights` (n,): optional per-move
    weights (recency weighting for the online M2c filter -- recent moves count more); None = uniform."""
    model.eval()
    U, mu, cand_logp, cand_mask, played_slot = _precompute_U(model, feats, device)
    delta = torch.zeros(model.d_z, device=device, requires_grad=True)
    w = None if weights is None else torch.as_tensor(weights, dtype=torch.float32, device=device)

    def loss_fn(d):
        z = mu + d                                                          # (n,d_z), mu fixed
        style = (U * z.unsqueeze(1)).sum(-1)                               # (n,K)
        logit = (cand_logp + style).masked_fill(~cand_mask, -1e9)
        nll = -F.log_softmax(logit, -1).gather(1, played_slot.view(-1, 1)).squeeze(1)
        nll = nll if w is None else nll * w
        return nll.sum() + lam * (d ** 2).sum()

    opt = torch.optim.LBFGS([delta], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(); l = loss_fn(delta); l.backward(); return l
    opt.step(closure)
    delta = delta.detach()

    Hinv = None
    if laplace:
        H = torch.autograd.functional.hessian(loss_fn, delta.clone().requires_grad_(True))
        Hinv = torch.linalg.inv(H + 1e-4 * torch.eye(model.d_z, device=device))
    return delta, Hinv


@torch.no_grad()
def score_nll(model, feats, delta, device="cpu"):
    """Per-position NLL of the played move under z = mu(Elo)+delta (delta=0 -> prior-only)."""
    U, mu, cand_logp, cand_mask, played_slot = _precompute_U(model, feats, device)
    z = mu + delta.to(device)
    style = (U * z.unsqueeze(1)).sum(-1)
    logit = (cand_logp + style).masked_fill(~cand_mask, -1e9)
    return (-F.log_softmax(logit, -1).gather(1, played_slot.view(-1, 1)).squeeze(1)).cpu()


@torch.no_grad()
def base_nll(model, feats, device="cpu"):
    """A0: raw Maia-2 (z=0) NLL over the same candidate set."""
    cand_logp = feats["cand_logp"].to(device); cand_mask = feats["cand_mask"].to(device)
    played_slot = feats["played_slot"].to(device)
    return model.base_nll(cand_logp, cand_mask, played_slot).cpu()
