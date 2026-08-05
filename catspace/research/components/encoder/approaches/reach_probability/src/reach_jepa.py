"""ReachJEPA -- a POSITIVES-ONLY joint-embedding predictive model whose prediction target is
reachability (Encoder component; Kaveh 2026-08-05).

THE OBJECT. Given position a, the predictor maps phi(a) to a REGION of embedding space -- the
futures a can lead to -- rather than to a single point. b is scored by how well it fits that
region. This is deliberately the same object the AND-OR forcing search needs for its "distribution
of where I'll end up as I go down the plies", so the search does not have to carry a second model.

WHY NO NEGATIVES. Kaveh 2026-08-05: "I don't want negatives." Nothing in lichess data is labelled
unreachable, and manufacturing negatives by splicing games at NON-matching positions produces a
label whose correctness nobody can check, on exactly the near-miss pairs that matter most. So this
trains only on observed reachable pairs, and the "not reachable" verdict is supplied afterwards by
a conformal threshold calibrated on held-out positives (see src/probability_less_than.py). That
route needs no negative class at all, which is what makes it compatible with the constraint.

WHAT THAT COSTS, AND THE GUARD. With no negative term, REPRESENTATION COLLAPSE is the primary
failure mode and it is silent: a constant encoder makes every pair fit the region perfectly and
drives the loss to zero. Two defences, neither optional:
  1. an explicit VICReg-style variance/covariance penalty on the ONLINE embeddings -- the standard
     positives-only remedy (Bardes/Ponce/LeCun), imported from the shared loss library;
  2. bootstrapped eff_rank logged as a gate on every eval (repo standing rule), so a collapse shows
     up as a number in the run log rather than as a mysteriously excellent loss.
The EMA target encoder is the third defence and the reason this is a JEPA rather than a plain
regression: gradients never flow into the target branch, so there is no path by which the model can
make the target easier to predict.

L1 ON THE PREDICTOR (Kaveh: "some regression of some sort with a strong L1 penalty"). The penalty
sits on the FIRST predictor layer, so what is sparsified is the map from phi(a) to the region. That
is what makes the interpretation pass readable: a handful of surviving input coordinates can be
correlated against piece count / ply / pawn count afterwards to ask what the model decided
irreversibility actually is. Piece count is never an input -- if it comes back, it was inferred.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


# Heteroscedastic regression standardly clamps the predicted log-variance; without it a single
# easy example can drive sigma -> 0 and the NLL to -inf, which is a divergence rather than a fit.
LOG_SIGMA_MIN, LOG_SIGMA_MAX = -6.0, 3.0


class ReachJEPA(nn.Module):
    """frozen-trunk features (B,C,8,8) -> z (B,d); predictor z_a -> Normal(mu, sigma) over z_b.

    The adapter deliberately mirrors IQEHead's (1x1 conv -> ReLU -> flatten -> linear) so that this
    head and the existing field head consume byte-identical trunk features and any difference
    between them is the objective, not the input stage.
    """

    def __init__(self, in_ch: int = 256, d: int = 64, adapter_ch: int = 32,
                 hidden: int = 256, ema_decay: float = 0.996):
        super().__init__()
        self.d = int(d)
        self.ema_decay = float(ema_decay)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, adapter_ch, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(adapter_ch * 64, d))
        # Target encoder: an EMA copy, never optimised, never receives gradient.
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        # Predictor: z_a -> (mu, log_sigma). `head_in` is the L1-penalised layer.
        self.head_in = nn.Linear(d, hidden)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 2 * d))

    def encode(self, feats):
        """(B,C,8,8) -> (B,d), the ONLINE branch (gradients flow)."""
        return self.encoder(feats)

    @torch.no_grad()
    def encode_target(self, feats):
        """(B,C,8,8) -> (B,d), the EMA TARGET branch (no gradient, by construction)."""
        return self.target_encoder(feats)

    def predict(self, z_a):
        """(B,d) -> (mu (B,d), log_sigma (B,d)): the predicted reachable region from a."""
        out = self.head(self.head_in(z_a))
        mu, log_sigma = out.chunk(2, dim=-1)
        return mu, log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)

    def score(self, z_a, z_b):
        """(B,) log-density of z_b under the region predicted from z_a.

        This is the conformal nonconformity score, up to sign: HIGHER means b looks more like a
        genuine future of a. It is a proper log-density (not a bare distance) so that the spread the
        model predicts is actually used -- a wide region should not be penalised for being wide when
        the future genuinely is uncertain.
        """
        mu, log_sigma = self.predict(z_a)
        var = torch.exp(2.0 * log_sigma)
        return -0.5 * (((z_b - mu) ** 2) / var + 2.0 * log_sigma).sum(-1)

    @torch.no_grad()
    def update_target(self):
        """EMA the online encoder into the target encoder. Call once per optimiser step."""
        m = self.ema_decay
        for pt, po in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            pt.mul_(m).add_(po.detach(), alpha=1.0 - m)
        for bt, bo in zip(self.target_encoder.buffers(), self.encoder.buffers()):
            bt.copy_(bo)

    def l1_penalty(self):
        """L1 on the predictor's input layer -- reported for monitoring. NOT the mechanism that
        produces sparsity; see prox_l1."""
        return self.head_in.weight.abs().mean()

    @torch.no_grad()
    def prox_l1(self, lam: float):
        """Proximal soft-threshold on the predictor's input layer: W <- sign(W) * relu(|W| - lam).

        Adding an L1 term to the loss and running Adam does NOT sparsify -- measured here, w_l1=0.5
        via the subgradient route left 64/64 input coordinates alive. Two reasons: Adam's per-
        parameter preconditioning rescales the constant L1 subgradient differently for every weight,
        so nothing lands exactly on zero; and a penalty on the MEAN of 16k weights supplies a per-
        weight gradient far too small to cross the origin. The proximal operator is the standard fix
        (ISTA) -- applied AFTER the optimiser step, it sets genuinely small weights to exact zero, so
        'how many coordinates survive' becomes a real count rather than a thresholding convention."""
        w = self.head_in.weight
        w.copy_(torch.sign(w) * torch.clamp(w.abs() - lam, min=0.0))

    def input_support(self, tol: float = 0.0):
        """(d,) bool: which input coordinates the predictor still reads (any non-zero weight)."""
        return (self.head_in.weight.abs() > tol).any(dim=0)
