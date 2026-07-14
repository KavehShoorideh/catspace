"""
two_field.py — Stage-3 runtime: two perspectives over one slow embedding.

One ω-conditioned slow net; two goal-fields (my mate / their mate); two fast
MemoryFields (my evidence / theirs). Per candidate move the planner scores:

    score(s') = reach_mine(s')                      (slow field, my goal)
              - beta  * reach_theirs(s')            (deny their progress)
              + gamma * pvar_theirs(s')             (steer into THEIR muddy regions)
              - gamma_self * pvar_mine(s')          (stay out of MY muddy regions)

with the fast fields re-pricing reach where they have support (effective-distance
override, JOURNAL 2026-07-14): if my memory knows this region (support >= s_min),
use plies + lam*(-ln P̂_mine) as my distance instead of the slow d. p_var gating:
where MY p_var is high we don't trust either estimate -- the caller should search
deeper there (sharpness -> allocate compute; wired by the search layer later).

Unit-testable without chess: score_components() takes raw numbers.
"""
from __future__ import annotations

import numpy as np


def effective_distance(d_slow: float, mem: dict | None, lam: float = 8.0,
                       s_min: float = 0.9, floor: float = 1e-3) -> tuple[float, float]:
    """Blend slow distance with fast-field evidence. Returns (d_eff, p_var).
    mem = MemoryField.query() dict or None."""
    if mem is None or mem["support"] < s_min:
        return d_slow, 0.0
    p = max(mem["p_hat"], floor)
    plies = mem["plies"] if mem["plies"] is not None else d_slow * 50.0
    return (plies + lam * (-np.log(p))) / 50.0, mem["p_var"]


def score_components(d_mine: float, d_theirs: float, pvar_mine: float,
                     pvar_theirs: float, beta: float = 0.5, gamma: float = 0.3,
                     gamma_self: float = 0.3) -> float:
    """The two-perspective move score (higher = better for me)."""
    return (-d_mine + beta * d_theirs
            + gamma * pvar_theirs - gamma_self * pvar_mine)


class TwoFieldPolicy:
    """Wraps an FBSearchPolicy-style base: rescore its candidate children with the
    two-perspective objective. Start-simple: 1-ply rescoring over legal moves."""

    def __init__(self, fb, z_mine, z_theirs, mem_mine=None, mem_theirs=None,
                 device="cpu", beta=0.5, gamma=0.3, gamma_self=0.3, lam=8.0,
                 elo=1800, clock=float("nan")):
        import torch
        from catspace.nn.features import omega_ids
        self.fb = fb.to(device).eval()
        self.torch = torch
        as_t = lambda z: (z.to(device).float() if torch.is_tensor(z)
                          else torch.as_tensor(np.asarray(z), dtype=torch.float32, device=device))
        self.zm, self.zt = as_t(z_mine), as_t(z_theirs)
        self.mm, self.mt = mem_mine, mem_theirs
        self.device = device
        self.beta, self.gamma, self.gamma_self, self.lam = beta, gamma, gamma_self, lam
        self._omega = omega_ids(np.array([elo]), np.array([elo]), np.array([clock]))[0]

    def _embed(self, boards):
        from catspace.data.encode import encode_meta, encode_packed
        from catspace.nn.features import feature_planes
        packed = np.stack([encode_packed(b) for b in boards])
        meta = np.stack([encode_meta(b) for b in boards])
        with self.torch.no_grad():
            pl = self.torch.from_numpy(feature_planes(packed, meta)).to(self.device)
            om = self.torch.from_numpy(np.tile(self._omega, (len(boards), 1))).to(self.device)
            f = self.fb.embed_F(pl, om)
            dm = self.fb.distance_matrix(f, self.zm[None, :])[:, 0].cpu().numpy()
            dt = self.fb.distance_matrix(f, self.zt[None, :])[:, 0].cpu().numpy()
        return f.cpu().numpy(), dm, dt

    def move(self, board, rng):
        moves = list(board.legal_moves)
        succ = []
        for m in moves:
            b2 = board.copy(stack=False); b2.push(m)
            if b2.is_checkmate():
                return m                                     # immediate mate: take it
            succ.append(b2)
        F, dm, dt = self._embed(succ)
        best, best_s = None, -np.inf
        for i, m in enumerate(moves):
            qm = self.mm.query(F[i]) if self.mm else None
            qt = self.mt.query(F[i]) if self.mt else None
            dme, pvm = effective_distance(dm[i], qm, self.lam)
            dte, pvt = effective_distance(dt[i], qt, self.lam)
            s = score_components(dme, dte, pvm, pvt, self.beta, self.gamma, self.gamma_self)
            if s > best_s:
                best_s, best = s, m
        return best
