#!/usr/bin/env python
"""pointer_policy.py -- the RL commitment head (Kaveh 2026-08-12): point at a surfaced
alert token (pursue an opportunity / deny a worry) or HOLD the current plan, AND pick a
search budget -- rewarded for winning with the least effort:

    R = outcome - lambda_effort * sum_t evals(b_t)

Search budgets include b=0 (PREMOVE: trust the certificate, no verification) -- under the
effort penalty the policy learns to premove exactly where the certificate is safe and to
spend verification exactly where an alert spiked (value-of-information, learned).

Gradient boundary (docs/SUBGOALFORMER.md): this head trains ONLY itself. Certificates,
GeoAttention, rulers and trunk arrive DETACHED -- a policy must never train its own
measuring instrument, and a policy-shaped p-hat would be propaganda.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

BUDGETS = (0.0, 0.4, 1.5, 4.0)          # seconds of coherent search; 0 = premove
ALERT_F = 5 + 4                          # token feats + [kind, side, d_p, salience]


class PointerPolicy(nn.Module):
    def __init__(self, d=64, k_max=16):
        super().__init__()
        self.k_max = k_max
        self.enc = nn.Sequential(nn.Linear(ALERT_F, d), nn.GELU(), nn.Linear(d, d))
        self.plan = nn.Sequential(nn.Linear(3, d), nn.GELU())   # [p_hat, max_worry, premove]
        self.point = nn.Linear(d, 1)                            # per-alert pointer logit
        self.hold = nn.Parameter(torch.zeros(1))                # HOLD-the-plan logit
        self.budget = nn.Linear(d, len(BUDGETS))

    def _alert_feats(self, alerts):
        rows = []
        for a in alerts[:self.k_max]:
            rows.append(np.concatenate([np.asarray(a.feats, np.float32),
                                        [1.0 if a.kind == "opportunity" else 0.0,
                                         float(a.side), a.d_p, a.salience]]))
        if not rows:
            rows = [np.zeros(ALERT_F, np.float32)]
        return torch.as_tensor(np.stack(rows), dtype=torch.float32)

    def forward(self, cert, alerts):
        """-> (pointer_logits (K+1,): [alerts..., HOLD], budget_logits (4,))"""
        A = self.enc(self._alert_feats(alerts))                        # (K,d)
        ctx = self.plan(torch.tensor([cert.p_hat,
                                      float(cert.worry.max(initial=0.0)),
                                      1.0 if cert.premove_safe() else 0.0]))
        h = A + ctx
        ptr = torch.cat([self.point(h).squeeze(-1), self.hold])
        bud = self.budget(ctx + A.mean(0))
        return ptr, bud

    def act(self, cert, alerts, greedy=False):
        ptr, bud = self.forward(cert, alerts)
        pd, bd = torch.distributions.Categorical(logits=ptr), \
            torch.distributions.Categorical(logits=bud)
        a = ptr.argmax() if greedy else pd.sample()
        b = bud.argmax() if greedy else bd.sample()
        logp = pd.log_prob(a) + bd.log_prob(b)
        return int(a), BUDGETS[int(b)], logp


def reinforce_loss(logps, rewards, baseline=0.0):
    """REINFORCE with a scalar baseline. rewards already include -lambda*evals; per-game."""
    lp = torch.stack(list(logps))
    adv = torch.as_tensor(rewards, dtype=torch.float32) - baseline
    return -(lp * adv).mean()


def effort_reward(outcome, total_evals, lam=2e-6):
    """R = outcome - lambda*evals. Default lam: a FULL deep game (~100k evals) costs 0.2 --
    less than win-vs-draw (0.5), so winning strictly dominates; effort tiebreaks."""
    return float(outcome) - lam * float(total_evals)


def _tests():
    torch.manual_seed(0)
    from catspace.research.components.planner.approaches.quasimetric_nav.subgoal_former import (
        Alert, Certificate)
    import numpy as np
    cert = Certificate(hc=np.array([[0, 1], [2, 3]]), sides=np.array([0, 1]), committed=0,
                       p_hat=0.99, p_all=np.array([0.99, 0.2], np.float32),
                       worry=np.array([0.0, 0.01], np.float32),
                       attn=np.array([0.5, 0.5], np.float32))
    alerts = [Alert(hc=(2, 3), side=1, kind="worry", salience=0.02, d_p=-0.1,
                    feats=np.ones(5, np.float32)),
              Alert(hc=(4, 5), side=1, kind="opportunity", salience=0.4, d_p=0.3,
                    feats=np.zeros(5, np.float32))]
    pol = PointerPolicy()
    ptr, bud = pol(cert, alerts)
    ok = ptr.shape == (3,) and bud.shape == (len(BUDGETS),)
    a, b, logp = pol.act(cert, alerts)
    ok &= 0 <= a <= 2 and b in BUDGETS and logp.requires_grad
    # reward: winning slow beats losing fast; among wins, cheap beats expensive
    r_cheap = effort_reward(1.0, 2_000)
    r_exp = effort_reward(1.0, 300_000)
    r_loss = effort_reward(0.0, 0)
    ok &= r_cheap > r_exp > r_loss
    # trainability: pushing reward for pointing at the high-salience opportunity
    opt = torch.optim.Adam(pol.parameters(), lr=5e-3)
    for _ in range(200):
        logps, rs = [], []
        for _e in range(8):
            a, b, lp = pol.act(cert, alerts)
            logps.append(lp); rs.append(1.0 if a == 1 else 0.0)
        opt.zero_grad(); reinforce_loss(logps, rs, baseline=0.5).backward(); opt.step()
    picks = sum(pol.act(cert, alerts, greedy=True)[0] == 1 for _ in range(4))
    ok &= picks == 4
    print(f"[pointer] shapes OK | reward order cheapWin>expWin>loss: "
          f"{r_cheap:.3f}>{r_exp:.3f}>{r_loss:.3f} | learned to point at the "
          f"opportunity: {picks}/4 greedy")
    print("ALL POINTER-POLICY TESTS PASSED" if ok else "TESTS FAILED")


if __name__ == "__main__":
    _tests()
