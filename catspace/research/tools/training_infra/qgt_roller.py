"""QGT dynamics: linear Koopman carrier roller with spectral control.

Spec (JOURNAL 2026-08-28, amended section 5): c' = K c + B m on a CONTINUOUS carrier;
concepts are quantized observables read AFTER rolling, never rolled themselves. Spectral
clamp rho<=1 + penalty w=0.2 (gko1b's blowup at rho 1.041 over 6-fold composition is the
scar); never re-lift; trained through composition on chains.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class KoopmanRoller(nn.Module):
    def __init__(self, d_carrier: int, d_move: int, w_spec: float = 0.2):
        super().__init__()
        self.K = nn.Parameter(torch.eye(d_carrier) +
                              0.01 * torch.randn(d_carrier, d_carrier))
        self.B = nn.Linear(d_move, d_carrier, bias=False)
        self.w_spec = w_spec
        self.register_buffer("_pv", torch.randn(d_carrier))

    def rho(self, iters: int = 8) -> torch.Tensor:
        """Power-iteration estimate of the spectral radius of K."""
        v = self._pv
        with torch.no_grad():
            for _ in range(iters):
                v = self.K @ v
                v = v / v.norm().clamp(min=1e-12)
            self._pv.copy_(v)
        Kv = self.K @ v
        return Kv.norm() / v.norm().clamp(min=1e-12)

    @torch.no_grad()
    def clamp_(self):
        """Scale K back inside the unit spectral radius. Call after optimizer steps."""
        r = float(self.rho())
        if r > 1.0:
            self.K.mul_(1.0 / r)

    def spectral_penalty(self) -> torch.Tensor:
        return self.w_spec * torch.relu(self.rho() - 1.0) ** 2

    def step(self, c: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return c @ self.K.T + self.B(m)

    def roll(self, c0: torch.Tensor, moves: list[torch.Tensor]) -> list[torch.Tensor]:
        """Compose k steps, NEVER re-lifting. Returns the carrier at each horizon."""
        out, c = [], c0
        for m in moves:
            c = self.step(c, m)
            out.append(c)
        return out


def _test():
    torch.manual_seed(0)
    D, M, B_ = 64, 16, 8
    r = KoopmanRoller(D, M)
    with torch.no_grad():
        r.K.mul_(1.15)                      # push unstable on purpose
    assert float(r.rho()) > 1.05
    r.clamp_()
    assert float(r.rho()) <= 1.001, f"clamp failed: {float(r.rho())}"
    c0 = torch.randn(B_, D)
    mv = [torch.randn(B_, M) for _ in range(6)]
    outs = r.roll(c0, mv)
    assert len(outs) == 6
    n0, n6 = float(c0.norm()), float(outs[-1].norm())
    assert n6 < 20 * n0, f"6-fold composition explodes: {n0} -> {n6}"
    outs[-1].sum().backward()
    assert r.K.grad is not None and float(r.K.grad.abs().sum()) > 0
    assert r.B.weight.grad is not None
    p = r.spectral_penalty()
    assert float(p) >= 0 and torch.isfinite(p)
    print(f"  rho clamp {float(r.rho()):.4f}<=1; 6-fold roll stable ({n0:.1f}->{n6:.1f}); "
          f"gradient reaches K and B through the composition; penalty finite")
    print("  QGT ROLLER TESTS PASS")


if __name__ == "__main__":
    _test()
