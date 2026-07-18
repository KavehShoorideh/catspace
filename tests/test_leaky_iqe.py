"""tests/test_leaky_iqe.py — the leaky (relaxed-relu) IQE escape hatch and the
PID spike guards (Kaveh 2026-07-18, after the lam-spike implosion @8000).

The exact-IQE axioms stay pinned by tests/test_invariants.py (leak_beta=0 is
the untouched default path); these tests cover what the leak is FOR."""
from __future__ import annotations

import torch

from catspace.nn.iqe import IQE


def _dead_zone(n=8, d=64, gap=5.0, seed=0):
    """All-V-below-all-U coordinatewise: the ordering-collapse dead zone."""
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, d, generator=g)
    v = u - gap                                   # strictly dominated
    return u, v.requires_grad_(True)


def test_exact_iqe_dead_zone_is_flat():
    """The documented failure: d==0 AND zero gradient (why it is absorbing)."""
    iqe = IQE(64, components=8)
    u, v = _dead_zone()
    d = iqe(u, v)
    assert torch.all(d == 0.0)
    d.sum().backward()
    assert torch.all(v.grad == 0.0)


def test_leaky_iqe_boundary_zone_has_live_gradient():
    """leak_beta>0 near the collapse SURFACE (gap ~0.5 at beta=10): small but
    real distance and a usable gradient — the surface becomes repulsive, so
    trajectories cannot settle into the zone. (Deep-zone escape is NOT
    claimed: see the next test.)"""
    iqe = IQE(64, components=8, leak_beta=10.0)
    u, v = _dead_zone(gap=0.5)
    d = iqe(u, v)
    assert torch.all(d > 0.0)
    assert torch.all(d < 0.2)                     # miniscule, not a real distance
    d.sum().backward()
    assert v.grad.abs().mean() > 1e-5             # usable, not just nonzero


def test_leaky_iqe_deep_zone_gradient_nonzero_but_impractical():
    """Deep in the zone (gap 5) the forward is float32-absorbed to exactly 0
    and the gradient is ~e^{-beta*gap} — alive for autograd, useless for
    escape. Honest scope: the leak PREVENTS settling (boundary repulsion);
    the PID guards must prevent deep shoves. Both are required."""
    iqe = IQE(64, components=8, leak_beta=10.0)
    u, v = _dead_zone(gap=5.0)
    d = iqe(u, v)
    d.sum().backward()
    assert v.grad.abs().max() > 0.0               # strictly alive
    assert v.grad.abs().max() < 1e-6              # ... but exponentially small


def test_leaky_iqe_converges_to_exact():
    """beta -> inf recovers the paper object (relaxation, not replacement)."""
    g = torch.Generator().manual_seed(1)
    u, v = torch.randn(16, 64, generator=g), torch.randn(16, 64, generator=g)
    exact = IQE(64, components=8)
    leaky = IQE(64, components=8, leak_beta=200.0)
    leaky.load_state_dict(exact.state_dict(), strict=True)
    assert torch.allclose(exact(u, v), leaky(u, v), atol=0.05)
    assert torch.allclose(exact.pairwise(u, v), leaky.pairwise(u, v), atol=0.05)


def test_pid_lambda_cap_and_eclip():
    """lam never exceeds lam_max (with anti-windup on I), and pid_eclip bounds
    the per-step response regardless of the violation size."""
    from catspace.nn.fb import TorchFB
    fb = TorchFB(d=64, channels=8, blocks=1, enc_out=32, dh=64,
                 iqe=True, iqe_components=8, iqe_embed_scale=1.0)
    planes = torch.randn(6, 20, 8, 8)
    omega = torch.randint(0, 4, (6, 3))
    valid = torch.ones(6, dtype=torch.bool)
    for _ in range(8):                            # let I integrate a while
        loss, st = fb.qrl_loss(planes, omega, planes + 0.1, planes.flip(0), valid,
                               push_offset=15.0, use_pid=True, two_sided=True,
                               pid_kp=5.0, pid_ki=1.0, pid_kd=5.0,   # hot gains
                               pid_eclip=3.0, lam_max=2.0)
    assert float(st["lam"]) <= 2.0 + 1e-6
    assert float(fb.qrl_pid_I) <= 2.0 + 1e-6      # anti-windup held
