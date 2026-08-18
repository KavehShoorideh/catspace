#!/usr/bin/env python
"""test_concept_quasimetric.py -- the geometry's axioms and the QRL machinery, before any run.

A quasimetric that quietly loses an axiom is worse than no quasimetric: every downstream claim
about "distance" would be false in a way no accuracy number would reveal. These run on CPU in
seconds and need no corpus, so there is no excuse for launching without them.

    .venv/bin/python -m ...test_concept_quasimetric
"""
from __future__ import annotations

import numpy as np
import torch

from catspace.research.tools.training_infra.losses import (
    qrl_constraint_violation, qrl_maximise, qrl_phi)
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE
from catspace.research.components.encoder.approaches.reach_probability.experiments \
    .concept_quasimetric import ConceptQuasimetric


def main():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        cond = bool(cond); ok &= cond
        print(f"  {'OK  ' if cond else 'FAIL'} {name}{('  ' + extra) if extra else ''}")

    torch.manual_seed(0)
    d, C = 128, 8
    iqe = IQE(d, components=C)
    u = torch.randn(64, d); v = torch.randn(64, d); w = torch.randn(64, d)

    # ---- the axioms, tested rather than trusted -------------------------------------
    chk("d(x,x) == 0 exactly", float(iqe(u, u).abs().max()) < 1e-6,
        f"max {float(iqe(u,u).abs().max()):.2e}")
    duv, dvu = iqe(u, v), iqe(v, u)
    chk("ASYMMETRIC: d(u,v) != d(v,u) on random pairs",
        float((duv - dvu).abs().mean()) > 1e-3,
        f"mean |gap| {float((duv-dvu).abs().mean()):.3f}")
    duw, dvw = iqe(u, w), iqe(v, w)
    slack = (duv + dvw) - duw
    chk("TRIANGLE INEQUALITY d(u,w) <= d(u,v) + d(v,w)",
        float(slack.min()) >= -1e-5, f"worst slack {float(slack.min()):+.2e}")
    chk("non-negative", float(iqe(u, v).min()) >= 0.0)

    # DIRECTION SEMANTICS: d(u->v) must be ~0 when u already dominates v ("already reached")
    # and large when v exceeds u ("must climb"). A flipped convention makes the whole geometry
    # run backwards against time's arrow, and has been shipped by accident before.
    lo = torch.zeros(4, d); hi = torch.ones(4, d) * 3.0
    chk("direction: d(high -> low) ~ 0 (already reached)",
        float(iqe(hi, lo).abs().max()) < 1e-6)
    chk("direction: d(low -> high) large (must climb)", float(iqe(lo, hi).min()) > 1.0,
        f"{float(iqe(lo,hi).mean()):.2f}")

    # ---- QRL halves ------------------------------------------------------------------
    # phi's gradient is sigmoid(beta*(knee-d)), so with knee=60 and beta=0.03 the switch-off is
    # DELIBERATELY gradual: 0.86 at d=0, 0.29 at d=90, 0.063 at d=150. The right test is that it
    # decays monotonically and is well down at the TOP of our horizon (chess games run 40-150
    # plies), not that it has collapsed just past the knee -- a soft knee is the point.
    def gmag(dv, n=100):
        """Per-element |d(loss)/d(distance)|. qrl_maximise takes a mean, so the raw gradient is
        divided by n -- undo that, or the numbers look 100x too small and mean nothing."""
        t = torch.full((n,), float(dv), requires_grad=True)
        return float(torch.autograd.grad(qrl_maximise(t, knee=60.0), t)[0].mean()) * n

    gs = [gmag(x) for x in (0.5, 30, 60, 90, 150)]
    chk("qrl_maximise PUSHES UP (gradient on d is negative, so descending raises d)",
        all(x < 0 for x in gs), f"{gs[0]:+.3f} at d=0.5")
    mag = [abs(x) for x in gs]
    chk("phi's push decays MONOTONICALLY with distance",
        all(mag[i] > mag[i + 1] for i in range(len(mag) - 1)),
        " > ".join(f"{x:.3f}" for x in mag))
    chk("phi is well down at the top of our horizon (d=150)",
        mag[-1] < mag[0] / 10, f"{mag[-1]:.3f} vs {mag[0]:.3f} ({mag[0]/mag[-1]:.1f}x)")
    chk("phi is exactly half-strength AT the knee (sigmoid(0) = 0.5)",
        abs(mag[2] - 0.5) < 0.02, f"{mag[2]:.3f}")

    v_sat = qrl_constraint_violation(torch.full((100,), 0.9), ceiling=1.0, eps=0.25)
    v_bad = qrl_constraint_violation(torch.full((100,), 3.0), ceiling=1.0, eps=0.25)
    chk("constraint satisfied -> violation <= 0", float(v_sat) <= 0, f"{float(v_sat):+.4f}")
    chk("constraint broken -> violation > 0", float(v_bad) > 0, f"{float(v_bad):+.3f}")
    # dual ascent must raise lambda exactly while the constraint is broken, and relax otherwise
    lam = torch.tensor(1.0)
    lam_up = (lam + 0.01 * v_bad.detach()).clamp(min=0.0)
    lam_dn = (lam + 0.01 * v_sat.detach()).clamp(min=0.0)
    chk("dual ascent raises lambda under violation and relaxes it when satisfied",
        float(lam_up) > 1.0 > float(lam_dn), f"{float(lam_dn):.3f} < 1 < {float(lam_up):.3f}")

    # ---- the state gather must reconstruct what the encoder produced ------------------
    H, Cg, dc, SC, df = 4, 8, 6, 16, 6
    g_book = torch.randn(H, Cg, dc); s_book = torch.randn(SC, df)
    g = torch.randint(0, Cg, (10, H)); s = torch.randint(0, SC, (10, 64))
    zg = g_book[torch.arange(H)[None, :], g].reshape(10, -1)
    zs = s_book[s].reshape(10, -1)
    zq = torch.cat([zg, zs], -1)
    manual = torch.stack([torch.cat([torch.cat([g_book[h, g[i, h]] for h in range(H)]),
                                     torch.cat([s_book[s[i, j]] for j in range(64)])])
                          for i in range(10)])
    chk("codebook gather reconstructs the concept state exactly",
        torch.allclose(zq, manual, atol=1e-6),
        f"max |diff| {float((zq-manual).abs().max()):.2e}")
    chk("state width is heads*d_code + 64*d_fine",
        zq.shape[1] == H * dc + 64 * df, f"{zq.shape[1]}")

    # ---- the model end to end ---------------------------------------------------------
    net = ConceptQuasimetric(d_state=zq.shape[1], d_iqe=64, components=8, hidden=32)
    za, zb = torch.randn(8, zq.shape[1]), torch.randn(8, zq.shape[1])
    dd = net.dist(za, zb)
    chk("ConceptQuasimetric.dist -> (B,)", dd.shape == (8,))
    chk("d(z,z) == 0 through the projection too",
        float(net.dist(za, za).abs().max()) < 1e-6)
    gr = torch.autograd.grad(dd.sum(), [p for p in net.parameters()], allow_unused=True)
    chk("gradient reaches every parameter",
        all(x is not None and torch.isfinite(x).all() for x in gr))

    print("\nALL CONCEPT-QUASIMETRIC TESTS PASSED" if ok else "\nTESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
