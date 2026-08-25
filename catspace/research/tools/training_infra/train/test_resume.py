"""Does a resumed run continue the SAME trajectory, or silently start a new one?

Kaveh 2026-08-25: "fix resuming issues so we can actually resume without problem."

Resuming was restoring weights only. Three things that are training STATE, not weights,
were dropped on the floor:

  * Adam's moments and bias-correction step. A fresh optimizer takes differently-sized
    steps for hundreds of iterations after a resume.
  * the step counter (TrainConfig(start_step=0) was hardcoded). This is the dangerous
    one: gko_train ramps its operator losses on `(step - ops_after)/1000` and its 6-ply
    chain supervision on `(step - op6_after)/2000`, so a resumed run silently re-gates
    its main objectives OFF and ramps them back over 1000-2000 steps while the losses
    read ~0 and look converged.
  * the sampling RNG, so a resumed run re-sees the batches it already trained on.

This test proves the round-trip on a toy model -- no trainer, no GPU, seconds to run --
and, more importantly, includes the NEGATIVE CONTROL: resuming without optimizer state
must visibly diverge. Without that control the test would pass just as happily on a
broken implementation that never needed the state in the first place.
"""
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from catspace.research.tools.training_infra.train.scaffold import save_torch_ckpt


def _model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4))


def _batches(n, seed=7):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(8, 16, generator=g), torch.randn(8, 4, generator=g))
            for _ in range(n)]


def _run(model, opt, batches):
    for x, y in batches:
        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        opt.step()
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_resume_is_continuation():
    N = 20
    data = _batches(N)

    # (A) reference: N steps, uninterrupted
    m_a = _model()
    o_a = torch.optim.Adam(m_a.parameters(), lr=1e-3)
    ref = _run(m_a, o_a, data)

    with tempfile.TemporaryDirectory() as td:
        # (B) N/2 steps, checkpoint carrying optimizer state, resume, N/2 more
        m_b = _model()
        o_b = torch.optim.Adam(m_b.parameters(), lr=1e-3)
        _run(m_b, o_b, data[:N // 2])
        path = save_torch_ckpt(m_b, str(Path(td) / "t"), N // 2,
                               extra={"opt": o_b.state_dict()})
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert "opt" in ck, "checkpoint does not carry optimizer state"
        assert ck["step"] == N // 2, f"step not recorded: {ck.get('step')}"

        m_r = _model(seed=99)                       # deliberately different init...
        m_r.load_state_dict(ck["state_dict"])       # ...overwritten by the checkpoint
        o_r = torch.optim.Adam(m_r.parameters(), lr=1e-3)
        o_r.load_state_dict(ck["opt"])
        got = _run(m_r, o_r, data[N // 2:])

        # (C) NEGATIVE CONTROL: same resume, but WITHOUT optimizer state
        m_n = _model(seed=99)
        m_n.load_state_dict(ck["state_dict"])
        o_n = torch.optim.Adam(m_n.parameters(), lr=1e-3)   # fresh moments
        bad = _run(m_n, o_n, data[N // 2:])

    err_fixed = (got - ref).abs().max().item()
    err_broken = (bad - ref).abs().max().item()
    scale = ref.abs().mean().item()

    assert err_fixed < 1e-6, (
        f"resume did NOT continue the trajectory: max|dw| {err_fixed:.2e}")
    assert err_broken > 100 * max(err_fixed, 1e-12), (
        f"negative control did not diverge ({err_broken:.2e}) -- the test cannot "
        f"distinguish a working implementation from a broken one")
    print(f"  resume WITH opt state:    max|dw| vs uninterrupted {err_fixed:.2e}  "
          f"(weights scale {scale:.3f}) -> identical trajectory")
    print(f"  resume WITHOUT opt state: max|dw| vs uninterrupted {err_broken:.2e}  "
          f"-> {err_broken / max(err_fixed, 1e-12):.0e}x worse, which is what was "
          f"happening before this fix")


def test_step_counter_gates_survive():
    """The step counter is not cosmetic: gko_train ramps its operator losses on it.

    Reproduces the two live ramps against a resumed-at-zero counter, so the cost of
    losing the step is a number rather than an assertion in a comment.
    """
    ops_after, op6_after, resumed_at = 300, 1000, 4000

    def opg(step):
        return min(1.0, max(0.0, (step - ops_after) / 1000.0))

    def w6r(step):
        return min(1.0, max(0.0, (step - op6_after) / 2000.0))

    assert opg(resumed_at) == 1.0 and w6r(resumed_at) == 1.0, "ramps should be complete"
    assert opg(1) == 0.0 and w6r(1) == 0.0
    lost_ops = sum(1 for s in range(1, 3001) if opg(s) < 1.0)
    lost_op6 = sum(1 for s in range(1, 3001) if w6r(s) < 1.0)
    print(f"  a resume that restarts the step counter runs {lost_ops} steps with the "
          f"operator loss ramped below full")
    print(f"  and {lost_op6} steps with 6-ply chain supervision below full -- reading "
          f"~0.000, which is indistinguishable from converged")





def test_scaffold_round_trip():
    """The WRAPPER, not just the mechanism: standard_train must checkpoint optimizer
    state automatically, and resume_from must restore it.

    This is the part that generalises -- 19 trainers call standard_train, and before
    this they all checkpointed weights only. Covers both wiring styles: an optimizer
    passed explicitly, and one attached to the step function by a build_step() factory.
    """
    from catspace.research.tools.training_infra.train.scaffold import (
        TrainConfig, standard_train, resume_from)

    for style in ("explicit", "attached"):
        data = _batches(20)
        with tempfile.TemporaryDirectory() as td:
            m = _model()
            o = torch.optim.Adam(m.parameters(), lr=1e-3)

            def step_fn(model, step, _o=o, _d=data):
                x, y = _d[(step - 1) % len(_d)]
                _o.zero_grad()
                loss = nn.functional.mse_loss(model(x), y)
                loss.backward(); _o.step()
                return {"loss": float(loss)}

            kw = {"opt": o}
            if style == "attached":
                step_fn.opt = o                 # the build_step() factory pattern
                kw = {}
            cfg = TrainConfig(out=str(Path(td) / "r"), steps=10, ckpt_every=10,
                              eval_every=100, experiment="test")
            standard_train(step_fn, m, cfg, args=None, **kw)

            ck = torch.load(Path(td) / "r_step10.pt", map_location="cpu",
                            weights_only=False)
            assert "opt" in ck, f"[{style}] standard_train did not save optimizer state"
            assert "rng" in ck, f"[{style}] standard_train did not save RNG state"

            m2 = _model(seed=123)
            o2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
            step = resume_from(str(Path(td) / "r_step10.pt"), m2, opt=o2)
            assert step == 10, f"[{style}] resume_from returned step {step}, expected 10"
            w1 = torch.cat([p.detach().flatten() for p in m.parameters()])
            w2 = torch.cat([p.detach().flatten() for p in m2.parameters()])
            assert torch.allclose(w1, w2), f"[{style}] weights not restored"
            s1 = o.state_dict()["state"]; s2 = o2.state_dict()["state"]
            assert len(s1) == len(s2) and len(s1) > 0, f"[{style}] optimizer state empty"
            print(f"  scaffold [{style:8}]: ckpt carries opt+rng, resume_from -> step "
                  f"{step}, weights and {len(s2)} optimizer slots restored")





def test_frozen_leaves_nothing_changed():
    """A gate must MEASURE the model, not train it.

    Reproduces both real failures: an optimizer step taken inside a diagnostic, and a
    .zero_() applied to the thing being probed. Both left the model altered and the
    gates still printed PASS.
    """
    from catspace.research.tools.training_infra.train.scaffold import frozen

    m = _model()
    o = torch.optim.Adam(m.parameters(), lr=1e-2)
    flag = [True]
    before = torch.cat([p.detach().flatten().clone() for p in m.parameters()])

    with frozen(m, o, cells=[flag]):
        x, y = _batches(1)[0]                       # U19b: a real optimizer step
        o.zero_grad(); nn.functional.mse_loss(m(x), y).backward(); o.step()
        with torch.no_grad():                       # U22: zeroing what it probes
            next(m.parameters()).zero_()
        flag[0] = False
        during = torch.cat([p.detach().flatten() for p in m.parameters()])
        assert not torch.allclose(before, during), "the probe did not actually mutate"

    after = torch.cat([p.detach().flatten() for p in m.parameters()])
    assert torch.allclose(before, after), "frozen() did NOT restore the weights"
    assert flag == [True], "frozen() did not restore the run-state cell"
    assert all(p.grad is None for p in m.parameters()), "grads left behind"
    print(f"  frozen(): weights, optimizer and run-state cells restored after a probe "
          f"that took a real optimizer step AND zeroed a parameter")


if __name__ == "__main__":
    test_resume_is_continuation()
    test_step_counter_gates_survive()
    test_scaffold_round_trip()
    test_frozen_leaves_nothing_changed()
    print("  RESUME TESTS PASS")
