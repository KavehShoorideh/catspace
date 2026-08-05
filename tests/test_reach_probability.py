"""reach_probability: the conformal guarantee and the sparsity mechanism.

The conformal coverage property is a THEOREM, not an empirical hope, so it is testable without any
trained model or chess data: feed exchangeable synthetic scores and the realised rejection rate must
track eps. That makes this a real regression guard -- if someone later drops Vovk's +1 correction or
calibrates on the wrong split, coverage breaks here rather than silently in a run.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from catspace.research.components.encoder.approaches.reach_probability.src.probability_less_than import (
    IMPOSSIBLE, REACHABLE, UNKNOWN, ReachPredicate)
from catspace.research.components.encoder.approaches.reach_probability.src.reach_jepa import ReachJEPA


class _Dummy:
    """Stands in for a trained net: ReachPredicate only needs .eval() and is used here for its
    p_value/tau arithmetic, which is where the guarantee lives."""
    def eval(self):
        return self


@pytest.mark.parametrize("eps", [0.01, 0.05, 0.10, 0.25])
def test_conformal_coverage_holds_on_exchangeable_scores(eps):
    """P(reject | the pair really is reachable) <= eps -- the whole contract of IMPOSSIBLE."""
    rng = np.random.default_rng(0)
    cal = rng.normal(size=20000)
    query = rng.normal(size=50000)                     # exchangeable with cal by construction
    pred = ReachPredicate(_Dummy(), cal)
    rate = float((pred.p_value(query) <= eps).mean())
    assert rate <= eps * 1.15 + 0.002, f"coverage violated at eps={eps}: realised {rate:.4f}"
    assert rate > eps * 0.6, f"absurdly conservative at eps={eps}: {rate:.4f} (test is not vacuous)"


def test_conformal_pvalue_is_uniform_under_the_null():
    """Valid p-values are ~U(0,1) on exchangeable data; that is what makes eps mean anything."""
    rng = np.random.default_rng(1)
    pred = ReachPredicate(_Dummy(), rng.normal(size=20000))
    p = pred.p_value(rng.normal(size=50000))
    for q in (0.1, 0.25, 0.5, 0.75):
        assert abs(float((p <= q).mean()) - q) < 0.02, f"p-value not uniform at {q}"


def test_vovk_plus_one_correction_is_present():
    """Without the +1s the p-value is anti-conservative at small n and coverage is simply wrong.
    A score below EVERY calibration point must still get p = 1/(n+1), never 0."""
    pred = ReachPredicate(_Dummy(), np.arange(10.0))
    assert pred.p_value([-100.0])[0] == pytest.approx(1.0 / 11.0)
    assert pred.p_value([1e9])[0] == pytest.approx(11.0 / 11.0)


def test_low_scores_are_flagged_and_high_scores_are_not():
    rng = np.random.default_rng(2)
    pred = ReachPredicate(_Dummy(), rng.normal(size=5000))
    pred.score = lambda a, b: np.array([-8.0, 0.0])    # far-below vs typical
    v = pred(None, None, eps=0.05)
    assert v[0].verdict == IMPOSSIBLE and v[1].verdict == UNKNOWN


def test_witness_always_wins_over_the_model():
    """An observed path is certain; no score may override it. REACHABLE is the one verdict that
    does not depend on the model being any good."""
    pred = ReachPredicate(_Dummy(), np.arange(1000.0))
    pred.score = lambda a, b: np.array([-1e9])          # model screams "impossible"
    v = pred(None, None, eps=0.5, witness=[(7, 3, 40)])
    assert v[0].verdict == REACHABLE and v[0].witness == (7, 3, 40)


def test_prox_l1_produces_exact_zeros_and_shrinks_support():
    """The measured failure this exists to fix: an L1 term in the loss under Adam left 64/64
    coordinates alive. The proximal operator must zero coordinates exactly."""
    net = ReachJEPA(in_ch=8, d=16, adapter_ch=4, hidden=32)
    with torch.no_grad():
        net.head_in.weight.copy_(torch.full_like(net.head_in.weight, 0.05))
    assert int(net.input_support().sum()) == 16
    net.prox_l1(0.10)                                   # threshold above every weight
    assert torch.count_nonzero(net.head_in.weight) == 0
    assert int(net.input_support().sum()) == 0


def test_prox_l1_keeps_large_weights_and_drops_small_ones():
    net = ReachJEPA(in_ch=8, d=4, adapter_ch=4, hidden=8)
    with torch.no_grad():
        net.head_in.weight.zero_()
        net.head_in.weight[:, 0] = 1.0                  # one strong coordinate
        net.head_in.weight[:, 1] = 0.01                 # one weak coordinate
    net.prox_l1(0.05)
    sup = net.input_support()
    assert bool(sup[0]) and not bool(sup[1]), "prox must keep the strong coord and drop the weak one"


def test_target_encoder_never_receives_gradient():
    """The EMA branch is the third anti-collapse defence; a gradient path into it would let the
    model make its own target easier to predict, which is collapse with extra steps."""
    net = ReachJEPA(in_ch=8, d=16, adapter_ch=4, hidden=32)
    assert all(not p.requires_grad for p in net.target_encoder.parameters())
    z = net.encode_target(torch.randn(4, 8, 8, 8))
    assert not z.requires_grad


def test_update_target_moves_toward_online_encoder():
    net = ReachJEPA(in_ch=8, d=16, adapter_ch=4, hidden=32, ema_decay=0.5)
    with torch.no_grad():
        for p in net.encoder.parameters():
            p.fill_(1.0)
        for p in net.target_encoder.parameters():
            p.fill_(0.0)
    net.update_target()
    assert all(torch.allclose(p, torch.full_like(p, 0.5)) for p in net.target_encoder.parameters())


def test_score_prefers_targets_inside_the_predicted_region():
    """score() must rank a target at the predicted mean above one far from it, at equal sigma --
    otherwise the conformal tail is meaningless."""
    net = ReachJEPA(in_ch=8, d=16, adapter_ch=4, hidden=32).eval()
    z_a = torch.randn(32, 16)
    with torch.no_grad():
        mu, _ = net.predict(z_a)
        near = net.score(z_a, mu)
        far = net.score(z_a, mu + 5.0)
    assert (near > far).all(), "a target at the region's centre must score above a distant one"
