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


# ---------------------------------------------------------------------------------------------
# The ViT / every-ply rebuild (2026-08-05). These guard properties that are ASSERTED in module
# docstrings and relied on downstream -- each one is a claim the design rests on, so it gets a
# regression test rather than a comment.

def test_packed_planes_round_trip_is_exact():
    """889 B/position instead of 7168 is what makes an every-ply plane cache affordable at all
    (18 GB rather than 143 GB). It is only allowed to be lossless: exactly one of the 112 planes
    (109, rule50) is non-binary and it is constant across squares, so 111 bit-pack and rule50
    takes a byte. If a future lc0 input format breaks that, this fails loudly rather than
    silently feeding the frozen trunk corrupted planes."""
    from lczerolens import LczeroBoard
    from catspace.research.components.encoder.approaches.reach_probability.src.lc0_prefix import (
        PACKED_BYTES, pack_planes, unpack_planes, RULE50_PLANE)
    fens = ["8/8/4k3/8/8/4K3/8/8 w - - 77 120", LczeroBoard().fen(),
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"]
    t = torch.stack([LczeroBoard(f).to_input_tensor() for f in fens])
    p = pack_planes(t)
    assert p.shape == (3, PACKED_BYTES) and p.dtype == np.uint8
    assert np.array_equal(unpack_planes(p), t.numpy()), "plane packing must be exactly lossless"
    # the one non-binary plane really is the only one, and really is the halfmove clock
    nb = [i for i in range(112) if not bool(((t[0, i] == 0) | (t[0, i] == 1)).all())]
    assert nb == [RULE50_PLANE], f"expected only rule50 to be non-binary, got {nb}"


def test_dual_iqe_sum_is_a_quasimetric_and_human_never_beats_best():
    """d_human = d_best + d_mistake. The sum of quasimetrics is a quasimetric (so every downstream
    use of the axioms stays valid for the human field), and d_human >= d_best by construction --
    a fallible player can never appear to reach a goal faster than perfect play. The second is the
    property a free-form residual would lose: it could learn a negative correction and quietly
    claim humans outplay Stockfish."""
    from catspace.research.components.encoder.approaches.reach_probability.src.reach_vit_jepa import (
        DualIQEHead)
    torch.manual_seed(0)
    h = DualIQEHead(d_in=32, d=64, components=16)
    with torch.no_grad():
        h.proj_res.weight.normal_(0, 0.1)
    phi = torch.randn(128, 32)
    zb, zr = h.embed(phi)
    x, y, z = zb[:40], zb[40:80], zb[80:120]
    rx, ry, rz = zr[:40], zr[40:80], zr[80:120]
    dh = lambda a, b, ra, rb: h.d_best(a, b) + h.d_mistake(ra, rb)
    assert float(dh(x, x, rx, rx).abs().max()) < 1e-6, "d(x,x) must be 0"
    assert bool((dh(x, y, rx, ry) >= 0).all()), "distances must be non-negative"
    assert bool((dh(x, z, rx, rz) <= dh(x, y, rx, ry) + dh(y, z, ry, rz) + 1e-4).all()), \
        "triangle inequality must survive the sum"
    assert bool((dh(x, y, rx, ry) >= h.d_best(x, y) - 1e-6).all()), "d_human >= d_best"


def test_dual_iqe_residual_is_zero_init_and_identifiable():
    """Two separate guards on the best/mistake split:

    ZERO INIT -- training must begin at 'humans play perfectly' and be pushed off it by data.
    IDENTIFIABILITY -- human rows must not train the BASE. Without the stop-gradient, any amount
    of human error can be absorbed into d_best and the decomposition becomes an artifact of
    optimisation order rather than a measurement (measured: base grad 2615.9 vs 0.0)."""
    from catspace.research.components.encoder.approaches.reach_probability.src.reach_vit_jepa import (
        DualIQEHead)
    torch.manual_seed(0)
    h = DualIQEHead(d_in=32, d=64, components=16)
    phi = torch.randn(64, 32)
    zb, zr = h.embed(phi)
    perm = torch.randperm(64)
    assert float(h.d_mistake(zr, zr[perm]).abs().max()) == 0.0, "residual must start at exactly 0"

    with torch.no_grad():
        h.proj_res.weight.normal_(0, 0.1)
    human = torch.ones(64, dtype=torch.long)
    h.zero_grad()
    zb, zr = h.embed(phi)
    h.distance(zb, zb[perm], zr, zr[perm], human, detach_base=True).sum().backward()
    assert float(h.proj_best.weight.grad.abs().sum()) == 0.0, \
        "human rows must not push gradient into the best-play head"
    h.zero_grad()
    zb, zr = h.embed(phi)
    h.distance(zb, zb[perm], zr, zr[perm], torch.zeros(64, dtype=torch.long)).sum().backward()
    assert float(h.proj_res.weight.grad.abs().sum()) == 0.0, \
        "SF rows must not push gradient into the mistake head"


def test_iqe_zero_distance_is_domination_so_subsumption_composes():
    """The subsumption hierarchy rests entirely on this: in an IQE, d(u->v)=0 means exactly that u
    dominates v coordinatewise, and it coexists with d(v->u) > 0. That is what lets a specific
    3-fold position sit at 0-ply from 'general 3-fold', which sits at 0-ply from 'general draw',
    while none of the reverses collapse -- and mutual zero forces equality, so it stays a genuine
    partial order rather than a pile."""
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE
    torch.manual_seed(0)
    q = IQE(d=32, components=4)
    gdraw = torch.zeros(1, 32)
    g3 = gdraw + torch.rand(1, 32) * 0.5              # dominates gdraw
    spec = g3 + torch.rand(3, 32) * 0.7               # each dominates g3
    assert float(q(spec, g3.expand(3, -1)).max()) == 0.0, "instance -> its abstraction must be 0"
    assert float(q(g3, gdraw).max()) == 0.0
    assert float(q(spec, gdraw.expand(3, -1)).max()) == 0.0, "domination must compose transitively"
    assert bool((q(g3.expand(3, -1), spec) > 0).all()), "the reverse must NOT be zero"
    assert bool((q(gdraw.expand(3, -1), spec) > 0).all())
    assert float(q(spec[:1], spec[:1])) == 0.0 and float(q(spec[:1], spec[1:2])) > 0, \
        "distinct points must not be mutually zero"


def test_pole_bank_starts_source_agnostic_and_chains_to_roots():
    """The dynamics residual is zero-initialised, so at step 0 the model says 'human and SF do not
    differ' and only data can push it apart -- otherwise a nonzero divergence at init would be
    reported as a finding. And every pole must reach a WIN/DRAW/LOSS root, since the outcome level
    is what the whole hierarchy subsumes into."""
    from catspace.research.components.encoder.approaches.reach_probability.src.reach_vit_jepa import (
        PoleBank)
    parent = torch.tensor([-1, -1, -1, 1, 1, 0, 5])
    pb = PoleBank(7, 32, parent, n_sources=2)
    src = torch.tensor([0, 0, 1, 1])
    P = pb.for_source(src)
    assert P.shape == (4, 7, 32)
    assert torch.equal(P[0], P[2]), "sources must be identical at init"
    assert float(pb.source_divergence().abs().max()) == 0.0
    with torch.no_grad():
        pb.delta[0, 3] += 0.5
    dv = pb.source_divergence()
    assert float(dv[3]) > 0 and float(dv[np.array([0, 1, 2, 4, 5, 6])].abs().max()) == 0.0, \
        "nudging one pole must move only that pole's divergence"
    for i in range(7):
        q, hops = i, 0
        while pb.parent[q] >= 0:
            q, hops = int(pb.parent[q]), hops + 1
        assert q < 3 and hops <= 3, "every pole must chain to an outcome root"


def test_terminal_taxonomy_outcomes_match_the_board():
    """Every declared terminal outcome is checked against what the position actually is. Three
    first drafts failed exactly here: WIN_MATE is impossible under mover-POV, RESIGN needed
    splitting because 11.7% of resignations fire on the winner's turn, and 'agreed' draws were
    99% SF adjudications."""
    import chess
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    # a real mate: the side to move is mated, so it is a LOSS for the mover
    b = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert b.is_checkmate()
    t = T.classify_terminal(b, flagged=False)
    assert T.TERMINALS[t] == "LOSS_MATE" and T.TERM_OUTCOME[t] == T.LOSS
    assert "WIN_MATE" not in T.TERMINALS, "mover-POV admits no WIN_MATE terminal"
    # a flagged game is CENSORED, never given a pole -- its board says nothing about the result
    assert T.classify_terminal(chess.Board(), flagged=True) == T.TERM_TIME
    # stalemate is a draw by rule
    sm = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert sm.is_stalemate()
    assert T.TERMINALS[T.classify_terminal(sm, False)] == "DRAW_STALEMATE"
    # radius: all draws sit AT the pole, only resignation sits off it
    for name in T.TERMINALS:
        r = float(T.TERM_RADIUS[T.TERM_ID[name]])
        assert r == (1.0 if name.startswith("RESIGN") else 0.0), f"{name} radius {r}"
    assert len(T.TERM_RADIUS) == len(T.TERM_OUTCOME) == len(T.TERMINALS)
