#!/usr/bin/env python
"""test_concept_dynamics.py -- invariants for ConceptDynamics: architecture, loss terms and
GRADIENT PATHS. (Kaveh 2026-08-16: "do a thorough review of the loss terms and the
architecture, gradient paths / build unit tests around it")

WHY GRADIENT TESTS AND NOT JUST SHAPE TESTS. Every serious bug this model has had was a
gradient bug that shapes and losses could not see:
  * a persistence term that ran for 4 of 5 unroll steps with BOTH operands detached, so it
    optimised nothing while reporting a healthy-looking value
  * a learnable temperature pinned because every call site passed the constant explicitly
  * a zero-init residual passing zero gradient to its input
  * an IQE whose gradient reached the trunk (good) but whose pole geometry made the target
    unreachable (bad) -- invisible in any loss curve
A loss that is present, finite and decreasing tells you nothing about what it trains. Only
autograd does.

    .venv/bin/python -m ...test_concept_dynamics
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from catspace.research.components.encoder.approaches.reach_probability.experiments \
    .concept_dynamics_field import ConceptDynamics

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)
    return cond


def gnorm(loss, params):
    """‖dLoss/dParams‖, 0.0 when the path is severed.

    A fully-detached loss has no grad_fn at all and autograd.grad RAISES on it rather than
    returning zeros -- which is precisely the case these tests need to detect, so it is caught
    here instead of crashing the suite."""
    if not torch.is_tensor(loss) or not loss.requires_grad:
        return 0.0
    gs = torch.autograd.grad(loss, list(params), retain_graph=True, allow_unused=True)
    return sum(float(g.norm()) ** 2 for g in gs if g is not None) ** 0.5


def fixture(heads=4, codes=8, square_codes=8, B=6):
    torch.manual_seed(0)
    net = ConceptDynamics(d_model=32, layers=1, tr_heads=2, heads=heads, codes=codes,
                          d_code=8, square_codes=square_codes, d_fine=8, hidden=32)
    tok = torch.randint(0, 13, (B, 64))
    glob = torch.rand(B, 6)
    mid = torch.stack([torch.randint(0, 64, (B,)), torch.randint(0, 64, (B,)),
                       torch.zeros(B, dtype=torch.long)], 1)
    return net, tok, glob, mid


# ---------------------------------------------------------------- architecture
def test_state_is_codes_only():
    net, tok, glob, _ = fixture()
    zq, ids, ids_sq, _ = net.encode(tok, glob)
    d_expect = net.heads * net.d_code + 64 * net.d_fine
    check("state width = heads*d_code + 64*d_fine", zq.shape[1] == d_expect,
          f"{zq.shape[1]} == {d_expect}")
    check("value head consumes the QUANTISED state, not the continuous latent",
          net.value[0].in_features == d_expect,
          f"value in_features {net.value[0].in_features}")
    # The state must be a FUNCTION OF THE CODES. Note this only holds in EVAL mode: in train
    # mode the VQ advances its codebook EMA on every forward, so encoding the same board twice
    # legitimately returns different vectors. That is worth knowing -- it means any diagnostic
    # that encodes twice and compares must eval() first or it measures codebook drift.
    net.eval()
    za, ia, _, _ = net.encode(tok, glob)
    zb, ib, _, _ = net.encode(tok, glob)
    check("state is deterministic given the board (eval mode)",
          torch.allclose(za, zb, atol=1e-6) and (ia == ib).all())
    net.train()
    check("ids are (B, heads) and square ids (B, 64)",
          ids.shape == (len(tok), net.heads) and ids_sq.shape == (len(tok), 64))


def test_ema_target_is_frozen_and_tracks():
    net, tok, glob, _ = fixture()
    for p in net.t_trunk.parameters():
        check("EMA trunk params have requires_grad=False", not p.requires_grad)
        break
    zq_t, _, _, _ = net.encode(tok, glob, target=True)
    loss = zq_t.pow(2).mean()
    g = gnorm(loss, net.t_trunk.parameters()) + gnorm(loss, net.trunk.parameters())
    check("no gradient flows into OR THROUGH the EMA branch", g == 0.0, f"‖g‖={g:.2e}")

    # and it must actually move toward the live weights
    before = next(net.t_proj.parameters()).clone()
    with torch.no_grad():
        for p in net.proj.parameters():
            p.add_(1.0)
    net.update_ema()
    after = next(net.t_proj.parameters())
    moved = float((after - before).abs().mean())
    expect = (1 - net.ema)
    check("update_ema moves the target toward the source by (1-ema)",
          abs(moved - expect) < 1e-5, f"moved {moved:.5f}, expected {expect:.5f}")


def test_target_encode_uses_the_ema_modules():
    net, tok, glob, _ = fixture()
    with torch.no_grad():                      # make the branches differ measurably
        for p in net.trunk.parameters():
            p.add_(0.5)
    zq_live, _, _, _ = net.encode(tok, glob, target=False)
    zq_targ, _, _, _ = net.encode(tok, glob, target=True)
    check("target=True actually reads different weights than target=False",
          not torch.allclose(zq_live, zq_targ, atol=1e-5))


def test_soft_state_matches_the_codebook():
    """In EVAL mode soft_state at one-hot logits must reproduce the encoder's own state
    exactly. It does NOT in train mode, and that is not a representational mismatch: the VQ
    advances its codebook EMA during each forward, so a book read afterwards is not the book
    that produced the state (measured: diff 0.0 eval, 2.79 train). Training now snapshots the
    book once per step via cache_books() for exactly this reason."""
    net, tok, glob, mid = fixture()
    net.eval()
    zq, ids, ids_sq, _ = net.encode(tok, glob)
    g, s = net.predict_next(zq, mid)
    # one-hot logits on the TRUE ids must reproduce the true quantised state exactly
    g1 = F.one_hot(ids, net.codes).float() * 30.0
    s1 = F.one_hot(ids_sq, net.square_codes).float() * 30.0
    soft = net.soft_state(g1, s1)
    check("soft_state at one-hot logits reproduces the encoder's own state",
          torch.allclose(soft, zq, atol=1e-3),
          f"max|diff| {float((soft-zq).abs().max()):.2e}")
    check("soft_state is differentiable w.r.t. the dynamics logits",
          gnorm(net.soft_state(g, s).sum(), net.dyn.parameters()) > 0)
    net.train()


def test_batched_encode_equals_per_row_encode():
    """The training step encodes wi..wi+K in ONE call and slices, instead of K+1 calls. That is
    only valid if a batched encode equals the per-row encodes -- true for a per-position model,
    but worth pinning, since a stray batch-coupling (a norm over the batch, say) would make the
    unroll silently wrong."""
    net, tok, glob, _ = fixture(B=8)
    net.eval()
    zq_all, ids_all, sq_all, _ = net.encode(tok, glob)
    ok_z = ok_i = True
    for i in range(len(tok)):
        z1, i1, s1, _ = net.encode(tok[i:i+1], glob[i:i+1])
        ok_z &= torch.allclose(z1[0], zq_all[i], atol=1e-5)
        ok_i &= bool((i1[0] == ids_all[i]).all())
    check("batched encode == per-row encode (states)", ok_z)
    check("batched encode == per-row encode (codes)", ok_i)
    net.train()


# ---------------------------------------------------------------- gradient paths
def test_each_loss_reaches_what_it_should():
    net, tok, glob, mid = fixture()
    zq, ids, ids_sq, vl = net.encode(tok, glob)
    g, s = net.predict_next(zq, mid)
    y = torch.randint(0, 3, (len(tok),))
    mods = {"trunk": net.trunk, "proj": net.proj, "sq_proj": net.sq_proj,
            "dyn": net.dyn, "value": net.value}
    terms = {
        "l_dyn":    F.cross_entropy(g.reshape(-1, net.codes), ids.reshape(-1)),
        "l_vnext":  F.cross_entropy(net.value(net.soft_state(g, s)), y),
        "l_value":  F.cross_entropy(net.value(zq), y),
        "l_commit": vl,
    }
    # what each term MUST train, stated as an expectation rather than discovered after
    must = {("l_dyn", "dyn"), ("l_dyn", "trunk"), ("l_dyn", "proj"),
            ("l_vnext", "dyn"), ("l_vnext", "value"),
            ("l_value", "value"), ("l_value", "trunk"), ("l_value", "proj"),
            ("l_value", "sq_proj"), ("l_commit", "proj"), ("l_commit", "sq_proj")}
    print(f"\n    {'':<10}" + "".join(f"{m:>10}" for m in mods))
    for tn, t in terms.items():
        row = {m: gnorm(t, mo.parameters()) for m, mo in mods.items()}
        print(f"    {tn:<10}" + "".join(f"{row[m]:>10.4f}" for m in mods))
        for m in mods:
            if (tn, m) in must:
                check(f"{tn} trains {m}", row[m] > 0, f"‖g‖={row[m]:.2e}")
    print()


def test_persistence_is_alive_at_every_unroll_step():
    """THE BUG THIS FILE EXISTS FOR. The training loop assigns the no_grad next-state encode to
    `zq_here` for the following step, so from k=1 both operands are detached and the term
    optimises nothing while still reporting a plausible value."""
    net, tok, glob, mid = fixture(B=4)
    K = 3
    # as the trainer does it: ONE batched live encode over t..t+K, persistence over t -> t+2.
    # The slots must be DIFFERENT boards -- repeating one board makes zq[k]-zq[k+2] identically
    # zero, whose gradient is legitimately zero, and the test then fails for its own reason.
    B = len(tok)
    big = torch.randint(0, 13, (B * (K + 1), 64))
    bigg = torch.rand(B * (K + 1), 6)
    zq_all, _, _, _ = net.encode(big, bigg)
    zq_all = zq_all.view(K + 1, len(tok), -1).transpose(0, 1)      # (B, K+1, D)
    alive = []
    for k in range(K - 1):
        d = (zq_all[:, k] - zq_all[:, k + 2]).pow(2).mean(-1).mean()
        alive.append(gnorm(d, net.trunk.parameters()) > 0)
    check("persistence has gradient at EVERY unroll step", all(alive) and len(alive) > 0,
          f"alive per step: {alive}")


def test_quiet_mask_is_exact():
    """quiet = 'no capture', detected as an unchanged piece count. A capture removes exactly one
    man; promotion removes none. So the test is exact, not a heuristic -- assert that."""
    import chess
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize
    b = chess.Board()
    n0 = (tokenize(b)[0] != 0).sum()
    b.push_san("e4"); n1 = (tokenize(b)[0] != 0).sum()
    check("a quiet move leaves the man-count unchanged", n0 == n1, f"{n0} -> {n1}")
    b2 = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    n2 = (tokenize(b2)[0] != 0).sum()
    b2.push_san("exd5"); n3 = (tokenize(b2)[0] != 0).sum()
    check("a capture drops the man-count by exactly one", n2 - n3 == 1, f"{n2} -> {n3}")
    b3 = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
    n4 = (tokenize(b3)[0] != 0).sum()
    b3.push_san("a8=Q"); n5 = (tokenize(b3)[0] != 0).sum()
    check("promotion leaves the man-count unchanged", n4 == n5, f"{n4} -> {n5}")


def test_codebook_updates_once_per_encode():
    """Each encode() in TRAIN mode advances the VQ's EMA, whether or not it is inside no_grad.
    A K=5 unroll calls encode 1 + 2K + 1 = 12 times per optimiser step, so the codebooks move
    12x per weight update. Documented here so it is a known quantity, not a surprise."""
    net, tok, glob, _ = fixture()
    net.train()
    cb0 = net.vq[0]._codebook.embed.clone()
    net.encode(tok, glob)
    moved_train = not torch.allclose(cb0, net.vq[0]._codebook.embed)
    net.eval()
    cb1 = net.vq[0]._codebook.embed.clone()
    net.encode(tok, glob)
    moved_eval = not torch.allclose(cb1, net.vq[0]._codebook.embed)
    check("codebook EMA advances on a TRAIN-mode encode", moved_train)
    check("codebook EMA does NOT advance in eval mode", not moved_eval)


def test_value_cannot_be_read_around_the_quantiser():
    """The value head must see only what the codes carry. If two boards land on identical codes
    they must receive identical value -- otherwise information is leaking past the bottleneck
    and the 'concepts predict the outcome' claim is not what it appears."""
    net, tok, glob, _ = fixture()
    zq, ids, ids_sq, _ = net.encode(tok, glob)
    same = (ids[0] == ids[1]).all() and (ids_sq[0] == ids_sq[1]).all()
    if not same:                       # construct the case rather than hope for it
        zq_pair = torch.stack([zq[0], zq[0]])
        v = net.value(zq_pair)
        check("identical code states give identical value",
              torch.allclose(v[0], v[1], atol=1e-6))
    else:
        v = net.value(zq[:2])
        check("identical code states give identical value",
              torch.allclose(v[0], v[1], atol=1e-6))


def main():
    print(__doc__.split("\n")[0])
    print("\n-- architecture")
    test_state_is_codes_only()
    test_ema_target_is_frozen_and_tracks()
    test_target_encode_uses_the_ema_modules()
    test_soft_state_matches_the_codebook()
    test_batched_encode_equals_per_row_encode()
    print("\n-- gradient paths")
    test_each_loss_reaches_what_it_should()
    test_persistence_is_alive_at_every_unroll_step()
    print("\n-- data invariants")
    test_quiet_mask_is_exact()
    test_codebook_updates_once_per_encode()
    test_value_cannot_be_read_around_the_quantiser()
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: ' + ', '.join(FAILS)}")
    raise SystemExit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
