#!/usr/bin/env python
"""test_concept_capture.py -- structural invariants for the capture-trained concept field.

Ported from test_concept_dynamics.py (minus the EMA/unroll checks, which no longer exist) and
extended with the factored resolution head. Every check here is a SCAR: something that was
silently wrong once and cost a run.

    .venv/bin/python -m ...test_concept_capture
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from catspace.research.tools.training_infra.losses import (
    capture_ce, compose_resolution_logits, effective_number_weights, factorized_resolution_ce)
from catspace.research.components.encoder.approaches.reach_probability.experiments \
    .concept_capture_field import (ConceptCapture, N_CAP, N_PIECE, factor_vocab,
                               signature_of)

VOCAB = (["WDL:W", "WDL:D", "WDL:L"]
         + [f"{s}={o}" for s in ["KRP|KR", "KQ|KR", "KP|K", "KBN|K"] for o in ["W", "D", "L"]]
         + ["OTHER"])


def _net(factored=True, **kw):
    cr, ce, co, egs = factor_vocab(VOCAB)
    return ConceptCapture(d_model=64, layers=2, tr_heads=4, heads=4, codes=8, d_code=16,
                          square_codes=16, d_fine=16, hidden=64, n_res=len(VOCAB),
                          factored=factored, n_eg=len(egs),
                          cls_reach=cr, cls_eg=ce, cls_out=co, **kw)


def _batch(net, b=6):
    """Global-feature width is read off the trunk, not guessed -- the encoder owns it."""
    torch.manual_seed(0)
    tok = torch.randint(0, N_PIECE, (b, 64))
    glob = torch.randn(b, net.trunk.glob_proj.in_features)
    return tok, glob


def main():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        cond = bool(cond); ok &= cond
        print(f"  {'OK  ' if cond else 'FAIL'} {name}{('  ' + extra) if extra else ''}")

    # ---- vocabulary factorisation -------------------------------------------------------
    cr, ce, co, egs = factor_vocab(VOCAB)
    sig_of, sigs = signature_of(egs)
    chk("factor_vocab: 12 endings over 4 signatures", len(egs) == 12 and len(sigs) == 4,
        f"{len(egs)} endings, {sigs}")
    chk("factor_vocab: WDL rows are reach=0, no ending, outcome kept",
        (cr[:3] == 0).all() and (ce[:3] == -1).all() and (co[:3] == [0, 1, 2]).all())
    chk("factor_vocab: endgame rows are reach=1 with an ending index",
        (cr[3:15] == 1).all() and (ce[3:15] >= 0).all())
    # THE double-count guard: an ending's outcome lives inside its index, so cls_out must be -1
    chk("factor_vocab: endings carry NO separate outcome (no double count)",
        (co[3:15] == -1).all())
    chk("factor_vocab: OTHER is reach=2 with NO factors",
        cr[-1] == 2 and ce[-1] == -1 and co[-1] == -1)
    seen = {(int(a), int(b), int(c)) for a, b, c in zip(cr, ce, co)}
    chk("factor_vocab: the (reach,eg,out) triples are unique per class",
        len(seen) == len(VOCAB), f"{len(seen)}/{len(VOCAB)}")
    # marginalising the joint head must recover the signature -- the interpretable readout
    chk("signature_of: same material, different outcome -> same signature",
        sig_of[0] == sig_of[1] == sig_of[2] and sig_of[0] != sig_of[3],
        f"{egs[0]},{egs[1]},{egs[2]} share a signature")

    # ---- model plumbing -----------------------------------------------------------------
    net = _net(); net.eval()
    tok, glob = _batch(net)
    with torch.no_grad():
        zq, ids, ids_sq, vl = net.encode(tok, glob)
    chk("encode: state is range-normalised to [0,1]",
        float(zq.min()) >= -1e-6 and float(zq.max()) <= 1 + 1e-6,
        f"[{float(zq.min()):.3f}, {float(zq.max()):.3f}]")
    chk("encode: global ids (B,H) in range", ids.shape == (6, 4) and int(ids.max()) < 8)
    chk("encode: square ids (B,64) in range", ids_sq.shape == (6, 64) and int(ids_sq.max()) < 16)

    # BATCHING MUST NOT CHANGE THE ANSWER. A speed optimisation once carried a no_grad encode
    # into a live term and killed persistence at k>=1; batched-vs-single is the cheap guard.
    with torch.no_grad():
        one = torch.stack([net.encode(tok[i:i + 1], glob[i:i + 1])[0][0] for i in range(6)])
    chk("encode: batched == per-row", torch.allclose(zq, one, atol=1e-5),
        f"max |diff| {float((zq - one).abs().max()):.2e}")

    # ---- factored resolution head -------------------------------------------------------
    r, e, o = net.predict_resolution(zq)
    chk("predict_resolution: shapes (B,3) (B,n_eg) (B,3)",
        r.shape == (6, 3) and e.shape == (6, 12) and o.shape == (6, 3))
    lg = net.resolution_logits(zq)
    chk("resolution_logits: (B, n_res) over the ORIGINAL flat classes",
        lg.shape == (6, len(VOCAB)))
    # THE correctness property: composing the factors is a proper distribution over flat classes
    lse = lg.logsumexp(-1)
    chk("resolution_logits: composed distribution sums to 1",
        torch.allclose(lse, torch.zeros(6), atol=1e-5), f"logsumexp max {float(lse.abs().max()):.2e}")

    # ---- the collapse this replaces -----------------------------------------------------
    # with a batch that is 75% WDL, the material head must see ONLY the reaching rows
    rt = torch.tensor([0, 0, 0, 1, 1, 2])
    et = torch.tensor([-1, -1, -1, 0, 7, -1])
    ot = torch.tensor([0, 1, 2, -1, -1, -1])
    r2 = torch.randn(6, 3, requires_grad=True)
    m2 = torch.randn(6, 12, requires_grad=True)
    o2 = torch.randn(6, 3, requires_grad=True)
    tot, parts = factorized_resolution_ce(r2, m2, o2, rt, et, ot)
    g = torch.autograd.grad(tot, m2, retain_graph=True)[0]
    chk("factored loss: ending gradient touches only reaching rows",
        g[[0, 1, 2, 5]].abs().sum().item() == 0 and g[[3, 4]].abs().sum().item() > 0)
    chk("factored loss: reports its three parts separately",
        set(parts) == {"res_reach", "res_eg", "res_out"})
    go = torch.autograd.grad(tot, o2, retain_graph=True)[0]
    chk("factored loss: outcome gradient skips reaching AND other rows",
        go[[3, 4, 5]].abs().sum().item() == 0 and go[[0, 1, 2]].abs().sum().item() > 0)
    # a batch with no reaching rows at all must stay finite (whole-batch-masked case)
    tot0, _ = factorized_resolution_ce(r2, m2, o2, torch.zeros(6, dtype=torch.long),
                                       torch.full((6,), -1, dtype=torch.long), ot)
    chk("factored loss: batch with zero reaching rows stays finite", torch.isfinite(tot0))

    # ---- class balance ------------------------------------------------------------------
    w = effective_number_weights(torch.tensor([57681, 1197, 75]))
    chk("class balance: rare signature up-weighted, mean 1",
        float(w[2]) > float(w[0]) and abs(float(w.mean()) - 1) < 1e-4,
        f"{float(w[2])/float(w[0]):.1f}x spread")
    totw, _ = factorized_resolution_ce(r2, m2, o2, rt, et, ot,
                                       eg_weight=effective_number_weights(
                                           torch.tensor([100]*6 + [50, 25, 10, 8, 6, 4])))
    chk("class balance: weighted loss runs and is finite", torch.isfinite(totw))

    # ---- capture heads ------------------------------------------------------------------
    s, v, a = net.predict_captures(zq)
    chk("predict_captures: (B,N,64) (B,N,13) (B,N,13)",
        s.shape == (6, N_CAP, 64) and v.shape == (6, N_CAP, N_PIECE)
        and a.shape == (6, N_CAP, N_PIECE))
    tgt = torch.full((6,), -1, dtype=torch.long); tgt[:2] = torch.tensor([5, 9])
    lc = capture_ce(s[:, 0], tgt)
    chk("capture_ce: -1 slots skipped, loss finite", torch.isfinite(lc))

    # ---- flat head still loads (ccf1 comparability) --------------------------------------
    flat = _net(factored=False)
    chk("flat head: keeps its original module name (old ckpts load)",
        any(k.startswith("resolution.") for k in flat.state_dict()))
    with torch.no_grad():
        chk("flat head: resolution_logits returns (B, n_res) too",
            flat.resolution_logits(flat.encode(tok, glob)[0]).shape == (6, len(VOCAB)))
    try:
        flat.predict_resolution(zq); raised = False
    except RuntimeError:
        raised = True
    chk("flat head: predict_resolution REFUSES rather than inventing factors", raised)

    # ---- gradients reach the trunk ------------------------------------------------------
    net.train()
    zq2, _, _, vl2 = net.encode(tok, glob)
    r3, e3, o3 = net.predict_resolution(zq2)
    lt, _ = factorized_resolution_ce(r3, e3, o3, rt, et, ot)
    lt.backward()
    tg = [p.grad.abs().sum().item() for n_, p in net.named_parameters()
          if n_.startswith("trunk") and p.grad is not None]
    chk("resolution loss trains the TRUNK end-to-end (not just the head)",
        len(tg) > 0 and sum(tg) > 0, f"{len(tg)} trunk tensors with gradient")

    # ---- codebook config is the one that fixed collapse ----------------------------------
    chk("VQ: dead-code replacement is on (4.7 -> 16/16 alive)",
        all(float(q._codebook.threshold_ema_dead_code) >= 2 for q in net.vq)
        and float(net.sq_vq._codebook.threshold_ema_dead_code) >= 2)

    print("\nALL CONCEPT-CAPTURE TESTS PASSED" if ok else "\nCONCEPT-CAPTURE TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
