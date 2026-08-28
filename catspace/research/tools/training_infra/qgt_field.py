"""QGTField: pure-GNN base -> RVQ observables of a rollable carrier -> QA levels -> readout.

Locked spec 2026-08-28 (+ carrier/observable amendment). V1 SCOPE DEVIATION, recorded:
squares + moves only; board/piece/histogram families join in v2 -- fail-fast first.

    base      GraphBase(NoMixer): relations only, no dense attention anywhere
    carrier   c in R^(64*24): per-square continuous pre-quant features. ROLLABLE:
              c' = K c + B m (KoopmanRoller), never re-lift.
    level-1   ResidualLFQ(num_quantizers=2, EnCodec dropout) per square -> 24 bits/square
              = quantized OBSERVABLES of the carrier (read-after-roll, never rolled)
    level-2+  stacked QuantizedAttention over the 64 level-1 tokens, ST-MoE router losses
    readout   linear over all levels' bits + turn bias + PERMUTATION-flipped move term
              (qat9 semantics: W<->L swap for Black, D fixed; CE sees mov detached)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from catspace.research.tools.training_infra.qgt_base import GraphBase, router_losses
from catspace.research.tools.training_infra.qgt_roller import KoopmanRoller


class QGTField(nn.Module):
    def __init__(self, d=128, layers=3, sq_bits=24, rvq_levels=2, mv_bits=32,
                 qa_levels=2, n_q=16, qa_heads=4, top_k=4, out_bits=8):
        super().__init__()
        from vector_quantize_pytorch import LFQ
        self.base = GraphBase(d=d, layers=layers)          # NoMixer by default
        self.sq_bits, self.mv_bits = sq_bits, mv_bits
        self.carrier_proj = nn.Linear(d, sq_bits)          # square carrier, rollable
        # EXPLICIT RVQ: ResidualLFQ returns the SUMMED reconstruction, which is not
        # bits. Here each level LFQ-quantizes (sq_bits/levels) bits of the residual the
        # previous level left; the OBSERVABLE is the concatenation of level codes
        # (strictly +-1, coarse-to-fine), the reconstruction stays internal.
        w = sq_bits // rvq_levels
        self.rvq_down = nn.ModuleList([nn.Linear(sq_bits, w) for _ in range(rvq_levels)])
        self.rvq_up = nn.ModuleList([nn.Linear(w, sq_bits) for _ in range(rvq_levels)])
        self.rvq_q = nn.ModuleList([LFQ(dim=w, codebook_size=2, num_codebooks=w)
                                    for _ in range(rvq_levels)])
        self.mv_proj = nn.Linear(d, mv_bits)
        self.mv_lfq = LFQ(dim=mv_bits, codebook_size=2, num_codebooks=mv_bits)
        self.dyn_mv_proj = nn.Linear(d, mv_bits)
        self.dyn_mv_lfq = LFQ(dim=mv_bits, codebook_size=2, num_codebooks=mv_bits)
        self.qa = nn.ModuleList()
        from catspace.research.tools.training_infra.quantized_attention import (
            QuantizedAttention)
        for _ in range(qa_levels):
            self.qa.append(QuantizedAttention(sq_bits if not self.qa else out_bits,
                                              n_q=n_q, n_head=qa_heads, top_k=top_k,
                                              out_bits=out_bits))
        self.d_l1 = 64 * sq_bits
        self.d_l2 = qa_levels * 64 * out_bits
        self.d_pos = self.d_l1 + self.d_l2
        self.readout = nn.Linear(self.d_pos + mv_bits, 3)
        self.turn = nn.Parameter(torch.zeros(2, 3))
        self.roller = KoopmanRoller(64 * sq_bits, mv_bits)

    def carrier(self, sqf, mvf, edges, mv_mask):
        h, aux = self.base(sqf, mvf, edges, mv_mask)
        c = self.carrier_proj(h[:, :64])                   # (B,64,sq_bits) continuous
        return c, h[:, 64:], aux

    def rvq_bits(self, c):
        """carrier -> level-1 bits only (B,64,sq_bits) + the RVQ aux loss."""
        res, lvl_bits, l_rvq = c, [], torch.zeros((), device=c.device)
        for dn, up, q in zip(self.rvq_down, self.rvq_up, self.rvq_q):
            qb, _, lq = q(dn(res))
            res = res - up(qb)
            lvl_bits.append(qb)
            l_rvq = l_rvq + (lq.sum() if torch.is_tensor(lq) else 0.0)
        return torch.cat(lvl_bits, -1), l_rvq              # strictly +-1

    def levels_from_bits(self, q1):
        """level-1 bit tokens (B,64,sq_bits) -> (all-level bits, router aux). The QA
        stack reads bit tokens directly, so a DISCRETE roll reuses levels 2+ as-is."""
        B = q1.shape[0]
        toks, bits, l_route = q1, [q1.reshape(B, -1)], torch.zeros((), device=q1.device)
        for qa in self.qa:
            b2, aux = qa(toks)
            l_route = l_route + aux["vq_loss"] + router_losses(aux["qlog"],
                                                               aux["attn_logit"])
            bits.append(b2.reshape(B, -1))
            toks = b2
        return torch.cat(bits, -1), l_route

    def observe(self, c):
        """carrier -> (all-level bits, aux losses). Quantize-after-roll lives here."""
        q1, l_rvq = self.rvq_bits(c)
        z, l_route = self.levels_from_bits(q1)
        return z, l_rvq + l_route

    def forward(self, sqf, mvf, edges, mv_mask, stm=None):
        """-> (per-child logits, logit_ce, z_bits, aux)."""
        c, h_mv, aux_b = self.carrier(sqf, mvf, edges, mv_mask)
        z, l_obs = self.observe(c)
        mvb, _, l_mv = self.mv_lfq(self.mv_proj(h_mv))
        W = self.readout.weight
        pos = F.linear(z, W[:, :self.d_pos], self.readout.bias)
        mov = F.linear(mvb, W[:, self.d_pos:])
        if stm is not None:
            pos = pos + self.turn[stm]
            mov = torch.where((stm == 0)[:, None, None], mov[:, :, [2, 1, 0]], mov)
        logit = pos[:, None, :] + mov
        logit_ce = pos[:, None, :] + mov.detach()
        aux = {"loss_aux": l_obs + l_mv + aux_b["mixer_loss"], "carrier": c,
               "h_mv": h_mv}
        return logit, logit_ce, z, aux

    def roll_and_read(self, c, dyn_moves, stm_seq):
        """Roll the carrier through move codes, read observables + position value at each
        horizon. NEVER re-lifts; the only nonlinearity is binarization at read time."""
        B = c.shape[0]
        cs = self.roller.roll(c.reshape(B, -1), dyn_moves)
        out = []
        for k, ck in enumerate(cs):
            z, _ = self.observe(ck.reshape(B, 64, self.sq_bits))
            pv = F.linear(z, self.readout.weight[:, :self.d_pos], self.readout.bias)
            pv = pv + self.turn[stm_seq[k]]
            out.append((z, pv))
        return out


class BitTransition(nn.Module):
    """qgt3d (Kaveh 2026-08-28: "is there anything that says concepts can't be rolled
    or is it just our own tests?" -- just our own tests, and they only ever indicted
    EMA bit-matching): a DISCRETE transition on level-1 bits, z' = sign(T(z, m)). The
    state that advances IS the concepts; there is no carrier and no spectral radius to
    contract (the mechanism that killed qgt1/qgt2). Trained TEACHER-FORCED on real
    chain pairs -- true bits in, true next-position bits as target -- the IRIS-class
    scheme this project never tried.

    Structure: a strong COPY PRIOR (w_id * z, head's last layer zero-init, so T starts
    as the identity and learns only the deltas -- most bits do not flip per ply, and
    "fewest concept changes per ply" is the standing goal) + one QuantizedAttention
    layer for the relational path (the discrete questions become "what does this move
    change") + a per-square MLP. Nonlinear on purpose: the interpretability constraint
    is on the READOUT, not the transition (qat_train doctrine)."""

    def __init__(self, sq_bits=24, mv_bits=32, n_q=16, heads=4, top_k=4, out_bits=8,
                 w_id_init=0.5):
        super().__init__()
        from catspace.research.tools.training_infra.quantized_attention import (
            QuantizedAttention)
        self.mv_in = nn.Linear(mv_bits, sq_bits)
        self.qa = QuantizedAttention(sq_bits, n_q=n_q, n_head=heads, top_k=top_k,
                                     out_bits=out_bits)
        self.head = nn.Sequential(nn.Linear(sq_bits + out_bits, 4 * sq_bits),
                                  nn.GELU(), nn.Linear(4 * sq_bits, sq_bits))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        # w_id_init sets the BARRIER the head must overcome before a bit may flip.
        # 3.0 proved un-erodable in a live run (qgt3d first launch: 1000 steps, T
        # still exactly the identity, tfacc glued to the copy baseline while the
        # identity rolls fed noise to the readout). 0.5 keeps the copy INIT --
        # sign(0.5 z) = z -- but lets learned deltas act within a few hundred steps.
        self.w_id = nn.Parameter(torch.tensor(float(w_id_init)))

    def forward(self, z, m, tok_mask=None):
        """z: (B,64,sq_bits) +-1 bits; m: (B,mv_bits) move code -> next-bit LOGITS.
        tok_mask (B,64) bool, optional: squares the relational path may LOOK AT --
        the locality probe's attention-masked mode (qgt_local_roll). None = all."""
        t = z + self.mv_in(m)[:, None, :]
        b2, aux = self.qa(t, tok_mask=tok_mask)
        loss = aux.get("vq_loss", torch.zeros((), device=z.device))
        if "qlog" in aux and "attn_logit" in aux:
            loss = loss + router_losses(aux["qlog"], aux["attn_logit"])
        return self.w_id * z + self.head(torch.cat([t, b2], -1)), loss

    def roll_bits(self, z0, moves):
        """Free-run: apply T repeatedly, RE-BINARIZING each step. Forward values are
        exactly +-1 (zb + lg - lg.detach() adds an exact zero); gradients pass to the
        logits by straight-through. Returns the bit state at each horizon."""
        out, z = [], z0
        for m in moves:
            lg, _ = self.forward(z, m)
            zb = torch.sign(lg)
            zb = torch.where(zb == 0, torch.ones_like(zb), zb)
            # STE. The parentheses are load-bearing: (lg - lg.detach()) is an EXACT
            # zero, so forward values stay exactly +-1. The associated form
            # (zb + lg) - lg.detach() leaves ~1e-6 rounding residue that downstream
            # exact-equality bit comparisons read as ~25% phantom flips (measured).
            z = zb + (lg - lg.detach())
            out.append(z)
        return out


def _test():
    torch.manual_seed(0)
    from catspace.research.components.encoder.approaches.reach_probability \
        .experiments.graph_move_policy import MV_CAP
    B = 3
    f = QGTField(d=64, layers=2, qa_levels=2)
    sqf, mvf = torch.randn(B, 64, 28), torch.randn(B, MV_CAP, 19)
    mask = torch.zeros(B, MV_CAP, dtype=torch.bool); mask[:, :14] = True
    NN = 64 + MV_CAP
    edges = {"att": (torch.randint(0, B * NN, (150,)), torch.randint(0, B * NN, (150,)))}
    stm = torch.tensor([1, 0, 1])
    logit, logit_ce, z, aux = f(sqf, mvf, edges, mask, stm=stm)
    assert logit.shape == (B, MV_CAP, 3)
    u = set(torch.unique(z).tolist())
    assert u <= {-1.0, 1.0}, f"observables not binary: {sorted(u)[:4]}"
    # permutation invariants (U11's two checks, ported)
    l1, _, _, _ = f(sqf, mvf, edges, mask, stm=torch.ones(B, dtype=torch.long))
    l0, _, _, _ = f(sqf, mvf, edges, mask, stm=torch.zeros(B, dtype=torch.long))
    sD = float((l1[..., 1] - l0[..., 1]).std(dim=1).max())
    sX = float((l1[..., 0] - l0[..., 2]).std(dim=1).max())
    assert sD < 1e-5 and sX < 1e-5, f"permutation broken: D {sD}, X {sX}"
    # roll -> observe -> value: differentiable end to end, bits stay binary after rolling
    c = aux["carrier"]
    dm = [torch.randn(B, 32) for _ in range(3)]
    outs = f.roll_and_read(c, dm, [stm, 1 - stm, stm])
    zr, pv = outs[-1]
    assert set(torch.unique(zr).tolist()) <= {-1.0, 1.0}, "rolled observables not binary"
    (pv.sum() + logit.sum() + aux["loss_aux"]).backward()
    for nm, p_ in (("K", f.roller.K), ("relation", f.base.rel[0]["att"].w.weight),
                   ("readout", f.readout.weight)):
        assert p_.grad is not None and float(p_.grad.abs().sum()) > 0, f"no grad: {nm}"
    print(f"  observables binary at every horizon incl. after 3 rolls; permutation "
          f"invariants exact; gradients reach K, relations, readout; aux losses finite")
    # BitTransition: copy prior at init, exact +-1 free-run, STE gradient flows
    bt = BitTransition(sq_bits=24, mv_bits=32)
    zb0 = torch.sign(torch.randn(B, 64, 24)); zb0[zb0 == 0] = 1.0
    mvs = [torch.sign(torch.randn(B, 32)) for _ in range(3)]
    lg0, _ = bt(zb0, mvs[0])
    agree = float(((lg0 > 0) == (zb0 > 0)).float().mean())
    assert agree == 1.0, f"copy prior broken at init: {agree}"
    outs_b = bt.roll_bits(zb0, mvs)
    ub = set(torch.unique(outs_b[-1].detach()).tolist())
    assert ub <= {-1.0, 1.0}, f"free-run bits not exactly +-1: {sorted(ub)[:4]}"
    zf, _ = f.levels_from_bits(outs_b[-1])
    zf.sum().backward()
    # the zero-init LAST layer receives gradient at init; earlier layers unblock
    # after its first update (gradient to them is proportional to its weights)
    for nm, p_ in (("head[-1]", bt.head[-1].weight), ("w_id", bt.w_id)):
        assert p_.grad is not None and float(p_.grad.abs().sum()) > 0, \
            f"no STE grad through 3 discrete rolls: {nm}"
    # the regression that cost a debugging session: with NON-representable logits
    # (perturbed head), the STE must still emit exactly +-1 -- the un-parenthesized
    # form left ~1e-6 residue that read as ~25% phantom bit-flips downstream
    with torch.no_grad():
        bt.head[-1].weight.add_(1e-4 * torch.randn_like(bt.head[-1].weight))
        ub2 = set(torch.unique(bt.roll_bits(zb0, mvs)[-1]).tolist())
    assert ub2 <= {-1.0, 1.0}, f"STE leaks rounding residue: {sorted(ub2)[:4]}"
    print(f"  BitTransition: identity at init (agree {agree:.2f}); free-run exactly "
          f"+-1 after 3 steps, including under a perturbed (non-representable) head; "
          f"STE gradients reach the zero-init last layer + w_id through the roll")
    print("  QGT FIELD TESTS PASS")


if __name__ == "__main__":
    _test()
