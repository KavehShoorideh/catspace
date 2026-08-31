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


class GraphBitTransition(nn.Module):
    """qgt6 (Kaveh: "is there a way to learn the edge types... not have to code them
    in?"): the transition's neighborhood comes from QuantizedAttention's discrete
    questions -- a QUESTION IS A LEARNED EDGE TYPE (NRI's insight, in-house form:
    (square i) --[question q]--> top-k answering squares). Connectivity is recomputed
    from the CURRENT bit-state every step, so the GNS recompute-connectivity loop is
    built into attention and no hand-coded att/vac/dsc enters the dynamics. The move
    injects only at its from/to squares. FRAME AXIOM: the no-injection baseline pass
    is subtracted, so a square whose features the injection never reaches (through
    the learned attention) copies exactly. WARM START (Kaveh): head 0's attention
    logits are softly supervised toward the true attack adjacency (edge_warm_loss,
    free labels) -- training wheels; heads/questions beyond it are unsupervised, and
    python-chess remains the VALIDATOR (relation_alignment). Copy prior as
    BitTransition (w_id + zero-init out).

    T2-CONDITIONAL (Kaveh 2026-08-29: "I think you're missing the second input,
    which is the stockfish's move ... we have that as a label"): the REPLY enters as
    a second injection at its own from/to squares, so both plies of the move unit
    are given and the target is deterministic board mechanics -- nothing left to
    marginalize, nothing for sign() to round away (the qgt7-10 hang-blindness
    mechanism). The reply carries NO move code: given the post-m1 state, a reply is
    determined by its squares (the piece is already in the bits), so two learned
    marker vectors suffice -- and search can then enumerate replies from legal
    squares alone, no encoder pass per child. Known limit: reply underpromotions
    are indistinguishable from queening (rare; recorded, accepted)."""

    def __init__(self, sq_bits=24, mv_bits=32, d=64, layers=2, n_q=16, heads=4,
                 top_k=4, out_bits=8, w_id_init=0.5, qa_continuous=False,
                 edge_mode="question", direct_out=False):
        super().__init__()
        from catspace.research.tools.training_infra.quantized_attention import (
            QuantizedAttention)
        self.sq_bits = sq_bits
        self.emb = nn.Linear(sq_bits, d)
        self.mv_from = nn.Linear(mv_bits, d)
        self.mv_to = nn.Linear(mv_bits, d)
        if edge_mode == "pair":
            from catspace.research.tools.training_infra.quantized_attention \
                import PairTypedAttention
            self.qa = nn.ModuleList([PairTypedAttention(
                d, out_bits=out_bits, top_k=top_k) for _ in range(layers)])
        else:
            self.qa = nn.ModuleList([QuantizedAttention(
                d, n_q=n_q, n_head=heads, top_k=top_k, out_bits=out_bits,
                continuous_content=qa_continuous) for _ in range(layers)])
        self.proj = nn.ModuleList([nn.Linear(out_bits, d) for _ in range(layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                                nn.Linear(2 * d, d))
                                  for _ in range(layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.out = nn.Linear(d, sq_bits)
        # DIRECT mode (Kaveh 8/30: "get rid of the identity plus delta formulation
        # in the dynamics operator"): the hard gate owns out-of-cone copying, so
        # the operator predicts the next state OUTRIGHT -- no copy prior to hide
        # behind (qgt17's inertness), no baseline pass (one forward, not two).
        # Standard init: zero-init would emit sign(0)=+1 everywhere at start.
        self.direct_out = direct_out
        if not direct_out:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        self.w_id = nn.Parameter(torch.tensor(float(w_id_init)))
        self.rep_from = nn.Parameter(torch.randn(d) * 0.02)
        self.rep_to = nn.Parameter(torch.randn(d) * 0.02)

    def _inj(self, z, m, frm, to, frm2, to2):
        B = z.shape[0]
        ar = torch.arange(B, device=z.device)
        inj = torch.zeros(B, 64, self.emb.out_features, device=z.device,
                          dtype=self.emb.weight.dtype)
        inj[ar, frm] = self.mv_from(m)
        inj[ar, to] = inj[ar, to] + self.mv_to(m)        # frm==to safe
        if frm2 is not None:
            inj[ar, frm2] = inj[ar, frm2] + self.rep_from
            inj[ar, to2] = inj[ar, to2] + self.rep_to
        return inj

    def _encode(self, z, inj, need_attn=False, tok_mask=None):
        h = self.emb(z)
        if inj is not None:
            h = h + inj
        attn0, l_route = None, torch.zeros((), device=z.device)
        for i, qa in enumerate(self.qa):
            b2, aux = qa(self.ln[i](h), tok_mask=tok_mask)
            l_route = l_route + aux.get("vq_loss", 0.0)
            if "qlog" in aux and "attn_logit" in aux:
                l_route = l_route + router_losses(aux["qlog"], aux["attn_logit"])
            if need_attn and i == 0 and "attn_logit" in aux:
                attn0 = aux["attn_logit"]                # (B,H,T,T) pre-softmax
            h = h + self.proj[i](b2)
            h = h + self.ffn[i](h)
        return h, attn0, l_route

    def forward(self, z, m, frm, to, frm2=None, to2=None, need_attn=False,
                tok_mask=None):
        """z (B,64,sq_bits) +-1; m (B,mv_bits); frm/to (B,) long squares; frm2/to2
        (B,) long reply squares (None = legacy marginal mode) ->
        (next-bit logits, head-0 attention logits or None, router aux loss).
        tok_mask (B,64) bool: squares attention may LOOK AT (locality probe)."""
        inj = self._inj(z, m, frm, to, frm2, to2)
        h, attn0, l_route = self._encode(z, inj, need_attn, tok_mask=tok_mask)
        if self.direct_out:                    # one pass, no identity, no delta
            return self.out(h), attn0, l_route
        h0, _, _ = self._encode(z, None, tok_mask=tok_mask)   # same mask both passes:
        # the frame axiom stays exact RELATIVE to the restricted attention
        # parentheses load-bearing (the STE scar's sibling): out(h) - out(h0) is an
        # EXACT zero on untouched squares; (w_id*z + out(h)) - out(h0) is not
        return self.w_id * z + (self.out(h) - self.out(h0)), attn0, l_route

    def touched(self, z, m, frm, to, frm2=None, to2=None):
        """(B,64) bool: squares the injection reached through the learned attention
        -- the move's influence cone, exactly measurable thanks to the frame axiom.
        A true-flipped bit on an UNtouched square is architecturally unpredictable
        (the under-reach instrument, pre-registered failure H2/H3)."""
        inj = self._inj(z, m, frm, to, frm2, to2)
        h, _, _ = self._encode(z, inj)
        h0, _, _ = self._encode(z, None)
        return (h != h0).any(-1)

    @staticmethod
    def edge_warm_loss(attn0, att_target):
        """Soft supervision of head 0 toward the true attack adjacency.
        attn0 (B,H,64,64) pre-softmax; att_target (B,64,64) float 0/1. BCE with a
        positive-class reweight (attacks are ~4% of pairs)."""
        lg = attn0[:, 0].clamp(min=-30, max=30)
        pos = att_target.mean().clamp(min=1e-4)
        w = torch.where(att_target > 0, (1 - pos) / pos, torch.ones_like(att_target))
        return torch.nn.functional.binary_cross_entropy_with_logits(
            lg, att_target, weight=w)

    def roll_bits(self, z0, moves):
        """Free-run: moves = [(m, frm, to), ...] or T2-conditional
        [(m, frm, to, frm2, to2), ...]. Connectivity recomputes from the rolled bits
        inside attention each step -- no external edges needed. Returns the +-1 bit
        state at each horizon (STE exact-zero form)."""
        out, z = [], z0
        for pk in moves:
            m, frm, to = pk[0], pk[1], pk[2]
            frm2, to2 = (pk[3], pk[4]) if len(pk) > 3 else (None, None)
            lg, _, _ = self.forward(z, m, frm, to, frm2=frm2, to2=to2)
            zb = torch.sign(lg)
            zb = torch.where(zb == 0, torch.ones_like(zb), zb)
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
    # GraphBitTransition (QA-learned edges): identity at init; frame axiom holds
    # empirically (delta exactly 0 wherever the injection never reached the
    # features); warm-start loss wired; GNS-in-attention roll exact +-1
    gt = GraphBitTransition(sq_bits=24, mv_bits=32, d=32, layers=2)
    zg = torch.sign(torch.randn(3, 64, 24)); zg[zg == 0] = 1.0
    mg = torch.sign(torch.randn(3, 32))
    frm = torch.tensor([0, 8, 16]); to = torch.tensor([1, 9, 17])
    lg0, attn0, l_r0 = gt(zg, mg, frm, to, need_attn=True)
    ag = float(((lg0 > 0) == (zg > 0)).float().mean())
    assert ag == 1.0, f"graph transition not identity at init: {ag}"
    assert attn0 is not None and attn0.shape[-2:] == (64, 64), "attn logits missing"
    assert torch.isfinite(torch.as_tensor(l_r0)), "router aux not finite"
    with torch.no_grad():                        # break identity, test the frame axiom
        gt.out.weight.add_(3.0 * torch.randn_like(gt.out.weight))
        ar3 = torch.arange(3)
        inj = torch.zeros(3, 64, gt.emb.out_features)
        inj[ar3, frm] = gt.mv_from(mg); inj[ar3, to] = inj[ar3, to] + gt.mv_to(mg)
        h1, _, _ = gt._encode(zg, inj)
        h0, _, _ = gt._encode(zg, None)
        untouched = (h1 == h0).all(-1)                            # (3,64)
        lg1, _, _ = gt(zg, mg, frm, to)
        delta = lg1 - gt.w_id * zg
        assert bool(untouched.any()), "frame-axiom test vacuous: no untouched square"
        assert bool((~untouched).any()), "frame-axiom test vacuous: nothing touched"
        du = float(delta[untouched].abs().max())
        assert du == 0.0, f"untouched squares got nonzero delta: {du}"
    att_t = torch.zeros(3, 64, 64); att_t[:, 0, 2] = 1.0; att_t[:, 5, 7] = 1.0
    lw = GraphBitTransition.edge_warm_loss(attn0.detach().requires_grad_(True), att_t)
    assert torch.isfinite(lw) and float(lw) > 0, "warm-start loss not finite"
    lw2 = GraphBitTransition.edge_warm_loss(gt(zg, mg, frm, to, need_attn=True)[1],
                                            att_t)
    g_k = torch.autograd.grad(lw2, gt.qa[0].k_proj.weight, allow_unused=True)[0]
    assert g_k is not None and float(g_k.abs().sum()) > 0, \
        "warm-start supervision does not reach head-0 keys"
    # PAIR-TYPED edge mode: same identity/frame-axiom/roll battery
    gp = GraphBitTransition(sq_bits=24, mv_bits=32, d=32, layers=2,
                            edge_mode="pair")
    lp0, ap0, _ = gp(zg, mg, frm, to, need_attn=True)
    agp = float(((lp0 > 0) == (zg > 0)).float().mean())
    assert agp == 1.0, f"pair transition not identity at init: {agp}"
    assert ap0 is not None and ap0.shape[-2:] == (64, 64), "pair attn logits missing"
    with torch.no_grad():
        gp.out.weight.add_(3.0 * torch.randn_like(gp.out.weight))
        tchp = gp.touched(zg, mg, frm, to)
        lp1, _, _ = gp(zg, mg, frm, to)
        dp = lp1 - gp.w_id * zg
        if bool((~tchp).any()):
            assert float(dp[~tchp].abs().max()) == 0.0, "pair frame axiom broken"
    zp = gp.roll_bits(zg, [(mg, frm, to)])[0]
    assert set(torch.unique(zp.detach()).tolist()) <= {-1.0, 1.0}
    zp.sum().backward()
    assert gp.qa[0].route.weight.grad is not None and \
        float(gp.qa[0].route.weight.grad.abs().sum()) > 0, "no grad to pair router"
    u_ids = int(gp.qa[0].last["question"].unique().numel())
    print(f"  PairTypedAttention: identity at init; frame axiom holds; roll +-1; "
          f"grads reach the FSQ router; {u_ids} distinct pair codes at init")
    outs_g = gt.roll_bits(zg, [(mg, frm, to)] * 3)
    ug = set(torch.unique(outs_g[-1].detach()).tolist())
    assert ug <= {-1.0, 1.0}, "graph free-run bits not exactly +-1"
    outs_g[-1].sum().backward()
    for nm, p_ in (("out", gt.out.weight), ("w_id", gt.w_id),
                   ("qa q_select", gt.qa[0].q_select.weight)):
        assert p_.grad is not None and float(p_.grad.abs().sum()) > 0, f"no grad: {nm}"
    # DIRECT mode: no identity at init (by design), roll stays +-1, grads flow
    gd = GraphBitTransition(sq_bits=24, mv_bits=32, d=32, layers=2,
                            edge_mode="pair", direct_out=True)
    ld, _, _ = gd(zg, mg, frm, to)
    assert float(ld.abs().mean()) > 1e-4, "direct out degenerate at init"
    zd = gd.roll_bits(zg, [(mg, frm, to)])[0]
    assert set(torch.unique(zd.detach()).tolist()) <= {-1.0, 1.0}
    zd.sum().backward()
    assert gd.out.weight.grad is not None and \
        float(gd.out.weight.grad.abs().sum()) > 0, "no grad in direct mode"
    print("  DIRECT dynamics: single-pass next-state prediction; roll +-1; grads OK")
    # T2-conditional (the reply input): frame axiom stays exact with BOTH
    # injections; the reply markers actually reach the output; grads flow to them
    frm2 = torch.tensor([40, 48, 56]); to2 = torch.tensor([41, 49, 57])
    with torch.no_grad():
        inj2 = gt._inj(zg, mg, frm, to, frm2, to2)
        h2, _, _ = gt._encode(zg, inj2)
        h0b, _, _ = gt._encode(zg, None)
        unt2 = (h2 == h0b).all(-1)
        lg2, _, _ = gt(zg, mg, frm, to, frm2=frm2, to2=to2)
        d2 = (lg2 - gt.w_id * zg)
        assert bool(unt2.any()), "conditional frame-axiom test vacuous"
        assert float(d2[unt2].abs().max()) == 0.0, "conditional frame axiom broken"
        assert float((lg2 - lg1).abs().max()) > 0, "reply injection inert"
    gt.rep_from.grad = None
    zc = gt.roll_bits(zg, [(mg, frm, to, frm2, to2)])[0]
    zc.sum().backward()
    assert gt.rep_from.grad is not None and \
        float(gt.rep_from.grad.abs().sum()) > 0, "no grad to reply marker"
    print(f"  GraphBitTransition (QA edges): identity at init; frame axiom EXACT on "
          f"{int(untouched.sum())}/{3*64} untouched squares under a perturbed out; "
          f"warm-start loss reaches head-0 keys; GNS-in-attention roll exactly +-1; "
          f"grads reach out/w_id/question-selector")
    print("  QGT FIELD TESTS PASS")


if __name__ == "__main__":
    _test()
