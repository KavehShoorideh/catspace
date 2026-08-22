#!/usr/bin/env python
"""graph_concept_field.py -- the graph encoder UNDER the concepts; concepts predict the move
and their own future. (Kaveh 2026-08-19)

His spec, verbatim: "use the graph neural network as an underlying mechanism to encode the
board, then run some sort of transformer layer on it, and predict a latent state made up of
concepts where some concepts are moving fast, some slow, with a loss term that makes the
concepts code different things. And from the concepts, predict a future set of concepts and a
head that predicts the stockfish move -- that way we get some sibling resolution."

MAPPING, component by component:
    GNN + transformer      the gmp encoder: L blocks of (per-relation message passing over
                           att/src/dst/mut/VAC/DSC -> full MultiheadAttention -> FFN). The
                           deletion relations are new -- v1 carried only the added edges, and
                           "some edges are deleted" is half the move's derivative.
    latent concept state   VQ on top of the graph: 64x256 square codes from square-nodes,
                           16x32 global heads from the pooled board. kmeans_init +
                           threshold_ema_dead_code, the collapse fix, non-negotiable.
    slow vs fast           first 8 global heads SLOW via VAMP-2 at lag 8 (vamp2_score in
                           losses.py, tested: constant features score 0 -- collapse cannot
                           fake slowness, unlike the refuted MSE persistence that nulled cef5).
                           Last 8 heads fast, unconstrained in time.
    code different things  VAMP's whitening decorrelates the slow block structurally;
                           vicreg_covariance (tested; drove the measured 77%% head redundancy
                           to zero in cef4+) covers the fast block.
    future concepts        dynamics head: (zq_t, played move) -> the child position's own codes
                           (targets detached; child encoded from the slim child block, never
                           through any transformer trunk).
    stockfish move         PL top-1 over move logits = MLP(concepts-as-context (+) move-node).
                           The concat is deliberate: a concepts-only head cannot see moves, a
                           move-node-only head bypasses concepts; jointly the gradient forces
                           the CONCEPTS to carry choice-relevant context -- his "that way we
                           get some sibling resolution".

Also kept: the factorized resolution head (the committor every downstream consumer reads) and
a small current-material head (representation sanity; w kept low after the cef2 lesson).

    .venv/bin/python -m ...graph_concept_field --graphs data/derived/move_graphs_v2_*.npz
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from catspace.research.tools.training_infra.losses import (
    effective_number_weights, factorized_resolution_ce, vamp2_score, vicreg_covariance, cross_head_decorrelation)
from catspace.research.components.encoder.approaches.reach_probability.experiments \
    .concept_capture_field import factor_vocab
from catspace.research.components.encoder.approaches.reach_probability.experiments \
    .graph_move_policy import MV_CAP, N_NODES, Relation

RELS = ("att", "src", "dst", "mut", "vac", "dsc")
# gcf3 kinematic relations: STATIC board geometry, independent of occupancy -- the wiring a
# piece could ever move along. file/rank/diag are full lines (a rook/bishop hears its whole
# line even through blockers; occupancy is what the learned messages must gate), knight is
# the jump graph, king the step graph ("manhattan" + single diagonals).
KIN_RELS = ("kfile", "krank", "kdiag", "kadiag", "kknight", "kking")
# file-node relations: 8 file-nodes aggregate their squares (per-file pawn structure lives
# here -- the feature gcf1/gcf2 measurably lacked) and broadcast back.
FN_RELS = ("fsq", "sqf")
N_COUNT, N_CLASS = 9, 12


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -g


def grad_reverse(x):
    return _GradReverse.apply(x)


def kin_templates():
    """-> {rel: (src, dst) LongTensors} over 64 local square indices, both directions."""
    fr = torch.arange(64); f, r = fr % 8, fr // 8
    df = f[:, None] - f[None, :]; dr = r[:, None] - r[None, :]
    ad, adr = df.abs(), dr.abs()
    ident = torch.eye(64, dtype=torch.bool)
    masks = {
        "kfile": (df == 0) & ~ident,
        "krank": (dr == 0) & ~ident,
        "kdiag": (df == dr) & ~ident,
        "kadiag": (df == -dr) & ~ident,
        "kknight": ((ad == 1) & (adr == 2)) | ((ad == 2) & (adr == 1)),
        "kking": (ad <= 1) & (adr <= 1) & ~ident,
    }
    return {k: tuple(torch.nonzero(m, as_tuple=True)) for k, m in masks.items()}


class GraphConceptField(nn.Module):
    def __init__(self, d=128, layers=3, attn_heads=4, g_heads=16, g_codes=32, d_code=32,
                 sq_codes=256, d_fine=32, slow_heads=8, hidden=512, n_res=414, n_eg=397,
                 cls_reach=None, cls_eg=None, cls_out=None,
                 edge_codes=0, move_codes=0, gps_parallel=False, attn_bias=False,
                 agg_max=False, kin=False, file_nodes=False, phase_head=False,
                 threat_head=False, elo_cond=False, mh_vq=False):
        super().__init__()
        from vector_quantize_pytorch import VectorQuantize
        self.edge_codes, self.move_codes = edge_codes, move_codes
        self.gps_parallel, self.use_bias, self.agg_max = gps_parallel, attn_bias, agg_max
        self.kin, self.file_nodes = kin, file_nodes
        self.phase_head, self.threat_head = phase_head, threat_head
        self.n_nodes = N_NODES + (8 if file_nodes else 0)
        self.d, self.g_heads, self.d_code = d, g_heads, d_code
        self.sq_codes, self.d_fine, self.slow_heads = sq_codes, d_fine, slow_heads
        self.rels = RELS + (KIN_RELS if kin else ()) + (FN_RELS if file_nodes else ())
        self.sq_in = nn.Linear(13 + 7 + 6 + 2, d)
        self.mv_in = nn.Linear(6 + 4 + 5 + 4, d)
        self.rel = nn.ModuleList([nn.ModuleDict({r: Relation(d) for r in self.rels})
                                  for _ in range(layers)])
        if kin:
            for name, (ks, kd) in kin_templates().items():
                self.register_buffer(f"kin_s_{name}", ks.long())
                self.register_buffer(f"kin_d_{name}", kd.long())
        if file_nodes:
            self.file_emb = nn.Parameter(torch.randn(8, d) * 0.02)
            fr = torch.arange(64)
            self.register_buffer("fn_sq", fr.long())            # square index
            self.register_buffer("fn_file", (fr % 8).long())    # its file-node (local 0-7)
        if phase_head:
            # DESIGNATED phase head (Kaveh 2026-08-20: castling MAY own one head; phase owns
            # head 0 -- with 32 codes over 4 buckets there is room for "different codes for
            # the same phase"). The ADVERSARY pushes phase OUT of heads 1..15: gradient
            # reversal confined to the projection slices (trunk input detached), so the
            # trunk keeps phase for the move head while the other concept heads must encode
            # something else.
            self.phase_cls = nn.Linear(d_code, 4)
            self.phase_adv = nn.Sequential(nn.Linear((g_heads - 1) * d_code, 128), nn.GELU(),
                                           nn.Linear(128, 4))
        self.elo_cond = elo_cond
        if elo_cond:
            # Kaveh 2026-08-20: "incorporate elo into the concept heads" -- (mover, opp)
            # normalized ratings enter the trunk as a node-broadcast embedding (the codes
            # may be rating-dependent: a 1500's "mobility" is not a 2200's), and an
            # elo READOUT from the concept state measures how much rating the vocabulary
            # carries (the effective-Elo probe). elo=None at inference = zeros = the
            # 1500 population mean.
            self.elo_in = nn.Linear(2, d)
            self.elo_head = nn.Linear(0, 2)   # re-bound after d_state known (below)
        if threat_head:
            # threat -> execution (Kaveh: "how often is it actually taken... when is a good
            # time to execute"): per-square P(exec within 8 | threat), supervised only on
            # censused threat squares.
            self.te_head = nn.Linear(d, 1)
        if agg_max:      # PNA-lite: mean+max concatenated per relation (GraphGPS best MPNNs)
            self.rel_max = nn.ModuleList([nn.ModuleDict({r: Relation(d) for r in self.rels})
                                          for _ in range(layers)])
            self.agg_proj = nn.ModuleList([nn.Linear(2 * d, d) for _ in range(layers)])
        if attn_bias:    # Graphormer spatial encoding: learnable scalar per pair-relation bucket
            self.n_buckets = 17 if file_nodes else 14   # +file<->own-sq, file<->file, file<->other
            self.bias_emb = nn.Parameter(torch.zeros(attn_heads, self.n_buckets))
            fr = torch.arange(64); f, r = fr % 8, fr // 8
            df = (f[:, None] - f[None, :]).abs(); dr = (r[:, None] - r[None, :]).abs()
            bt = torch.full((64, 64), 7, dtype=torch.long)
            bt[(df == 1) & (dr == 2)] = 5; bt[(df == 2) & (dr == 1)] = 5   # knight
            bt[(df <= 1) & (dr <= 1)] = 6                                   # king-adjacent
            bt[(df == dr) & (df > 0)] = 3                                   # diagonal
            bt[(df == 0) & (dr > 0)] = 1                                    # same file
            bt[(dr == 0) & (df > 0)] = 2                                    # same rank
            bt.fill_diagonal_(0)
            self.register_buffer("sq_bucket", bt)
        self.ln_l = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, attn_heads, batch_first=True)
                                   for _ in range(layers)])
        self.ln_g = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                                nn.Linear(2 * d, d)) for _ in range(layers)])
        self.ln_f = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])

        vq_kw = dict(decay=0.9, commitment_weight=0.25, kmeans_init=True,
                     threshold_ema_dead_code=2)
        self.g_proj = nn.Linear(d, g_heads * d_code)
        self.mh_vq = mh_vq
        if mh_vq:
            # GROUPED multi-head VQ: 8 groups x (g_heads/8) heads each. One monolithic
            # 512-head module OOMs at construction (lib-internal init buffers); groups
            # keep per-head codebooks, bound memory, and cut the python loop 512 -> 8
            # (the loop was ~10s/step at the 512-concept scale).
            self.vq_groups = 8
            gh = g_heads // self.vq_groups
            assert gh * self.vq_groups == g_heads
            self.vq_mh = nn.ModuleList([
                VectorQuantize(dim=gh * d_code, heads=gh,
                               separate_codebook_per_head=True,
                               codebook_size=g_codes, **vq_kw)
                for _ in range(self.vq_groups)])
        else:
            self.vq = nn.ModuleList([VectorQuantize(dim=d_code, codebook_size=g_codes,
                                                    **vq_kw)
                                     for _ in range(g_heads)])
        self.sq_proj = nn.Linear(d, d_fine)
        self.sq_vq = VectorQuantize(dim=d_fine, codebook_size=sq_codes, **vq_kw)
        d_state = g_heads * d_code + 64 * d_fine
        if edge_codes:
            # EDGE CONCEPTS (Kaveh 2026-08-19: "global, node, EDGE concepts"). An att edge's
            # embedding is its endpoint pair; the VQ code is the tactic-type; the STATE carries
            # the histogram of codes -- a bag-of-tactics, fixed width, interpretable. The
            # dynamics head then predicts the CHILD's histogram: "some tactics are added and
            # some deleted" as a first-class training target.
            self.edge_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 32))
            self.edge_vq = VectorQuantize(dim=32, codebook_size=edge_codes, decay=0.9,
                                          commitment_weight=0.25, kmeans_init=True,
                                          threshold_ema_dead_code=2)
            d_state = d_state + edge_codes
        if move_codes:
            # MOVE CONCEPTS: annotative, not bottlenecking -- the score reads raw (+) quantized,
            # so the fine ranking keeps its resolution while the codes learn move types.
            self.mv_vq_proj = nn.Linear(d, 32)
            self.mv_vq = VectorQuantize(dim=32, codebook_size=move_codes, decay=0.9,
                                        commitment_weight=0.25, kmeans_init=True,
                                        threshold_ema_dead_code=2)
        self.d_state = d_state
        if elo_cond:
            self.elo_head = nn.Linear(d_state, 2)

        self.move_head = nn.Sequential(
            nn.Linear(d_state + d + (32 if move_codes else 0), hidden), nn.GELU(),
            nn.Linear(hidden, 1))
        self.mv_emb = nn.Sequential(nn.Linear(6 + 4 + 5 + 4, 64), nn.GELU())
        self.dyn = nn.Sequential(nn.Linear(d_state + 64, hidden), nn.GELU(),
                                 nn.Linear(hidden, g_heads * g_codes + 64 * sq_codes))
        if edge_codes:
            self.dyn_edge = nn.Sequential(nn.Linear(d_state + 64, hidden), nn.GELU(),
                                          nn.Linear(hidden, edge_codes))
        self.now_head = nn.Sequential(nn.Linear(d_state, hidden), nn.GELU(),
                                      nn.Linear(hidden, N_CLASS * N_COUNT))
        self.res_body = nn.Sequential(nn.Linear(d_state, hidden), nn.GELU())
        self.res_reach = nn.Linear(hidden, 3)
        self.res_eg = nn.Linear(hidden, n_eg)
        self.res_out = nn.Linear(hidden, 3)
        for nm, v in (("cls_reach", cls_reach), ("cls_eg", cls_eg), ("cls_out", cls_out)):
            self.register_buffer(nm, torch.as_tensor(
                v if v is not None else np.zeros(n_res, np.int64)).long())
        # slow-block projection for VAMP-2: 32-d so a few-hundred-pair batch supports the
        # covariances (rank-deficiency is the caller's bug, per the loss docstring)
        self.vamp_proj = nn.Linear(slow_heads * d_code, 32, bias=False)

    def _attn_bias(self, B, mvf, mv_mask, dev):
        """(B*heads, N, N) additive bias from static square geometry + move src/dst buckets."""
        nb = torch.full((B, self.n_nodes, self.n_nodes), 7, dtype=torch.long, device=dev)
        nb[:, :64, :64] = self.sq_bucket[None]
        # move-node buckets: 8 = its source square, 9 = its dest square, 10 = other sq<->mv,
        # 11/12/13 = move<->move same-from/same-to/other
        fr = (mvf[:, :, 6:8] * 7).round().long()      # from (file,rank) scaled back
        to = (mvf[:, :, 8:10] * 7).round().long()
        f_sq = fr[:, :, 1] * 8 + fr[:, :, 0]
        t_sq = to[:, :, 1] * 8 + to[:, :, 0]
        nb[:, :64, 64:] = 10; nb[:, 64:, :64] = 10
        bidx = torch.arange(B, device=dev)[:, None]
        mv_i = torch.arange(MV_CAP, device=dev)[None, :]
        nb[bidx, f_sq, mv_i + 64] = 8; nb[bidx, mv_i + 64, f_sq] = 8
        nb[bidx, t_sq, mv_i + 64] = 9; nb[bidx, mv_i + 64, t_sq] = 9
        mm_same_f = f_sq[:, :, None] == f_sq[:, None, :]
        mm_same_t = t_sq[:, :, None] == t_sq[:, None, :]
        mm = torch.where(mm_same_f, 11, torch.where(mm_same_t, 12, 13))
        nb[:, 64:64 + MV_CAP, 64:64 + MV_CAP] = mm
        if self.file_nodes:
            F0 = 64 + MV_CAP
            nb[:, F0:, :] = 16; nb[:, :, F0:] = 16                       # file<->anything
            fi = torch.arange(8, device=dev)
            nb[:, (F0 + fi)[:, None], (F0 + fi)[None, :]] = 15           # file<->file
            sq = torch.arange(64, device=dev)
            nb[:, F0 + (sq % 8), sq] = 14; nb[:, sq, F0 + (sq % 8)] = 14  # file<->own squares
        bias = self.bias_emb[:, nb]                    # (heads, B, N, N)
        return bias.permute(1, 0, 2, 3).reshape(-1, self.n_nodes, self.n_nodes)

    def _static_edges(self, B, dev):
        """kinematic + file-node edge lists, offset for a B-batch (they are the same for
        every position -- geometry, not occupancy)."""
        out = {}
        off = (torch.arange(B, device=dev) * self.n_nodes)[:, None]
        if self.kin:
            for name in KIN_RELS:
                ks = getattr(self, f"kin_s_{name}"); kd = getattr(self, f"kin_d_{name}")
                out[name] = ((ks[None] + off).reshape(-1), (kd[None] + off).reshape(-1))
        if self.file_nodes:
            F0 = 64 + MV_CAP
            fs = self.fn_sq; fd = F0 + self.fn_file
            out["fsq"] = ((fs[None] + off).reshape(-1), (fd[None] + off).reshape(-1))
            out["sqf"] = ((fd[None] + off).reshape(-1), (fs[None] + off).reshape(-1))
        return out

    def backbone(self, sqf, mvf, edges, mv_mask, elo=None):
        B = sqf.shape[0]
        h = torch.cat([self.sq_in(sqf), self.mv_in(mvf)], 1)
        if self.elo_cond:
            e = elo if elo is not None else torch.zeros(B, 2, device=sqf.device)
            h = h + self.elo_in(e)[:, None, :]
        if self.file_nodes:
            h = torch.cat([h, self.file_emb[None].expand(B, 8, self.d)], 1)
        pad = torch.cat([torch.zeros(B, 64, dtype=torch.bool, device=sqf.device), ~mv_mask], 1)
        if self.file_nodes:
            pad = torch.cat([pad, torch.zeros(B, 8, dtype=torch.bool, device=sqf.device)], 1)
        if self.kin or self.file_nodes:
            edges = {**edges, **self._static_edges(B, sqf.device)}
        abias = self._attn_bias(B, mvf, mv_mask, sqf.device) if self.use_bias else None

        def local(x, i):
            xf = x.reshape(B * self.n_nodes, self.d)
            agg = 0.0
            for r in self.rels:
                sd = edges.get(r)
                if sd is not None and len(sd[0]):
                    agg = agg + self.rel[i][r](xf, sd[0], sd[1], B * self.n_nodes)
            if self.agg_max:
                mx = 0.0
                for r in self.rels:
                    sd = edges.get(r)
                    if sd is not None and len(sd[0]):
                        m = self.rel_max[i][r].w(xf[sd[0]])
                        out = torch.zeros(B * self.n_nodes, self.d, device=x.device, dtype=m.dtype)
                        out.index_reduce_(0, sd[1], m, "amax", include_self=False)
                        mx = mx + out
                agg = self.agg_proj[i](torch.cat(
                    [agg if torch.is_tensor(agg) else torch.zeros_like(mx),
                     mx if torch.is_tensor(mx) else torch.zeros_like(agg)], -1))
            return (agg.reshape(B, self.n_nodes, self.d) if torch.is_tensor(agg) else
                    torch.zeros_like(x))

        for i in range(len(self.rel)):
            if self.gps_parallel:
                # GPS canonical form, pre-LN: parallel MPNN + attention on the same input,
                # summed into the residual stream, then FFN. (GraphGPS Eq. 2-4; Graphormer's
                # pre-LN placement.)
                a_l = local(self.ln_l[i](h), i)
                hg = self.ln_g[i](h)
                a_g, _ = self.attn[i](hg, hg, hg, key_padding_mask=pad,
                                      attn_mask=abias, need_weights=False)
                h = h + a_l + a_g
                h = h + self.ffn[i](self.ln_f[i](h))
            else:
                h = self.ln_l[i](h + local(h, i))
                a, _ = self.attn[i](h, h, h, key_padding_mask=pad,
                                    attn_mask=abias, need_weights=False)
                h = self.ln_g[i](h + a)
                h = self.ln_f[i](h + self.ffn[i](h))
        return h                                              # (B, 112, d)

    def encode(self, sqf, mvf, edges, mv_mask, want_edge_ids=False, elo=None):
        """-> zq (B,d_state) in [0,1], g_ids (B,H), sq_ids (B,64), vq_loss, h_mv, pre_g
        [, edge_hist (B,E)]"""
        B = sqf.shape[0]
        h = self.backbone(sqf, mvf, edges, mv_mask, elo=elo)
        h_sq, h_mv = h[:, :64], h[:, 64:64 + MV_CAP]
        pre_g = self.g_proj(h_sq.mean(1)).view(B, self.g_heads, self.d_code)
        # stashed for the aux heads (phase adversary reads projections of a DETACHED trunk,
        # so reversal shapes only the projection slices; te head reads square states)
        self.last_h_sq = h_sq
        if self.phase_head:
            self.last_pre_adv = self.g_proj(h_sq.mean(1).detach()).view(
                B, self.g_heads, self.d_code)[:, 1:]
        if self.mh_vq:
            gh = self.g_heads // self.vq_groups
            qs, ids, vl = [], [], 0.0
            for gi in range(self.vq_groups):
                blk = pre_g[:, gi * gh:(gi + 1) * gh].reshape(B, 1, -1)
                q_g, idx_g, l_g = self.vq_mh[gi](blk)
                qs.append(q_g.reshape(B, -1))
                ids.extend(idx_g.reshape(B, gh).unbind(-1))
                vl = vl + l_g
        else:
            qs, ids, vl = [], [], 0.0
            for i in range(self.g_heads):
                q, idx, l = self.vq[i](pre_g[:, i])
                qs.append(q); ids.append(idx); vl = vl + l
        q_sq, sq_ids, vl2 = self.sq_vq(self.sq_proj(h_sq))
        parts = qs + [q_sq.reshape(B, -1)]
        vl = vl + vl2
        hist = None
        if self.edge_codes:
            sd = edges.get("att")
            hf = h.reshape(B * self.n_nodes, self.d)
            if sd is not None and len(sd[0]):
                e = self.edge_mlp(torch.cat([hf[sd[0]], hf[sd[1]]], -1))
                _, e_ids, vl3 = self.edge_vq(e)
                vl = vl + vl3
                # stashed for live inspection (kitty edge-concepts panel)
                self.last_edge_ids, self.last_edge_sd = e_ids, sd
                # bucket edges by their DST node's sample, then pool per sample
                hist = torch.zeros(B, self.edge_codes, device=h.device)
                samp = (sd[1] // self.n_nodes).long()
                hist.index_put_((samp, e_ids), torch.ones_like(e_ids, dtype=hist.dtype),
                                accumulate=True)
                hist = hist / hist.sum(-1, keepdim=True).clamp(min=1)
            else:
                hist = torch.zeros(B, self.edge_codes, device=h.device)
            parts.append(hist)
        zq = torch.cat(parts, -1)
        lo = zq.min(-1, keepdim=True).values; hi = zq.max(-1, keepdim=True).values
        zq = (zq - lo) / (hi - lo).clamp(min=1e-5)
        if want_edge_ids:
            return zq, torch.stack(ids, 1), sq_ids, vl, h_mv, pre_g, hist
        return zq, torch.stack(ids, 1), sq_ids, vl, h_mv, pre_g

    def move_logits(self, zq, h_mv, mv_mask):
        parts = [zq[:, None, :].expand(-1, h_mv.shape[1], -1), h_mv]
        if self.move_codes:
            q_mv, _, vl_mv = self.mv_vq(self.mv_vq_proj(h_mv))
            parts.append(q_mv)
            self._mv_vq_loss = vl_mv
        x = torch.cat(parts, -1)
        return self.move_head(x).squeeze(-1).masked_fill(~mv_mask, float("-inf"))

    def predict_resolution(self, zq):
        b = self.res_body(zq)
        return self.res_reach(b), self.res_eg(b), self.res_out(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--n-human", type=int, default=0,
                    help="human (lichess) games; with --games 0 the corpus is human-only")
    ap.add_argument("--res-mode", default="tb", choices=["tb", "outcome"],
                    help="outcome: no tablebase vocab -- the resolution head trains as a "
                         "3-way White-POV game-result committor, censored (time-forfeit) "
                         "games excluded from that loss")
    ap.add_argument("--elo", type=int, default=0,
                    help="elo conditioning into the trunk + elo readout head")
    ap.add_argument("--w-elo", type=float, default=0.1)
    ap.add_argument("--g-heads", type=int, default=16)
    ap.add_argument("--g-codes", type=int, default=32)
    ap.add_argument("--d-code", type=int, default=32)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--vamp-batch", type=int, default=128,
                    help="also the decor batch: noise floor is 32/B, so keep >= 128")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--slow-heads", type=int, default=8)
    ap.add_argument("--edge-codes", type=int, default=0, help="0 off; 128 = edge concepts on")
    ap.add_argument("--move-codes", type=int, default=0, help="0 off; 64 = move concepts on")
    ap.add_argument("--gps-parallel", type=int, default=0)
    ap.add_argument("--attn-bias", type=int, default=0)
    ap.add_argument("--agg-max", type=int, default=0)
    ap.add_argument("--kin", type=int, default=0, help="kinematic square relations")
    ap.add_argument("--file-nodes", type=int, default=0, help="8 file-nodes (pawn structure)")
    ap.add_argument("--phase-head", type=int, default=0,
                    help="designated phase head 0 + confined GRL adversary on heads 1..15")
    ap.add_argument("--threat-head", type=int, default=0, help="per-square exec-within-8 head")
    ap.add_argument("--w-phase", type=float, default=0.3)
    ap.add_argument("--w-adv", type=float, default=0.2)
    ap.add_argument("--w-decor", type=float, default=0.5,
                    help="cross-head decorrelation (the castling-collapse fix)")
    ap.add_argument("--w-te", type=float, default=0.3)
    ap.add_argument("--w-edge-dyn", type=float, default=0.5,
                    help="child edge-histogram prediction: tactics added and deleted")
    ap.add_argument("--w-move", type=float, default=2.0)
    ap.add_argument("--w-dyn", type=float, default=1.0)
    ap.add_argument("--w-res", type=float, default=1.0)
    ap.add_argument("--w-now", type=float, default=0.25)
    ap.add_argument("--w-slow", type=float, default=1.0, help="on -vamp2/32, so O(1)")
    ap.add_argument("--w-dec", type=float, default=0.5)
    ap.add_argument("--cb-beta", type=float, default=0.999)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="", help="checkpoint to continue from (Adam resets)")
    ap.add_argument("--out", default="artifacts/experiments/gcf1")
    args = ap.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    dev = args.device

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_move_policy import featurize
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths
    from catspace.research.tools.training_infra.train.scaffold import (
        TrainConfig, standard_train)

    t0 = time.time()
    z = np.load(args.graphs)
    D = {k: torch.from_numpy(np.ascontiguousarray(z[k])) for k in z.files}
    N = len(D["p_row"])
    tr = T.build(n_human=args.n_human, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    n = len(tr.tok)
    if args.res_mode == "tb":
        lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
        vocab = [str(x) for x in lz["vocab"]]
        res_cls = lz["tb_class"].astype(np.int64)
        cr, ce, co, egs = factor_vocab(vocab)
    else:
        # OUTCOME mode (human corpus, no oracle anywhere): the resolution head becomes a
        # 3-way White-POV game-result committor; time-forfeit (censored) games are
        # excluded from that loss -- the board does not explain their ending.
        vocab, egs = ["W", "D", "L"], ["-"]
        res_cls = np.zeros(1, np.int64)
        cr = np.zeros(3, np.int64); ce = np.zeros(3, np.int64)
        co = np.arange(3, dtype=np.int64)
    gor = tr.game_of_row()
    rs = np.random.default_rng(args.seed); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True
    is_te = test_g[gor[D["p_row"].numpy()]]
    tr_i = np.flatnonzero(~is_te); te_i = np.flatnonzero(is_te)
    v8 = D["valid8"].numpy().astype(bool)
    tr8 = np.flatnonzero(~is_te & v8)
    RES = torch.from_numpy(res_cls)
    r_reach = np.where(res_cls >= 0, cr[res_cls.clip(0)], -1).astype(np.int64)
    r_eg = np.where(res_cls >= 0, ce[res_cls.clip(0)], -1).astype(np.int64)
    r_out = np.where(res_cls >= 0, co[res_cls.clip(0)], -1).astype(np.int64)
    p_np = D["p_row"].numpy()
    if args.res_mode == "tb":
        cnt = np.bincount(r_eg[p_np[tr_i]][r_eg[p_np[tr_i]] >= 0], minlength=len(egs))
        eg_w = effective_number_weights(torch.from_numpy(cnt), beta=args.cb_beta).to(dev)
    else:
        eg_w = None
        res_w = np.asarray(tr.result)[gor]                       # per-row game result
        OUTW = torch.from_numpy((1 - res_w[p_np]).astype(np.int64))   # W=0 D=1 L=2 White-POV
        term = np.asarray(tr.term)
        CEN = torch.from_numpy(term[gor[p_np]] == -2)            # TERM_TIME only: the board
                                                                  # does not explain a flag
        print(f"[gcf] outcome mode: {int(CEN.sum()):,}/{len(CEN):,} rows censored "
              f"(time forfeit)", flush=True)
    EL = None
    if args.elo:
        el_g = np.asarray(tr.elo, np.float32)                    # (games, 2) white/black
        el_rows = el_g[gor[p_np]]
        stm_rows = D["gl"][:, 0].numpy().astype(bool)
        mover = np.where(stm_rows, el_rows[:, 0], el_rows[:, 1])
        opp = np.where(stm_rows, el_rows[:, 1], el_rows[:, 0])
        from catspace.research.components.encoder.approaches.reach_probability.src \
            .trajectories import normalise_elo
        EL = torch.from_numpy(np.stack([normalise_elo(mover), normalise_elo(opp)], 1))
        print(f"[gcf] elo conditioning: mover mean {mover.mean():.0f} "
              f"range {mover.min():.0f}-{mover.max():.0f}", flush=True)
    print(f"[gcf] {N:,} graphs | train {len(tr_i):,} / held-out {len(te_i):,} | "
          f"lag-8 pairs {len(tr8):,} [{time.time()-t0:.0f}s]", flush=True)

    PH = TE = None
    if args.phase_head:
        from catspace.research.tools.training_infra.concepts_reference import TOK_PIECES
        ids = D["sq"][:, :, 0].numpy()
        minor_ids = [k + 1 for k, (pt, c) in enumerate(TOK_PIECES) if pt not in (1, 6)]
        npc = np.isin(ids, minor_ids).sum(1)
        PH = torch.from_numpy(np.digitize(npc, [4, 8, 12]))     # 0-3|4-7|8-11|12+ -> 4 buckets
        print(f"[gcf] phase buckets: {np.bincount(PH.numpy(), minlength=4).tolist()}", flush=True)
    if args.threat_head:
        import glob as _g
        from catspace.research.tools.training_infra.threat_exec_join import join_threat_exec
        te_path = sorted(_g.glob(str(paths.derived(f"threat_exec_k8_*_{n}.npz"))))[-1]
        TE = torch.from_numpy(join_threat_exec(p_np, te_path))
        n_ev = int((TE >= 0).sum())
        te_pw = torch.tensor((TE == 0).sum().item() / max(1, (TE == 1).sum().item()),
                             device=dev)
        print(f"[gcf] threat/exec: {n_ev:,} events, pos_weight {float(te_pw):.1f}", flush=True)

    net = GraphConceptField(d=args.d_model, layers=args.layers, slow_heads=args.slow_heads,
                            n_res=len(vocab), n_eg=len(egs),
                            cls_reach=cr, cls_eg=ce, cls_out=co,
                            edge_codes=args.edge_codes, move_codes=args.move_codes,
                            gps_parallel=bool(args.gps_parallel),
                            attn_bias=bool(args.attn_bias),
                            agg_max=bool(args.agg_max), kin=bool(args.kin),
                            file_nodes=bool(args.file_nodes),
                            phase_head=bool(args.phase_head),
                            threat_head=bool(args.threat_head),
                            elo_cond=bool(args.elo),
                            g_heads=args.g_heads, g_codes=args.g_codes,
                            d_code=args.d_code).to(dev)
    print(f"[gcf] {sum(p.numel() for p in net.parameters()):,} params | "
          f"{args.slow_heads} slow + {args.g_heads - args.slow_heads} fast global "
          f"heads x {args.g_codes} (d_code {args.d_code}) | 64x256 sq",
          flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    probe = te_i[rng.choice(len(te_i), min(2500, len(te_i)), replace=False)]

    def parent_batch(idx):
        b = {k: D[k][idx].to(dev) for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
        sqf, mvf, edges, mask = featurize(b, dev, n_nodes=net.n_nodes)
        # deletion relations, same flattening as featurize's mut
        off = (torch.arange(len(idx), device=dev) * net.n_nodes)[:, None]
        jj = torch.arange(MV_CAP, device=dev)[None, :]
        for name, arr in (("vac", D["vac"][idx].to(dev)), ("dsc", D["dsc"][idx].to(dev))):
            a = arr.long()
            val = (a != 255) & mask[:, :, None]
            s = ((jj + 64 + off)[:, :, None].expand_as(a))[val]
            t = (a + off[:, :, None])[val]
            edges[name] = (s, t)
        return sqf, mvf, edges, mask

    def slim_batch(idx, pfx):
        """child / t+8 square-only graphs: move block empty."""
        B = len(idx)
        b = {"sq": D[f"{pfx}_sq"][idx].to(dev), "gl": D[f"{pfx}_gl"][idx].to(dev),
             "mv": torch.zeros(B, MV_CAP, 10, dtype=torch.uint8, device=dev),
             "mut": torch.full((B, MV_CAP, 24), 255, dtype=torch.uint8, device=dev),
             "att": D[f"{pfx}_att"][idx].to(dev),
             "n_mv": torch.zeros(B, dtype=torch.uint8, device=dev),
             "n_att": D[f"{pfx}_n_att"][idx].to(dev)}
        return featurize(b, dev, n_nodes=net.n_nodes)

    def step_fn(model, step):
        model.train()
        idx = torch.from_numpy(tr_i[rng.integers(0, len(tr_i), args.batch)])
        sqf, mvf, edges, mask = parent_batch(idx)
        elo_b = EL[idx].to(dev) if EL is not None else None
        zq, g_ids, sq_ids, vl, h_mv, pre_g = model.encode(sqf, mvf, edges, mask, elo=elo_b)
        # capture NOW: the child/lag-8 encodes below overwrite the stashes
        h_sq_p = model.last_h_sq
        pre_adv_p = model.last_pre_adv if args.phase_head else None

        played = D["played"][idx].long().to(dev)
        lg = model.move_logits(zq, h_mv, mask)
        l_move = F.cross_entropy(lg, played)

        with torch.no_grad():
            csqf, cmvf, cedges, cmask = slim_batch(idx, "c")
            if args.edge_codes:
                _, cg_ids, csq_ids, _, _, _, c_hist = model.encode(
                    csqf, cmvf, cedges, cmask, want_edge_ids=True, elo=elo_b)
            else:
                _, cg_ids, csq_ids, _, _, _ = model.encode(csqf, cmvf, cedges, cmask,
                                                           elo=elo_b)
        mv_played = mvf[torch.arange(len(idx), device=dev), played]
        dy = model.dyn(torch.cat([zq, model.mv_emb(mv_played)], -1))
        H, C, SC = model.g_heads, args.g_codes, model.sq_codes
        dg = dy[:, :H * C].reshape(-1, C)
        ds = dy[:, H * C:].reshape(-1, SC)
        l_dyn = F.cross_entropy(dg, cg_ids.reshape(-1)) + F.cross_entropy(ds, csq_ids.reshape(-1))
        l_edyn = torch.zeros((), device=dev)
        if args.edge_codes:
            pred_h = F.log_softmax(model.dyn_edge(
                torch.cat([zq, model.mv_emb(mv_played)], -1)), -1)
            l_edyn = F.kl_div(pred_h, c_hist.clamp(min=1e-8), reduction="batchmean")
        l_mvvq = getattr(model, "_mv_vq_loss", torch.zeros((), device=dev))

        if args.res_mode == "tb":
            rt = torch.from_numpy(p_np[idx.numpy()])
            r_l, e_l, o_l = model.predict_resolution(zq)
            l_res, _ = factorized_resolution_ce(
                r_l, e_l, o_l, torch.from_numpy(r_reach[rt]).to(dev),
                torch.from_numpy(r_eg[rt]).to(dev), torch.from_numpy(r_out[rt]).to(dev),
                eg_weight=eg_w)
        else:
            _, _, o_l = model.predict_resolution(zq)
            ok_m = ~CEN[idx]
            l_res = F.cross_entropy(o_l[ok_m], OUTW[idx][ok_m].to(dev)) if ok_m.any() \
                else torch.zeros((), device=dev)
        l_elo = torch.zeros((), device=dev)
        if EL is not None:
            l_elo = F.mse_loss(model.elo_head(zq), elo_b)

        counts = torch.stack([(D["sq"][idx, :, 0] == t).sum(1) for t in range(1, 13)], 1)
        l_now = F.cross_entropy(model.now_head(zq).view(-1, N_COUNT),
                                counts.clamp(max=8).reshape(-1).long().to(dev))

        i8 = torch.from_numpy(tr8[rng.integers(0, len(tr8), args.vamp_batch)])
        elo8 = EL[i8].to(dev) if EL is not None else None
        s0f, m0f, e0, k0 = parent_batch(i8)
        _, _, _, _, _, pg0 = model.encode(s0f, m0f, e0, k0, elo=elo8)
        s8f, m8f, e8, k8 = slim_batch(i8, "c8")
        _, _, _, _, _, pg8 = model.encode(s8f, m8f, e8, k8, elo=elo8)
        sl = args.slow_heads * model.d_code
        v0 = model.vamp_proj(pg0.reshape(len(i8), -1)[:, :sl])
        v8_ = model.vamp_proj(pg8.reshape(len(i8), -1)[:, :sl])
        vamp = vamp2_score(v0, v8_)
        l_slow = -vamp / 32.0
        l_dec = vicreg_covariance(pg0.reshape(len(i8), -1)[:, sl:])
        # cross-head decorrelation on the LARGE (vamp) batch: noise floor is 32/B
        l_decor = cross_head_decorrelation(pg0) if args.w_decor > 0 else \
            torch.zeros((), device=dev)

        l_phase = l_adv = torch.zeros((), device=dev)
        adv_acc = 0.0
        if args.phase_head:
            ph_t = PH[idx].to(dev)
            l_phase = F.cross_entropy(model.phase_cls(pre_g[:, 0]), ph_t)
            adv_lg = model.phase_adv(grad_reverse(pre_adv_p.reshape(len(idx), -1)))
            l_adv = F.cross_entropy(adv_lg, ph_t)
            adv_acc = float((adv_lg.argmax(-1) == ph_t).float().mean())
        l_te = torch.zeros((), device=dev)
        if args.threat_head:
            te_t = TE[idx].to(dev)
            msk = te_t >= 0
            if msk.any():
                l_te = F.binary_cross_entropy_with_logits(
                    model.te_head(h_sq_p).squeeze(-1)[msk], te_t[msk].float(),
                    pos_weight=te_pw)

        loss = (args.w_move * l_move + args.w_dyn * l_dyn + args.w_edge_dyn * l_edyn
                + args.w_res * l_res + args.w_now * l_now + args.w_slow * l_slow
                + args.w_dec * l_dec + args.w_decor * l_decor
                + args.w_phase * l_phase + args.w_adv * l_adv + args.w_te * l_te
                + args.w_elo * l_elo + vl + l_mvvq)
        opt.zero_grad(); loss.backward(); opt.step()

        m = {}
        if step % args.eval_every == 0 or step == 1:
            model.eval()
            h1 = h3 = tot = 0
            with torch.no_grad():
                for a in range(0, len(probe), 256):
                    ii = torch.from_numpy(probe[a:a + 256])
                    sqf2, mvf2, ed2, mk2 = parent_batch(ii)
                    zq2, _, _, _, hm2, _ = model.encode(sqf2, mvf2, ed2, mk2)
                    lg2 = model.move_logits(zq2, hm2, mk2)
                    pl2 = D["played"][ii].long().to(dev)
                    rk = (lg2 > lg2.gather(1, pl2[:, None])).sum(1)
                    h1 += int((rk == 0).sum()); h3 += int((rk < 3).sum()); tot += len(ii)
                dyn_g = float((dg.argmax(-1) == cg_ids.reshape(-1)).float().mean())
                dyn_s = float((ds.argmax(-1) == csq_ids.reshape(-1)).float().mean())
            el = time.time() - t0
            print(f"[gcf] step {step:>6}/{args.steps} {el/60:>5.1f}m "
                  f"({el/max(1,step):.3f}s/step) | move {float(l_move):.3f} "
                  f"dyn {float(l_dyn):.3f} res {float(l_res):.3f} | "
                  f"HELD-OUT top1 {h1/tot:.3f} top3 {h3/tot:.3f} | dyn_g {dyn_g:.2f} "
                  f"dyn_s {dyn_s:.2f} edyn {float(l_edyn):.3f} | vamp {float(vamp):.1f}/32 "
                  f"dec {float(l_dec):.3f} decor {float(l_decor):.3f} "
                  f"ph {float(l_phase.detach()):.2f} advacc {adv_acc:.2f} "
                  f"te {float(l_te.detach()):.3f} elo {float(l_elo.detach()):.3f}",
                  flush=True)
            m = {"top1": h1 / tot, "top3": h3 / tot, "vamp": float(vamp),
                 "dyn_g": dyn_g, "dyn_s": dyn_s, "decor": float(l_decor),
                 "adv_acc": adv_acc}
            model.train()
        return {"loss": float(loss.detach()), "move": float(l_move.detach()),
                "dyn": float(l_dyn.detach()), "res": float(l_res.detach()),
                "slow": float(l_slow.detach()), **m}

    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["state_dict"]); start = int(ck.get("step", 0))
        print(f"[gcf] resumed from {args.resume} at step {start}", flush=True)
    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      start_step=start, eval_every=args.eval_every, experiment="catspace",
                      run_name=args.out.split("/")[-1], device=dev)
    standard_train(step_fn, net, cfg, args)


if __name__ == "__main__":
    main()
