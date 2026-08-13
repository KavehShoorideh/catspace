#!/usr/bin/env python
"""jqt.py -- JOINT QUANTIZED TRAINING (Kaveh 2026-08-12: "jointly train all the way down to the
concepts and see if we can predict future concept activations").

The inversion this module implements: concept-prediction becomes part of the FOUNDATION rather
than a post-hoc compression of a frozen field. The grounding stack (walls/basin/hinge/anchor)
stays intact in train_reach_vit as the collapse anchor; this module adds, jointly:

  1. VQ bottleneck on phi (EMA codebook) reconstructing the field's own evaluations --
     faithfulness by construction, unchanged from concept_vq.
  2. PERSISTENCE PRIOR: pre-quant latents of consecutive plies pulled together where the eval
     did not move (metastable codes are the predictable/legible/subgoal-worthy ones).
  3. JEPA future-code prediction in EMBEDDING space: predictor(parent codes, move) -> child's
     quantized embedding from the EMA target branch. Indices are never a target (codebook-churn
     escape); the target is stop-grad (JEPA guard).
  4. TWO CONCEPT-GOAL RULERS, codebook entries as goal anchors projected into z_B space:
     dA(s -> concept) trained as CENSORED plies-to-first-activation, dB(s -> concept) as a
     calibrated P(activate before game end) via first-hit BCE. The "can I activate this
     concept" question asked of both rulers.

Labels for (4) come from an ActivationIndex refreshed periodically from the EMA branch --
codes along real games, first-activation events per (game, head).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn


class JQTModule(nn.Module):
    def __init__(self, d_model=256, heads=8, codes=64, d_code=32, d=64, d_move=48,
                 hidden=384, ema=0.996, square_codes=0, piece_codes=0, d_fine=16):
        super().__init__()
        from vector_quantize_pytorch import VectorQuantize
        self.heads, self.codes, self.d_code, self.d = heads, codes, d_code, d
        self.ema = float(ema)
        self.enc = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(),
                                 nn.Linear(256, heads * d_code))
        self.vq = nn.ModuleList([VectorQuantize(dim=d_code, codebook_size=codes,
                                                decay=0.9, commitment_weight=0.25)
                                 for _ in range(heads)])
        self.dec = nn.Sequential(nn.Linear(heads * d_code, 256), nn.GELU(),
                                 nn.Linear(256, 6))
        # EMA target of the concept encoder (the JEPA target reads phi through THIS, then the
        # live EMA codebooks -- both drift slowly, so the embedding-space target is stable).
        self.t_enc = copy.deepcopy(self.enc)
        for p in self.t_enc.parameters():
            p.requires_grad_(False)
        # future-code predictor: (quantized parent, move token) -> child quantized embedding
        self.e_from = nn.Embedding(64, d_move)
        self.e_to = nn.Embedding(64, d_move)
        self.e_promo = nn.Embedding(5, d_move)
        self.pred = nn.Sequential(nn.Linear(heads * d_code + d_move, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, heads * d_code))
        # jqt5 HURDLE DYNAMICS (Kaveh 2026-08-13 "at every move there's a concept change;
        # predict which ones change"): flip BCE per head + destination CE on flipped heads
        # only. Replaces the JEPA MSE, whose convergence was measured to be ~persistence
        # (eval_jepa: copy-parent within 9-14%, move-conditioning 1.07-1.11x). CE cannot
        # mean-collapse and has no moving embedding floor.
        self.flip = nn.Sequential(nn.Linear(heads * d_code + d_move, hidden), nn.GELU(),
                                  nn.Linear(hidden, heads))
        self.dest = nn.Sequential(nn.Linear(heads * d_code + d_move, hidden), nn.GELU(),
                                  nn.Linear(hidden, heads * codes))
        # jqt5 NEGATIVE POLES (Kaveh: "a pole of where the concepts do not exist, something
        # to push against for CDB"): per-vocabulary anti-anchor; dead states sit at ~0
        # distance from it; the CDB logit becomes a two-pole contrast (committor pattern).
        self.anchor_neg = nn.ModuleList([nn.Linear(d_code, d) for _ in range(heads)])
        self.db_a2 = nn.Parameter(torch.tensor(1.0))
        self.db_b2 = nn.Parameter(torch.tensor(0.0))
        # concept-goal anchors: per-head projection of a codebook vector into z_B space (d)
        self.anchor = nn.ModuleList([nn.Linear(d_code, d) for _ in range(heads)])
        # activation-probability link: logit = a * (b0 - log1p(dB(s->anchor)))
        self.db_a = nn.Parameter(torch.tensor(1.0))
        self.db_b0 = nn.Parameter(torch.tensor(3.0))
        # running normalisation of the 6 reconstruction targets (dA3 + P3)
        self.register_buffer("y_mu", torch.zeros(6))
        self.register_buffer("y_sd", torch.ones(6))
        self.register_buffer("y_n", torch.zeros(1))
        # ---- v2 streams (Kaveh 2026-08-12): SQUARE and PIECE concept vocabularies, residual
        # on the global stream (RVQ-across-granularities: each finer vocabulary encodes only
        # what the coarser ones failed to explain -- collinearity removed by construction).
        self.square_codes, self.piece_codes = square_codes, piece_codes
        if square_codes:
            self.sq_proj = nn.Linear(d_model, d_fine)
            self.sq_vq = VectorQuantize(dim=d_fine, codebook_size=square_codes,
                                        decay=0.9, commitment_weight=0.25)
            # per-square ADDITIVE contribution to the six outputs: summing makes every
            # square's share of the evaluation an explicit, legible number
            self.sq_dec = nn.Sequential(nn.Linear(d_fine, 64), nn.GELU(), nn.Linear(64, 6))
        # v3 (jqt4, Kaveh 2026-08-13 'best possible base'): ANCHORS for the square and
        # piece vocabularies -- the planner cannot commit to 'f7:weak' or 'my knight:trapped'
        # without a z_B-space anchor and rulers trained against it. Square anchors are
        # ADDRESSED (code embedding + square position embedding); piece anchors are keyed by
        # the slot's piece TYPE (identity class) + code.
        if square_codes:
            self.anchor_sq = nn.Linear(d_fine, d)
            self.anchor_sq_pos = nn.Embedding(64, d)
            self.anchor_sq_neg = nn.Linear(d_fine, d)       # jqt5 anti-pole (shared pos emb)
        if piece_codes:
            self.anchor_pc = nn.Linear(d_fine, d)
            self.anchor_pc_type = nn.Embedding(13, d)
            self.anchor_pc_neg = nn.Linear(d_fine, d)       # jqt5 anti-pole (shared type emb)
            self.pc_flip = nn.Sequential(nn.Linear(d_fine + d_move, 128), nn.GELU(),
                                         nn.Linear(128, 1))
            self.pc_dest = nn.Sequential(nn.Linear(d_fine + d_move, 128), nn.GELU(),
                                         nn.Linear(128, piece_codes))
        if piece_codes:
            self.pc_id = nn.Embedding(13, d_model)          # identity: piece type at start
            self.pc_captured = nn.Parameter(torch.randn(d_model) * 0.02)
            lyr = nn.TransformerEncoderLayer(d_model=d_model, nhead=4,
                                             dim_feedforward=2 * d_model, batch_first=True,
                                             norm_first=True, dropout=0.0, activation="gelu")
            self.pc_tr = nn.TransformerEncoder(lyr, num_layers=2)
            self.pc_proj = nn.Linear(d_model, d_fine)
            self.pc_vq = VectorQuantize(dim=d_fine, codebook_size=piece_codes,
                                        decay=0.9, commitment_weight=0.25)
            self.pc_dec = nn.Sequential(nn.Linear(d_fine, 64), nn.GELU(), nn.Linear(64, 6))

    # ---- quantization ---------------------------------------------------------------------------
    def latents(self, phi):
        """phi -> pre-quant concept latents (B, H, d_code)."""
        return self.enc(phi).view(len(phi), self.heads, self.d_code)

    def quantize(self, phi):
        """phi -> (h_pre, z_q flat (B, H*d_code), ids (B,H), vq_loss)."""
        h = self.latents(phi)
        qs, ids, vloss = [], [], 0.0
        for i, vq in enumerate(self.vq):
            q, idx, l = vq(h[:, i])
            qs.append(q); ids.append(idx); vloss = vloss + l
        return h, torch.cat(qs, -1), torch.stack(ids, 1), vloss

    @torch.no_grad()
    def target_codes(self, phi_t):
        """EMA-branch quantized embedding of phi_t (which itself must come from the model's EMA
        encoder). Codebooks are read in eval mode so the target pass never updates their EMA."""
        h = self.t_enc(phi_t).view(len(phi_t), self.heads, self.d_code)
        was = self.training
        self.eval()
        qs, ids = [], []
        for i, vq in enumerate(self.vq):
            q, idx, _ = vq(h[:, i])
            qs.append(q); ids.append(idx)
        if was:
            self.train()
        return torch.cat(qs, -1).detach(), torch.stack(ids, 1)

    def predict_child(self, z_q_par, mids):
        m = self.e_from(mids[:, 0]) + self.e_to(mids[:, 1]) + self.e_promo(mids[:, 2])
        return self.pred(torch.cat([z_q_par, m], -1))

    def predict_flips(self, z_q_par, mids):
        """jqt5 hurdle dynamics -> (flip logits (B,H), destination logits (B,H,C))."""
        m = self.e_from(mids[:, 0]) + self.e_to(mids[:, 1]) + self.e_promo(mids[:, 2])
        x = torch.cat([z_q_par, m], -1)
        return self.flip(x), self.dest(x).view(len(x), self.heads, self.codes)

    def pc_predict_flips(self, h_pc, mids):
        """per-SLOT hurdle dynamics: (B,32,d_fine) pre-quant latents + move ->
        (flip logits (B,32), destination logits (B,32,C_pc))."""
        m = self.e_from(mids[:, 0]) + self.e_to(mids[:, 1]) + self.e_promo(mids[:, 2])
        x = torch.cat([h_pc, m[:, None].expand(-1, h_pc.shape[1], -1)], -1)
        return self.pc_flip(x).squeeze(-1), self.pc_dest(x)

    def square_stream(self, sq_tokens):
        """(B,64,d_model) per-square trunk outputs -> (h, y_contrib (B,6), ids (B,64), loss)."""
        h = self.sq_proj(sq_tokens)
        q, ids, vl = self.sq_vq(h)
        return h, self.sq_dec(q).sum(1), ids, vl

    def piece_stream(self, sq_tokens, slots, slot_type):
        """slots (B,64) int64 slot-of-square (-1 empty); slot_type (B,32) piece type at the
        game start (0 = slot unused). Gathers each ALIVE slot's current square token, adds the
        identity embedding, contextualizes with a 2-layer transformer over the 32 slots
        (identity-matched across plies: THE space where persistence is well-posed), quantizes
        per slot. Returns (h (B,32,d_fine), y_contrib (B,6), ids (B,32), vq_loss,
        alive (B,32) bool)."""
        B = len(sq_tokens)
        dev = sq_tokens.device
        sq_of_slot = torch.full((B, 32), -1, dtype=torch.long, device=dev)
        bi, si = torch.nonzero(slots >= 0, as_tuple=True)
        sq_of_slot[bi, slots[bi, si]] = si
        alive = sq_of_slot >= 0
        used = slot_type > 0
        gath = torch.zeros(B, 32, sq_tokens.shape[-1], device=dev, dtype=sq_tokens.dtype)
        bb, ss = torch.nonzero(alive, as_tuple=True)
        gath[bb, ss] = sq_tokens[bb, sq_of_slot[bb, ss]]
        gath = torch.where(alive[..., None], gath,
                           self.pc_captured[None, None, :].expand_as(gath))
        x = gath + self.pc_id(slot_type.long())
        x = self.pc_tr(x, src_key_padding_mask=~used)
        h = self.pc_proj(x)
        q, ids, vl = self.pc_vq(h)
        contrib = self.pc_dec(q) * used[..., None].float()
        return h, contrib.sum(1), ids, vl, alive & used

    # ---- concept goals --------------------------------------------------------------------------
    def anchors_for(self, hc):
        """(B,2) [head, code] -> z_B-space goal anchors (B, d). Codebook vectors DETACHED: the
        anchor projection learns; the codebook answers only to the VQ/JEPA losses."""
        cb = torch.stack([vq.codebook for vq in self.vq])           # (H, K, d_code)
        e = cb[hc[:, 0], hc[:, 1]].detach()
        out = torch.zeros(len(hc), self.d, device=e.device, dtype=e.dtype)
        for h in range(self.heads):
            m = hc[:, 0] == h
            if m.any():
                out[m] = self.anchor[h](e[m]).to(out.dtype)
        return out

    def anchors_for_sq(self, sq, code):
        """(N,) squares + (N,) square-codes -> z_B anchors: 'square X in state c'."""
        cb = self.sq_vq.codebook.detach()
        return self.anchor_sq(cb[code]) + self.anchor_sq_pos(sq)

    def anchors_for_pc(self, ptype, code):
        """(N,) piece types + (N,) piece-codes -> z_B anchors: 'a TYPE piece in state c'."""
        cb = self.pc_vq.codebook.detach()
        return self.anchor_pc(cb[code]) + self.anchor_pc_type(ptype)

    def activation_logit(self, dB):
        return self.db_a * (self.db_b0 - torch.log1p(dB))

    def anchors_neg_for(self, hc):
        """anti-pole anchors, global vocabulary (jqt5). Same codebook vector, its own
        projection: 'the region where head-h/code-c is DEAD'."""
        cb = torch.stack([vq.codebook for vq in self.vq])
        e = cb[hc[:, 0], hc[:, 1]].detach()
        out = torch.zeros(len(hc), self.d, device=e.device, dtype=e.dtype)
        for h in range(self.heads):
            m = hc[:, 0] == h
            if m.any():
                out[m] = self.anchor_neg[h](e[m]).to(out.dtype)
        return out

    def anchors_neg_for_sq(self, sq, code):
        cb = self.sq_vq.codebook.detach()
        return self.anchor_sq_neg(cb[code]) + self.anchor_sq_pos(sq)

    def anchors_neg_for_pc(self, ptype, code):
        cb = self.pc_vq.codebook.detach()
        return self.anchor_pc_neg(cb[code]) + self.anchor_pc_type(ptype)

    def activation_logit2(self, dB_pos, dB_neg):
        """jqt5 two-pole link: contrast against the anti-pole (the committor pattern) --
        self-normalizing under global dB scale drift, and 'actively dead' becomes
        representable instead of just 'far'."""
        return self.db_a2 * (torch.log1p(dB_neg) - torch.log1p(dB_pos)) + self.db_b2

    # ---- housekeeping ---------------------------------------------------------------------------
    @torch.no_grad()
    def update_target(self):
        for p, tp in zip(self.enc.parameters(), self.t_enc.parameters()):
            tp.mul_(self.ema).add_(p, alpha=1.0 - self.ema)

    @torch.no_grad()
    def update_y_stats(self, y):
        """running mean/sd of the 6 recon targets; frozen once warm (first ~50 batches)."""
        if float(self.y_n) < 50:
            n = float(self.y_n)
            self.y_mu.mul_(n / (n + 1)).add_(y.mean(0) / (n + 1))
            self.y_sd.mul_(n / (n + 1)).add_(y.std(0).clamp(min=1e-6) / (n + 1))
            self.y_n += 1

    @torch.no_grad()
    def perplexity(self, ids):
        """codebook usage per head -> mean exp(entropy); K = fully used, ~1 = collapsed."""
        ps = []
        for h in range(self.heads):
            c = torch.bincount(ids[:, h], minlength=self.codes).float()
            p = c / c.sum().clamp(min=1)
            ps.append(float(torch.exp(-(p * (p + 1e-9).log()).sum())))
        return float(np.mean(ps))


class ActivationIndex:
    """First-activation events per (game, head) over a coded sample of the corpus, refreshed
    periodically from the EMA branch. sample() draws balanced activation labels:
      positive: a code that DOES first-activate later in the game (plies-to-activation, hit=1)
      negative: a code that never activates later (censored at game end, hit=0)."""

    def __init__(self, rng, codes=64, sq_codes=0, pc_codes=0):
        self.rng = rng
        self.codes = int(codes)
        self.sq_codes, self.pc_codes = int(sq_codes), int(pc_codes)
        self.games = []          # (rows, codes (L,H), sq (L,64)|None, pc (L,32)|None,
                                 #  ptype (32,)|None)

    def refresh(self, games):
        self.games = games

    def ready(self):
        return len(self.games) > 0

    def sample_mixed(self, n, frac_sq=0.25, frac_pc=0.25, frac_now=0.3, frac_dead=0.0):
        """heterogeneous goal labels (jqt4): global / (square,code) / (slot-type,code).
        -> rows, gtype (0 glob/1 sq/2 pc), key1, key2, plies, hit.
        square goal: 'square k reaches code c'; piece goal: 'a piece of type t reaches c'
        (labeled on a specific alive slot of that type; anchors are type-keyed)."""
        rows = np.empty(n, np.int64)
        gtype = np.zeros(n, np.int64)
        k1 = np.empty(n, np.int64); k2 = np.empty(n, np.int64)
        plies = np.empty(n, np.float32); hit = np.empty(n, np.float32)
        dead = np.zeros(n, np.float32)
        gsel = self.rng.integers(0, len(self.games), n)
        for b in range(n):
            g = self.games[gsel[b]]
            rws, C = g[0], g[1]
            SQ = g[2] if len(g) > 2 else None
            PC = g[3] if len(g) > 3 else None
            PT = g[4] if len(g) > 4 else None
            AL = g[5] if len(g) > 5 else None
            L = len(rws)
            pp = int(self.rng.integers(0, max(1, L - 2)))
            r = self.rng.random()
            # jqt5 DEAD-STATE negative (anti-pole member): a piece TYPE with zero alive
            # representatives -- every piece-concept of that type is unreachable from here.
            if AL is not None and PT is not None and self.rng.random() < frac_dead:
                started = np.unique(PT[PT > 0])
                dd = [int(t) for t in started if AL[pp, int(t)] == 0]
                if dd:
                    gtype[b] = 2
                    rows[b] = rws[pp]
                    k1[b] = dd[int(self.rng.integers(0, len(dd)))]
                    k2[b] = int(self.rng.integers(0, self.pc_codes))
                    plies[b] = -1.0; hit[b] = 0.0; dead[b] = 1.0
                    continue
            if SQ is not None and r < frac_sq:
                gtype[b] = 1
                sq = int(self.rng.integers(0, 64))
                fut = SQ[pp + 1:, sq]; prev = SQ[pp:-1, sq]
                ev = np.flatnonzero(fut != prev)
                if bool(self.rng.integers(0, 2)) and len(ev):
                    if self.rng.random() < frac_now:
                        rows[b] = rws[pp]; k1[b] = sq; k2[b] = int(SQ[pp, sq])
                        plies[b] = 0.0; hit[b] = 1.0
                        continue
                    e = int(ev[self.rng.integers(0, len(ev))])
                    rows[b] = rws[pp]; k1[b] = sq; k2[b] = int(fut[e])
                    plies[b] = float(e + 1); hit[b] = 1.0
                else:
                    active = set(np.unique(fut[ev]).tolist()) if len(ev) else set()
                    active.add(int(SQ[pp, sq]))
                    c = int(self.rng.integers(0, self.sq_codes))
                    for _ in range(20):
                        if c not in active: break
                        c = int(self.rng.integers(0, self.sq_codes))
                    rows[b] = rws[pp]; k1[b] = sq; k2[b] = c
                    plies[b] = -1.0; hit[b] = 0.0
            elif PC is not None and r < frac_sq + frac_pc:
                gtype[b] = 2
                alive = np.flatnonzero(PT > 0)
                sl = int(alive[self.rng.integers(0, len(alive))]) if len(alive) else 0
                fut = PC[pp + 1:, sl]; prev = PC[pp:-1, sl]
                ev = np.flatnonzero(fut != prev)
                if bool(self.rng.integers(0, 2)) and len(ev):
                    if self.rng.random() < frac_now:
                        rows[b] = rws[pp]; k1[b] = int(PT[sl]); k2[b] = int(PC[pp, sl])
                        plies[b] = 0.0; hit[b] = 1.0
                        continue
                    e = int(ev[self.rng.integers(0, len(ev))])
                    rows[b] = rws[pp]; k1[b] = int(PT[sl]); k2[b] = int(fut[e])
                    plies[b] = float(e + 1); hit[b] = 1.0
                else:
                    active = set(np.unique(fut[ev]).tolist()) if len(ev) else set()
                    active.add(int(PC[pp, sl]))
                    c = int(self.rng.integers(0, self.pc_codes))
                    for _ in range(20):
                        if c not in active: break
                        c = int(self.rng.integers(0, self.pc_codes))
                    rows[b] = rws[pp]; k1[b] = int(PT[sl]); k2[b] = c
                    plies[b] = -1.0; hit[b] = 0.0
            else:
                h = int(self.rng.integers(0, C.shape[1]))
                fut = C[pp + 1:, h]; prev = C[pp:-1, h]
                ev = np.flatnonzero(fut != prev)
                if bool(self.rng.integers(0, 2)) and len(ev):
                    if self.rng.random() < frac_now:
                        # SUBSUMPTION sample (Kaveh 2026-08-13: the point cloud IS the
                        # region; members sit at ZERO distance from the concept pole, the
                        # triangle inequality makes d(s->pole) = distance to the NEAREST
                        # part of the region -- multimodality dissolves)
                        rows[b] = rws[pp]; k1[b] = h; k2[b] = int(C[pp, h])
                        plies[b] = 0.0; hit[b] = 1.0
                        continue
                    e = int(ev[self.rng.integers(0, len(ev))])
                    rows[b] = rws[pp]; k1[b] = h; k2[b] = int(fut[e])
                    plies[b] = float(e + 1); hit[b] = 1.0
                else:
                    active = set(np.unique(fut[ev]).tolist()) if len(ev) else set()
                    active.add(int(C[pp, h]))
                    c = int(self.rng.integers(0, self.codes))
                    for _ in range(20):
                        if c not in active: break
                        c = int(self.rng.integers(0, self.codes))
                    rows[b] = rws[pp]; k1[b] = h; k2[b] = c
                    plies[b] = -1.0; hit[b] = 0.0
        return rows, gtype, k1, k2, plies, hit, dead

    def sample(self, n):
        rows = np.empty(n, np.int64)
        hc = np.empty((n, 2), np.int64)
        plies = np.empty(n, np.float32)
        hit = np.empty(n, np.float32)
        gsel = self.rng.integers(0, len(self.games), n)
        H = self.games[0][1].shape[1]
        for b in range(n):
            rws, C = self.games[gsel[b]]
            L = len(rws)
            p = int(self.rng.integers(0, max(1, L - 2)))
            h = int(self.rng.integers(0, H))
            fut = C[p + 1:, h]
            prev = C[p:-1, h]
            ev = np.flatnonzero(fut != prev)             # activation events after p
            want_pos = bool(self.rng.integers(0, 2)) and len(ev) > 0
            if want_pos:
                e = int(ev[self.rng.integers(0, len(ev))])
                rows[b] = rws[p]; hc[b] = (h, int(fut[e]))
                plies[b] = float(e + 1); hit[b] = 1.0
            else:
                active = set(np.unique(fut[ev]).tolist()) if len(ev) else set()
                active.add(int(C[p, h]))                 # "activate" = newly enter, not hold
                K = self.codes
                c = int(self.rng.integers(0, K))
                for _ in range(20):                      # rejection: label must be a TRUE never
                    if c not in active:
                        break
                    c = int(self.rng.integers(0, K))
                rows[b] = rws[p]; hc[b] = (h, c)
                plies[b] = -1.0; hit[b] = 0.0
        return rows, hc, plies, hit
