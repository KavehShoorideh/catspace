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
                 hidden=384, ema=0.996):
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
        # concept-goal anchors: per-head projection of a codebook vector into z_B space (d)
        self.anchor = nn.ModuleList([nn.Linear(d_code, d) for _ in range(heads)])
        # activation-probability link: logit = a * (b0 - log1p(dB(s->anchor)))
        self.db_a = nn.Parameter(torch.tensor(1.0))
        self.db_b0 = nn.Parameter(torch.tensor(3.0))
        # running normalisation of the 6 reconstruction targets (dA3 + P3)
        self.register_buffer("y_mu", torch.zeros(6))
        self.register_buffer("y_sd", torch.ones(6))
        self.register_buffer("y_n", torch.zeros(1))

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
                out[m] = self.anchor[h](e[m])
        return out

    def activation_logit(self, dB):
        return self.db_a * (self.db_b0 - torch.log1p(dB))

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

    def __init__(self, rng, codes=64):
        self.rng = rng
        self.codes = int(codes)
        self.games = []          # list of (rows (L,), codes (L,H))

    def refresh(self, games):
        self.games = games

    def ready(self):
        return len(self.games) > 0

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
