"""catspace/nn/opponent.py -- the OPPONENT MODEL, option A (Kaveh 2026-07-23, decided:
candidate-set self-attention; INQUIRY_MULTICHANNEL_FIELD.md sec 8).

P(cohort plays move m | position) with SET-CONTEXTUAL move scores:
  1. legal moves -> TOKENS (from/to/piece/captured embeddings + board-context vector)
  2. SELF-ATTENTION among move tokens ("seeing m depends on what else is going on":
     distractor suppression, threat load, Einstellung-style competition -- learnable)
  3. CROSS-ATTENTION from move tokens to the cohort's SKILL TOKENS (Elo bins now;
     engine cohorts reserved; tilt = swapping/updating the cohort id online)
  4. masked softmax over legal moves.

Cohort ids: 0..10 = Elo bins (features.elo_bin), 11 = Stockfish, 12 = Leela (reserved),
13-15 headroom. Trained by masked CE on (position, move played, mover cohort) triples.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import BoardEncoder
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import N_PLANES

N_COHORTS = 24
# cohort ids: 0-10 human Elo bins (features.elo_bin) · 11 sf_full · 12 sf_2000 · 13 sf_1700
# · 14 sf_1400 · 15 random (known-uniform anchor) · 16 maia_1100 · 17 maia_1500 · 18 maia_1900
COHORT_ENGINE = {2: 11, 6: 12, 5: 13, 4: 14, 3: 15, 8: 16, 9: 17, 10: 18}


class OpponentModel(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4, enc_out: int = 256,
                 d_tok: int = 128, n_heads: int = 4, n_self_layers: int = 2,
                 n_skill_tokens: int = 8, max_moves: int = 80, seed: int = 0):
        torch.manual_seed(seed)
        super().__init__()
        self.config = dict(channels=channels, blocks=blocks, enc_out=enc_out, d_tok=d_tok,
                           n_heads=n_heads, n_self_layers=n_self_layers,
                           n_skill_tokens=n_skill_tokens, max_moves=max_moves, seed=seed)
        self.max_moves = max_moves
        self.enc = BoardEncoder(N_PLANES, channels, blocks, enc_out)
        self.ctx = nn.Linear(enc_out, d_tok)
        self.emb_from = nn.Embedding(64, d_tok)
        self.emb_to = nn.Embedding(64, d_tok)
        self.emb_piece = nn.Embedding(8, d_tok)     # 0 pad, 1-6 piece types, 7 spare
        self.emb_capt = nn.Embedding(8, d_tok)      # captured piece type (0 = none)
        layer = nn.TransformerEncoderLayer(d_model=d_tok, nhead=n_heads,
                                           dim_feedforward=4 * d_tok, batch_first=True,
                                           dropout=0.0, norm_first=True)
        self.self_attn = nn.TransformerEncoder(layer, num_layers=n_self_layers)
        self.skills = nn.Embedding(N_COHORTS * n_skill_tokens, d_tok)
        self.n_skill_tokens = n_skill_tokens
        self.cross = nn.MultiheadAttention(d_tok, n_heads, batch_first=True, dropout=0.0)
        self.cross_norm = nn.LayerNorm(d_tok)
        self.head = nn.Sequential(nn.LayerNorm(d_tok), nn.Linear(d_tok, d_tok), nn.GELU(),
                                  nn.Linear(d_tok, 1))

    def forward(self, planes, mv_from, mv_to, mv_piece, mv_capt, n_moves, cohort):
        """planes (B,20,8,8); mv_* (B,L) int64 padded; n_moves (B,); cohort (B,).
        Returns logits (B,L) with -inf on padding."""
        B, L = mv_from.shape
        h = self.enc(planes)                                    # (B, enc_out)
        tok = (self.emb_from(mv_from) + self.emb_to(mv_to)
               + self.emb_piece(mv_piece) + self.emb_capt(mv_capt)
               + self.ctx(h)[:, None, :])                       # (B, L, d)
        pad = torch.arange(L, device=planes.device)[None, :] >= n_moves[:, None]
        tok = self.self_attn(tok, src_key_padding_mask=pad)
        base = self.skills.weight.view(N_COHORTS, self.n_skill_tokens, -1)[cohort]  # (B, K, d)
        att, _ = self.cross(tok, base, base)
        tok = self.cross_norm(tok + att)
        logits = self.head(tok)[:, :, 0]
        return logits.masked_fill(pad, float("-inf"))
