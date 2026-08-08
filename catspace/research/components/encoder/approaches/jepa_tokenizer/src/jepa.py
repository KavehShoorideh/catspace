"""catspace/encoder/jepa.py -- the anchored-JEPA encoder stack (Kaveh's draft §3).

Components (T1 scope -- trained jointly, three losses, one clamp):
  tokenize()       : chess.Board -> (64,) square piece-ids + (6,) globals
  JepaEncoder      : 64 square tokens + globals -> phi in R^d (relational transformer)
  DynPredictor     : (phi(s), a) -> predicted target embedding of s' (L_dyn)
  AnyHazardHead    : per-horizon-bucket ANY-event hazard queries (aggregate kappa_0;
                     per-atom keys arrive at T2-T4 over this same interface)
  DestinationHead  : d(s) in Delta(C x {W,D,L}) -- TB-clamped at the boundary (L_dest)

EMA target encoder + stop-gradient + weaker predictor are the L_dyn guards; the
labelled terms and the exact terminal clamp protect the rest (paper Fig 3b).
"""
from __future__ import annotations

import chess
import numpy as np
import torch
import torch.nn as nn

# token vocab: 0 empty, 1-6 white PNBRQK, 7-12 black pnbrqk
_PIECE = {(chess.PAWN, True): 1, (chess.KNIGHT, True): 2, (chess.BISHOP, True): 3,
          (chess.ROOK, True): 4, (chess.QUEEN, True): 5, (chess.KING, True): 6,
          (chess.PAWN, False): 7, (chess.KNIGHT, False): 8, (chess.BISHOP, False): 9,
          (chess.ROOK, False): 10, (chess.QUEEN, False): 11, (chess.KING, False): 12}


def tokenize(board: chess.Board):
    """-> (tok (64,) uint8, glob (6,) uint8): pieces; [turn, K, Q, k, q, ep_file+1]."""
    tok = np.zeros(64, np.uint8)
    for sq, pc in board.piece_map().items():
        tok[sq] = _PIECE[(pc.piece_type, pc.color)]
    glob = np.array([board.turn,
                     board.has_kingside_castling_rights(chess.WHITE),
                     board.has_queenside_castling_rights(chess.WHITE),
                     board.has_kingside_castling_rights(chess.BLACK),
                     board.has_queenside_castling_rights(chess.BLACK),
                     0 if board.ep_square is None else chess.square_file(board.ep_square) + 1],
                    np.uint8)
    return tok, glob


# COLOR MIRROR (2026-08-08, the lc0/NNUE adaptation): rank-flip + piece-color swap + glob swap
# is an EXACT involution of chess. lc0 gets color symmetry by canonicalizing every input to the
# side to move; NNUE by two shared-weight perspectives (black's = the rank-flip color-swap of
# white's). Our frame is absolute (white-POV labels need it), so the same symmetry enters as a
# per-step training involution instead: mirrored batch + swapped W/L labels.
_MIRROR_PIECE = np.array([0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6], np.uint8)
_MIRROR_SQ = np.arange(64) ^ 56                     # chess.square_mirror, vectorized


def mirror_arrays(tok, glob):
    """(N,64) tok, (N,6) glob -> color-mirrored copies. Result/W-L labels must be swapped by
    the caller; mover-POV labels are mirror-invariant. ep FILE survives a rank flip unchanged."""
    tok2 = _MIRROR_PIECE[tok][:, _MIRROR_SQ]
    glob2 = glob[:, [0, 3, 4, 1, 2, 5]].copy()
    glob2[:, 0] = 1 - glob[:, 0]
    return tok2, glob2


def mirror_squares(sq):
    """move from/to squares under the color mirror (promo piece is unchanged)."""
    return sq ^ 56


def move_ids(move: chess.Move):
    """-> (from, to, promo 0-4) for the action embedding."""
    promo = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
    return move.from_square, move.to_square, promo.get(move.promotion, 4)


class JepaEncoder(nn.Module):
    """64 square tokens (+globals broadcast) -> phi in R^d. Relational: standard
    self-attention over squares with a learned relative-geometry bias."""

    def __init__(self, d: int = 256, layers: int = 6, heads: int = 8):
        super().__init__()
        self.d = d
        self.piece_emb = nn.Embedding(13, d)
        self.sq_emb = nn.Embedding(64, d)
        self.glob_proj = nn.Linear(6, d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d,
                                         batch_first=True, norm_first=True,
                                         dropout=0.0, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, num_layers=layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.out = nn.LayerNorm(d)

    def forward(self, tok, glob):
        B = tok.shape[0]
        x = self.piece_emb(tok.long()) + self.sq_emb.weight[None, :, :]
        g = self.glob_proj(glob.float())[:, None, :]
        x = torch.cat([self.cls.expand(B, -1, -1) + g, x], 1)     # (B, 65, d)
        return self.out(self.tr(x)[:, 0])                          # CLS -> phi


class DynPredictor(nn.Module):
    """(phi(s), a) -> predicted phi_target(s'). DELIBERATELY weaker than the encoder
    (2-layer MLP) -- one of the L_dyn anti-collapse guards."""

    def __init__(self, d: int = 256):
        super().__init__()
        self.from_emb = nn.Embedding(64, 32); self.to_emb = nn.Embedding(64, 32)
        self.promo_emb = nn.Embedding(5, 8)
        self.net = nn.Sequential(nn.Linear(d + 72, 512), nn.GELU(), nn.Linear(512, d))

    def forward(self, phi, a):
        act = torch.cat([self.from_emb(a[:, 0].long()), self.to_emb(a[:, 1].long()),
                         self.promo_emb(a[:, 2].long())], -1)
        return self.net(torch.cat([phi, act], -1))


class AnyHazardHead(nn.Module):
    """per-bucket ANY-checkpoint hazard: lambda(h | s, omega). One query per bucket;
    at T4 the same queries meet per-atom keys -- this head is kappa_0 (aggregate)."""

    def __init__(self, d: int = 256, d_ctx: int = 2, H: int = 8):
        super().__init__()
        self.H = H
        self.net = nn.Sequential(nn.Linear(d + d_ctx, 256), nn.GELU(),
                                 nn.Linear(256, H))

    def forward(self, phi, ctx):
        return self.net(torch.cat([phi, ctx], -1))                 # logits (B, H)


class DestinationHead(nn.Module):
    """d(s) in Delta(C x {W,D,L}): which material class the game first crosses the
    boundary in, x outcome there. Clamped to tablebase one-hots at the boundary."""

    def __init__(self, d: int = 256, n_class: int = 151):
        super().__init__()
        self.n_class = n_class
        self.net = nn.Sequential(nn.Linear(d, 256), nn.GELU(),
                                 nn.Linear(256, n_class * 3))

    def forward(self, phi):
        return self.net(phi).view(-1, self.n_class, 3)


class JepaT1(nn.Module):
    def __init__(self, d: int = 256, layers: int = 6, heads: int = 8,
                 H: int = 8, n_class: int = 151, ema_m: float = 0.996):
        super().__init__()
        self.enc = JepaEncoder(d, layers, heads)
        self.tgt = JepaEncoder(d, layers, heads)
        self.tgt.load_state_dict(self.enc.state_dict())
        for p in self.tgt.parameters():
            p.requires_grad_(False)
        self.dyn = DynPredictor(d)
        self.haz = AnyHazardHead(d, H=H)
        self.dest = DestinationHead(d, n_class)
        self.ema_m = ema_m

    @torch.no_grad()
    def ema_update(self):
        for p, q in zip(self.enc.parameters(), self.tgt.parameters()):
            q.mul_(self.ema_m).add_(p, alpha=1 - self.ema_m)
