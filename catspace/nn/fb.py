"""
nn/fb.py — TorchFB: the full-board Forward-Backward embedding. Two
BoardEncoders (F-side, B-side) + MLP heads; F is conditioned on omega
(white/black Elo bins + clock bucket -- README lesson 1: the cone is
opponent-conditioned), B is board-only (goals are positions, not regimes).

The InfoNCE loss follows cone/neural.py (logits = F @ B.T / tau,
one-directional cross-entropy over in-batch negatives) with ONE deliberate
deviation: embeddings are L2-NORMALIZED (cosine InfoNCE). The toy recipe's
unnormalized dot was only safe because one-hot toy encodings have constant
input norm; real boards lose material over a game, activation norms shrink
with ply, and unnormalized F.B inherits that decline -- measured as a -0.92
spearman(ply, reach-to-anything) on won AND lost games alike before
normalization (2026-07-11 diagnostic).

Real boards have no state indices, so TorchFB does NOT implement the chain
QuasimetricEmbedding protocol; the seam here is embed_F / embed_B / reach_z
(same shapes as EncodedNeuralFB's precomputed arrays).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from catspace.nn.encoder import BoardEncoder
from catspace.nn.features import N_CLOCK_BINS, N_ELO_BINS, N_PLANES


class TorchFB(nn.Module):
    def __init__(self, d: int = 64, channels: int = 64, blocks: int = 6,
                 enc_out: int = 256, dh: int = 512, omega_dim: int = 16,
                 tau: float = 0.1, seed: int = 0):
        torch.manual_seed(seed)          # one seed, sequential construction:
        super().__init__()               # encF and encB draw DIFFERENT inits
        self.config = dict(d=d, channels=channels, blocks=blocks, enc_out=enc_out,
                           dh=dh, omega_dim=omega_dim, tau=tau, seed=seed)
        self.encF = BoardEncoder(N_PLANES, channels, blocks, enc_out)
        self.encB = BoardEncoder(N_PLANES, channels, blocks, enc_out)
        self.emb_we = nn.Embedding(N_ELO_BINS, omega_dim)
        self.emb_be = nn.Embedding(N_ELO_BINS, omega_dim)
        self.emb_clk = nn.Embedding(N_CLOCK_BINS, omega_dim)
        self.headF = nn.Sequential(nn.Linear(enc_out + 3 * omega_dim, dh), nn.ReLU(),
                                   nn.Linear(dh, d))
        self.headB = nn.Sequential(nn.Linear(enc_out, dh), nn.ReLU(), nn.Linear(dh, d))
        self.tau = tau
        self.d = d

    def embed_F(self, planes: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        h = self.encF(planes)
        o = torch.cat([self.emb_we(omega[:, 0]), self.emb_be(omega[:, 1]),
                       self.emb_clk(omega[:, 2])], dim=1)
        return nn.functional.normalize(self.headF(torch.cat([h, o], dim=1)), dim=1)

    def embed_B(self, planes: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.headB(self.encB(planes)), dim=1)

    def loss_fn(self, planes_s: torch.Tensor, omega_s: torch.Tensor,
                planes_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """InfoNCE with in-batch negatives; returns (loss, top1 retrieval acc)."""
        f = self.embed_F(planes_s, omega_s)
        b = self.embed_B(planes_g)
        logits = (f @ b.T) / self.tau
        target = torch.arange(len(f), device=logits.device)
        loss = nn.functional.cross_entropy(logits, target)
        top1 = (logits.argmax(dim=1) == target).float().mean()
        return loss, top1

    @staticmethod
    def reach_z(f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return f @ z


def save_ckpt(fb: TorchFB, path, step: int = 0, opt: torch.optim.Optimizer | None = None,
              zgoals: dict | None = None) -> None:
    """zgoals: name -> (d,) numpy/tensor goal vectors (e.g. MATE_W) travel
    with the model -- a field without its goals is not a planner artifact."""
    payload = dict(state_dict=fb.state_dict(), config=fb.config, step=step,
                   zgoals={k: torch.as_tensor(v).cpu() for k, v in (zgoals or {}).items()})
    if opt is not None:
        payload["opt_state"] = opt.state_dict()
    path = Path(path)                      # atomic: an interrupted save must not
    tmp = path.with_suffix(path.suffix + ".tmp")   # corrupt the previous checkpoint
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_ckpt(path, device: str = "cpu") -> tuple[TorchFB, dict]:
    """Returns (model, payload). payload keeps step/opt_state/zgoals."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    fb = TorchFB(**payload["config"])
    fb.load_state_dict(payload["state_dict"])
    fb.to(device)
    return fb, payload


def pick_device(arg: str = "auto") -> str:
    if arg != "auto":
        return arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
