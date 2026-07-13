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

2026-07-11 `quasimetric=True` mode (JOURNAL.md, MRN -- Liu/Feng/Liu/Stone,
AAAI 2023): score(f,g) = -d(f,g) + r(f,g) instead of a plain dot product.
`d` is a REAL metric by construction (Euclidean distance between
per-dimension-rescaled embeddings: non-negative, symmetric, and satisfies
the triangle inequality for ANY three points in R^d regardless of which
network produced them -- that's a property of the norm on the shared
codomain, not of embed_F/embed_B individually), so multi-hop plans compose
correctly by construction instead of only when training data happened to
show that exact hop. `r` is a small unconstrained bilinear residual for
whatever directed/non-metric structure remains (the literature's own
finding: an unconstrained network provably cannot represent a consistent
quasimetric on its own -- PQE, Wang & Isola -- hence needing `d` to be
metric BY CONSTRUCTION, not learned freeform). `metric_scale` inits to
all-ones and `W` inits to zero, so a fresh quasimetric run starts with
score == -||f-g||_2 on the SAME unit-normalized embeddings the plain
cosine mode uses (monotonic in cosine similarity, d^2 = 2-2cos on unit
vectors) -- a smooth, not-arbitrarily-different starting point, same
spirit as this file's existing normalization discipline. Config-gated
(old checkpoints have quasimetric=False and never see metric_scale/W, so
they remain byte-for-byte unaffected); when False, score()/score_matrix()
reduce to exactly the prior `f @ b.T` behavior.
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
                 tau: float = 0.1, seed: int = 0, quasimetric: bool = False):
        torch.manual_seed(seed)          # one seed, sequential construction:
        super().__init__()               # encF and encB draw DIFFERENT inits
        self.config = dict(d=d, channels=channels, blocks=blocks, enc_out=enc_out,
                           dh=dh, omega_dim=omega_dim, tau=tau, seed=seed,
                           quasimetric=quasimetric)
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
        self.quasimetric = quasimetric
        if quasimetric:
            self.metric_scale = nn.Parameter(torch.ones(d))
            self.W = nn.Parameter(torch.zeros(d, d))

    def embed_F(self, planes: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        h = self.encF(planes)
        o = torch.cat([self.emb_we(omega[:, 0]), self.emb_be(omega[:, 1]),
                       self.emb_clk(omega[:, 2])], dim=1)
        return nn.functional.normalize(self.headF(torch.cat([h, o], dim=1)), dim=1)

    def embed_B(self, planes: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.headB(self.encB(planes)), dim=1)

    def loss_fn(self, planes_s: torch.Tensor, omega_s: torch.Tensor,
                planes_g: torch.Tensor, ply_gap: torch.Tensor | None = None,
                ply_gap_weight: float = 0.05, ply_gap_scale: float = 50.0
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """InfoNCE with in-batch negatives; returns (loss, top1 retrieval acc).

        2026-07-12 ply-gap calibration (Kaveh: "if the future leads to a mate
        for me, that's a good future... maybe we have to search deeper"):
        in-batch retrieval only enforces RELATIVE ranking (is g_true closer
        than the other g's in this batch?) -- nothing calibrates the
        ABSOLUTE scale of the quasimetric distance to anything real, so
        "down material with no path back" and "down material but
        recoverable" can score identically as long as their RANKING within
        a batch happens to work out. `ply_gap` (real anchor->goal ply
        distance, already flowing through the pipeline unused) lets d(f,g)
        regress toward the ACTUAL number of plies real play took to get
        from s to g -- calibrating distance to mean roughly "moves of real
        play between here and there", for winning AND losing trajectories
        alike (losing ones are what teach the geometry of "no way back" --
        which is why the 2026-07-11 winner-POV training filter was removed
        the day this term landed), not just an uncalibrated relative score. Only meaningful in quasimetric mode (there is no
        `d` to calibrate otherwise); silently ignored when quasimetric=False
        so non-quasimetric callers don't need to change."""
        f = self.embed_F(planes_s, omega_s)
        b = self.embed_B(planes_g)
        logits = self.score_matrix(f, b) / self.tau
        target = torch.arange(len(f), device=logits.device)
        loss = nn.functional.cross_entropy(logits, target)
        top1 = (logits.argmax(dim=1) == target).float().mean()
        if self.quasimetric and ply_gap is not None:
            d_true = self.distance_matrix(f, b).diagonal()
            target_d = ply_gap.to(d_true.dtype) / ply_gap_scale
            loss = loss + ply_gap_weight * nn.functional.mse_loss(d_true, target_d)
        return loss, top1

    def distance_matrix(self, f: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """All-pairs metric distance, (N,d) x (M,d) -> (N,M). Requires
        quasimetric=True (the metric only exists in that mode). Exposed
        separately from score_matrix so the triangle inequality can be
        tested directly on `d` alone -- `r` in score_matrix is an
        unconstrained residual that is NOT expected to respect it."""
        assert self.quasimetric, "distance_matrix requires quasimetric=True"
        fs, bs = f * self.metric_scale, b * self.metric_scale
        d2 = ((fs * fs).sum(1, keepdim=True) + (bs * bs).sum(1)[None, :]
              - 2.0 * (fs @ bs.T))
        return torch.sqrt(d2.clamp_min(1e-9))

    def score_matrix(self, f: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """All-pairs score, (N,d) x (M,d) -> (N,M). Plain dot product unless
        quasimetric=True, in which case -d(f,b)+r(f,b) (see module docstring)."""
        if not self.quasimetric:
            return f @ b.T
        return f @ self.W @ b.T - self.distance_matrix(f, b)

    def score(self, f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """(N,d) states against ONE fixed goal embedding z (d,) -> (N,).
        The search/readout-time counterpart of score_matrix -- used wherever
        the old code did `F(s) @ z` (a fixed zgoal like zMATE_W)."""
        if not self.quasimetric:
            return f @ z
        return self.score_matrix(f, z[None, :])[:, 0]

    @staticmethod
    def reach_z(f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return f @ z


def save_ckpt(fb: TorchFB, path, step: int = 0, opt: torch.optim.Optimizer | None = None,
              zgoals: dict | None = None, provenance: dict | None = None) -> None:
    """zgoals: name -> (d,) numpy/tensor goal vectors (e.g. MATE_W) travel
    with the model -- a field without its goals is not a planner artifact.
    provenance: catspace.audit.build_provenance(...) output, if the caller
    has one -- lets audit_checkpoint() certify this checkpoint never saw
    Stockfish-derived signal without re-deriving it from training logs."""
    payload = dict(state_dict=fb.state_dict(), config=fb.config, step=step,
                   zgoals={k: torch.as_tensor(v).cpu() for k, v in (zgoals or {}).items()},
                   provenance=provenance)
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
