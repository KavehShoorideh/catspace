"""catspace/viz/manifold.py — a uniform manifold-projection interface over
t-SNE / UMAP / VAE for the play atlas (Kaveh 2026-07-19: "let me select the
algo, and change the UI to include params for that algo").

Every projector:
  * fit(Fn) -> (N, 2) coords for the normalized background F (builds the atlas),
  * transform(Fn) -> (M, 2) OUT-OF-SAMPLE coords (the server maps the live
    board's F into the SAME map on every /project),
  * save(dir) / load(dir) round-trips the fitted state to disk,
  * note() -> a short human string for the build log.

The normalizer (zero-mean/unit-var over F) lives OUTSIDE the projector (shared
across algos) and is applied by the caller before fit/transform, exactly as the
t-SNE path did before this module existed.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

# ---- registry of per-algo params (name -> (default, cast)) so the build CLI,
# the server passthrough, and the UI all agree on the knob set for each algo.
PARAM_SPEC = {
    "tsne": {"perplexity": (40.0, float), "exaggeration": (1.0, float),
             "n_iter": (1500, int)},
    "umap": {"n_neighbors": (30, int), "min_dist": (0.1, float),
             "n_epochs": (500, int)},
    # beta<<1: a 64->2 bottleneck can't reconstruct, so beta=1 posterior-collapses
    # the latent to a point (measured xstd=0.00). Small beta keeps the latent used.
    "vae":  {"epochs": (200, int), "hidden": (256, int), "beta": (0.02, float)},
}


def clean_params(algo: str, raw: dict) -> dict:
    """Keep only the keys valid for `algo`, cast to the right type, fill defaults."""
    spec = PARAM_SPEC[algo]
    out = {}
    for k, (default, cast) in spec.items():
        v = raw.get(k, default)
        try:
            out[k] = cast(v)
        except (TypeError, ValueError):
            out[k] = default
    return out


class TSNEProjector:
    kind = "tsne"

    def __init__(self, perplexity=40.0, exaggeration=1.0, n_iter=1500, seed=0):
        self.perplexity = float(perplexity)
        self.exaggeration = float(exaggeration)   # >1 COMPRESSES; 1.0 neutral
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self._emb = None
        self._perp = perplexity

    def fit(self, Fn):
        from openTSNE import TSNE
        cap = max(5.0, len(Fn) / 3.0)             # perplexity must be << n
        self._perp = min(self.perplexity, cap)
        self._emb = TSNE(perplexity=self._perp, initialization="pca",
                         metric="cosine", exaggeration=self.exaggeration,
                         n_iter=self.n_iter, learning_rate="auto",
                         random_state=self.seed, n_jobs=-1).fit(Fn)
        return np.asarray(self._emb, dtype=np.float32)

    def transform(self, Fn):
        return np.asarray(self._emb.transform(Fn), dtype=np.float32)

    def save(self, d):
        with open(Path(d) / "projector.pkl", "wb") as f:
            pickle.dump(self._emb, f)

    @classmethod
    def load(cls, d, params):
        o = cls(**params)
        o._emb = pickle.load(open(Path(d) / "projector.pkl", "rb"))
        return o

    def note(self):
        return f"perp={self._perp:g} exag={self.exaggeration:g} iter={self.n_iter}"


class UMAPProjector:
    kind = "umap"

    def __init__(self, n_neighbors=30, min_dist=0.1, n_epochs=500, seed=0):
        self.n_neighbors = int(n_neighbors)
        self.min_dist = float(min_dist)
        self.n_epochs = int(n_epochs)
        self.seed = int(seed)
        self._reducer = None

    def fit(self, Fn):
        import umap
        # cosine metric matches the field's geometry (F@z reach is a dot product);
        # random_state pins reproducibility (forces single-thread, fine at 6k).
        self._reducer = umap.UMAP(
            n_neighbors=min(self.n_neighbors, max(2, len(Fn) - 1)),
            min_dist=self.min_dist, n_epochs=self.n_epochs, metric="cosine",
            n_components=2, random_state=self.seed)
        xy = self._reducer.fit_transform(Fn)
        return np.asarray(xy, dtype=np.float32)

    def transform(self, Fn):
        return np.asarray(self._reducer.transform(Fn), dtype=np.float32)

    def save(self, d):
        with open(Path(d) / "projector.pkl", "wb") as f:
            pickle.dump(self._reducer, f)

    @classmethod
    def load(cls, d, params):
        o = cls(**params)
        o._reducer = pickle.load(open(Path(d) / "projector.pkl", "rb"))
        return o

    def note(self):
        return f"nnb={self.n_neighbors} mind={self.min_dist:g} ep={self.n_epochs}"


class VAEProjector:
    """Compression VAE (the CompressionVAE idea, in PyTorch instead of the
    unmaintained TF package): a 64->2 variational autoencoder whose encoder mean
    IS the 2-D embedding. Naturally out-of-sample (the encoder is a function), so
    /project just runs the live F through the encoder."""
    kind = "vae"

    def __init__(self, epochs=200, hidden=256, beta=0.02, lr=1e-3, seed=0):
        self.epochs = int(epochs)
        self.hidden = int(hidden)
        self.beta = float(beta)
        self.lr = float(lr)
        self.seed = int(seed)
        self._model = None
        self._d_in = None

    def _build(self, d_in):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        h = self.hidden

        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(),
                                         nn.Linear(h, h), nn.ReLU())
                self.mu = nn.Linear(h, 2)
                self.lv = nn.Linear(h, 2)
                self.dec = nn.Sequential(nn.Linear(2, h), nn.ReLU(),
                                         nn.Linear(h, h), nn.ReLU(),
                                         nn.Linear(h, d_in))

            def encode(self, x):
                z = self.enc(x)
                return self.mu(z), self.lv(z)

            def forward(self, x):
                mu, lv = self.encode(x)
                std = torch.exp(0.5 * lv)
                z = mu + std * torch.randn_like(std)
                return self.dec(z), mu, lv

        return VAE()

    def fit(self, Fn):
        import torch
        self._d_in = Fn.shape[1]
        self._model = self._build(self._d_in)
        X = torch.from_numpy(np.asarray(Fn, dtype=np.float32))
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        n = len(X)
        bs = min(1024, n)
        g = torch.Generator().manual_seed(self.seed)
        t0 = time.time()
        self._model.train()
        warm = max(1, self.epochs // 2)               # KL annealing: 0 -> beta over
        for ep in range(self.epochs):                 # first half (avoids collapse)
            beta_t = self.beta * min(1.0, (ep + 1) / warm)
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, bs):
                xb = X[perm[i:i + bs]]
                rec, mu, lv = self._model(xb)
                recon = torch.nn.functional.mse_loss(rec, xb, reduction="mean")
                kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
                loss = recon + beta_t * kl
                opt.zero_grad(); loss.backward(); opt.step()
        self._fit_s = time.time() - t0
        return self.transform(Fn)

    def transform(self, Fn):
        import torch
        self._model.eval()
        with torch.no_grad():
            mu, _ = self._model.encode(torch.from_numpy(np.asarray(Fn, dtype=np.float32)))
        return mu.numpy().astype(np.float32)

    def save(self, d):
        import torch
        torch.save({"state": self._model.state_dict(), "d_in": self._d_in,
                    "hidden": self.hidden}, Path(d) / "projector.pt")

    @classmethod
    def load(cls, d, params):
        import torch
        o = cls(**params)
        ck = torch.load(Path(d) / "projector.pt", map_location="cpu", weights_only=False)
        o._d_in = ck["d_in"]; o.hidden = ck.get("hidden", o.hidden)
        o._model = o._build(o._d_in)
        o._model.load_state_dict(ck["state"]); o._model.eval()
        return o

    def note(self):
        return f"epochs={self.epochs} hidden={self.hidden} beta={self.beta:g}"


_REGISTRY = {"tsne": TSNEProjector, "umap": UMAPProjector, "vae": VAEProjector}


def make_projector(algo: str, params: dict, seed: int = 0):
    algo = (algo or "tsne").lower()
    if algo not in _REGISTRY:
        raise ValueError(f"unknown algo {algo!r} (have {list(_REGISTRY)})")
    return _REGISTRY[algo](**clean_params(algo, params), seed=seed)


def save_projector(proj, d, params: dict):
    """Persist the fitted projector + a manifest the server reads back."""
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    proj.save(d)
    (d / "manifest.json").write_text(json.dumps(
        {"algo": proj.kind, "params": clean_params(proj.kind, params)}))


def load_projector(d):
    """Load whatever algo was last built into `d` (reads manifest.json; falls
    back to legacy t-SNE embedding.pkl for atlases built before this module)."""
    d = Path(d)
    man = d / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        algo = m["algo"]; params = clean_params(algo, m.get("params", {}))
        return _REGISTRY[algo].load(d, params)
    # legacy: pre-manifest atlases stored the openTSNE embedding as embedding.pkl
    o = TSNEProjector()
    o._emb = pickle.load(open(d / "embedding.pkl", "rb"))
    return o
