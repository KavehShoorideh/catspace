"""
viz/projection.py — the pluggable 2D-projection seam.

Every interactive viewer downstream consumes ONLY per-state [x, y] pairs
(confirmed by reading krkn_viewer_template.html: no code path references
t-SNE by name, only `n.xy`). This module formalizes the fit/transform
contract so t-SNE, PCA, UMAP, or a learned 2D head are interchangeable
behind one FittedMap, replacing the ad hoc `tsne_cache.pkl` pickle.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class Projection2D(Protocol):
    name: str

    def fit(self, X: np.ndarray) -> "Projection2D": ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "Projection2D": ...


@dataclass
class PCAProjection:
    name: str = "pca"
    mean: np.ndarray | None = None
    components: np.ndarray | None = None   # (d, 2)
    _fit_points: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCAProjection":
        self.mean = X.mean(0)
        _, _, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        self.components = Vt[:2].T
        self._fit_points = (X - self.mean) @ self.components
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) @ self.components

    def fit_points(self) -> np.ndarray:
        return self._fit_points

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, components=self.components, fit_points=self._fit_points)

    @classmethod
    def load(cls, path: Path) -> "PCAProjection":
        z = np.load(path if str(path).endswith(".npz") else f"{path}.npz")
        return cls(mean=z["mean"], components=z["components"], _fit_points=z["fit_points"])


@dataclass
class TSNEProjection:
    """openTSNE fit + out-of-sample transform (same recipe as the original
    tsne_maps.py/tsne_cones.py: perplexity=40, PCA init, seed 0)."""
    name: str = "tsne"
    perplexity: float = 40.0
    seed: int = 0
    _embedding: object = None   # openTSNE TSNEEmbedding, holds the fitted state

    def fit(self, X: np.ndarray) -> "TSNEProjection":
        from openTSNE import TSNE
        self._embedding = TSNE(perplexity=self.perplexity, initialization="pca",
                                random_state=self.seed, n_jobs=1).fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._embedding.transform(X))

    def fit_points(self) -> np.ndarray:
        return np.asarray(self._embedding)

    def save(self, path: Path) -> None:
        path = Path(path)
        with open(path if path.suffix == ".pkl" else path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self._embedding, f)

    @classmethod
    def load(cls, path: Path) -> "TSNEProjection":
        path = Path(path)
        with open(path if path.suffix == ".pkl" else path.with_suffix(".pkl"), "rb") as f:
            embedding = pickle.load(f)
        obj = cls()
        obj._embedding = embedding
        return obj


PROJECTIONS: dict[str, type] = {"pca": PCAProjection, "tsne": TSNEProjection}
try:
    import umap  # noqa: F401

    @dataclass
    class UMAPProjection:
        name: str = "umap"
        seed: int = 0
        _reducer: object = None

        def fit(self, X: np.ndarray) -> "UMAPProjection":
            import umap as _umap
            self._reducer = _umap.UMAP(random_state=self.seed).fit(X)
            return self

        def transform(self, X: np.ndarray) -> np.ndarray:
            return np.asarray(self._reducer.transform(X))

        def fit_points(self) -> np.ndarray:
            return np.asarray(self._reducer.embedding_)

        def save(self, path: Path) -> None:
            path = Path(path)
            with open(path if path.suffix == ".pkl" else path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(self._reducer, f)

        @classmethod
        def load(cls, path: Path) -> "UMAPProjection":
            path = Path(path)
            with open(path if path.suffix == ".pkl" else path.with_suffix(".pkl"), "rb") as f:
                reducer = pickle.load(f)
            obj = cls(); obj._reducer = reducer
            return obj

    PROJECTIONS["umap"] = UMAPProjection
except ImportError:
    pass


@dataclass
class Normalizer:
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, eps: float = 1e-9) -> "Normalizer":
        return cls(mu=X.mean(0), sd=X.std(0) + eps)

    def apply(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd


def stratified_fit_index(dtm: np.ndarray, won: np.ndarray, n_live: int,
                          near_dtm: int = 3, sizes=(9000, 7000, 3000, 3000),
                          extra_pool: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """The fit-subsample recipe from tsne_maps.py/tsne_cones.py: stratify by
    won/drawn/near-mate (within the primary stratum, indices [0, n_live) or
    [0, len(dtm)) if smaller), plus an optional extra pool (e.g. the KRk
    sub-stratum in KRkn's union chain)."""
    rng = np.random.default_rng(seed)
    near = dtm <= near_dtm
    n_won, n_drawn, n_near, n_extra = sizes

    def sample(mask, k):
        idx = np.where(mask)[0]
        k = min(k, len(idx))
        return rng.choice(idx, k, replace=False)

    idx_won = sample(won & ~near, n_won)
    idx_drawn = sample(~won, n_drawn)
    idx_near = sample(near, n_near)
    parts = [idx_won, idx_drawn, idx_near]
    if extra_pool is not None and n_extra:
        k = min(n_extra, len(extra_pool))
        parts.append(rng.choice(extra_pool, k, replace=False))
    return np.concatenate(parts)


@dataclass
class FittedMap:
    """Formalizes the tsne_cache.pkl contract: a projection fitted on a
    subsample, plus the normalizer used to prepare F before fitting/
    transforming, plus the fit indices (for drawing the background cloud)."""
    projection: Projection2D
    normalizer: Normalizer
    fit_idx: np.ndarray

    def fit_points(self) -> np.ndarray:
        return self.projection.fit_points()

    def project(self, F: np.ndarray, idx: np.ndarray) -> np.ndarray:
        X = self.normalizer.apply(F[idx])
        return self.projection.transform(X)

    def save(self, dir: Path) -> None:
        dir = Path(dir); dir.mkdir(parents=True, exist_ok=True)
        self.projection.save(dir / "projection")
        np.savez(dir / "meta.npz", mu=self.normalizer.mu, sd=self.normalizer.sd,
                 fit_idx=self.fit_idx, name=self.projection.name)

    @classmethod
    def load(cls, dir: Path) -> "FittedMap":
        dir = Path(dir)
        meta = np.load(dir / "meta.npz")
        name = str(meta["name"])
        projection = PROJECTIONS[name].load(dir / "projection")
        normalizer = Normalizer(mu=meta["mu"], sd=meta["sd"])
        return cls(projection=projection, normalizer=normalizer, fit_idx=meta["fit_idx"])


def fit_map(F: np.ndarray, dtm: np.ndarray, won: np.ndarray, kind: str = "tsne",
            extra_pool: np.ndarray | None = None, seed: int = 0, **proj_kwargs) -> FittedMap:
    normalizer = Normalizer.fit(F)
    Fn = normalizer.apply(F)
    fit_idx = stratified_fit_index(dtm, won, len(dtm), extra_pool=extra_pool, seed=seed)
    projection = PROJECTIONS[kind](**proj_kwargs).fit(Fn[fit_idx])
    return FittedMap(projection=projection, normalizer=normalizer, fit_idx=fit_idx)
