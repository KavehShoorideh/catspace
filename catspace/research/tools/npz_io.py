"""Loading .npz without the lazy-decompression trap.

`np.load(x.npz)` hands back an NpzFile, which re-reads AND re-decompresses the whole
array on EVERY `z["key"]`. That is invisible: the results are correct, the code is
idiomatic, and it is silently 10-40x slow. Measured here 2026-08-25 -- a 737 KB array
indexed inside a triple-nested loop was decompressed 118,000 times per shard, turning a
3-second phase into 115 seconds.

The fix that generalises is not "remember to hoist the read". It is a drop-in whose
`__getitem__` decompresses at most once, so the call site keeps the natural spelling and
the trap cannot be written:

    from catspace.research.tools.npz_io import load_npz
    z = load_npz(path)
    for i in rows:
        use(z["packed"][i])       # one decompression, not len(rows)

`shards.py` solved this locally years ago by binding every array up front
(`data = {k: npz[k] for k in npz.files}`). That works but reads eagerly -- wrong when a
file holds several large arrays and you want one. This caches per key instead, so you
pay for exactly what you touch, once.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


class CachedNpz(Mapping):
    """Mapping view over an .npz that decompresses each array at most once.

    Implements Mapping, so it substitutes for an NpzFile anywhere that indexes by key,
    iterates `.files`, or does `k in z`.
    """

    __slots__ = ("_z", "_cache", "path")

    def __init__(self, path, **kw):
        self.path = str(path)
        self._z = np.load(self.path, **kw)
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, key: str) -> np.ndarray:
        arr = self._cache.get(key)
        if arr is None:
            arr = self._z[key]
            self._cache[key] = arr
        return arr

    def __iter__(self):
        return iter(self._z.files)

    def __len__(self) -> int:
        return len(self._z.files)

    @property
    def files(self) -> list[str]:
        return list(self._z.files)

    def close(self) -> None:
        self._cache.clear()
        try:
            self._z.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        return (f"CachedNpz({self.path!r}, {len(self._z.files)} arrays, "
                f"{len(self._cache)} materialised)")


def load_npz(path, **kw):
    """Drop-in for np.load that is safe to index inside a loop.

    Returns a CachedNpz for .npz; for .npy (or anything np.load resolves to a plain
    array) returns exactly what np.load returns, so this is a safe universal
    replacement rather than something the caller has to choose between.
    """
    if str(path).endswith(".npz"):
        return CachedNpz(path, **kw)
    return np.load(path, **kw)
