"""
memory/store.py — the position MEMORY (Kaveh 2026-07-19): a persistent ANN
index over F-embeddings so any position can retrieve the nearest SEEN
positions and their outcomes -- a non-parametric, retrieval-based committor
alongside the parametric phead.

What goes in (provenance-tagged, per Kaveh):
  human      training positions from the Lichess shards
  selfplay   positions from self-play shards
  play_ui    positions of games completed in the play-atlas UI
  mcts_sim   search-tree lines that reached a rules-certified terminal
             ("every monte carlo simulation we carry to completion")

Outcome labels follow the certified rules ([[data/certified.py]]): result is
white-POV {+1,0,-1} with -2 = unknown; `certified` marks board-honest labels
(mate|draw|material-backed win for shard data; rule-proven terminals for
play_ui/mcts_sim entries).

IMPORTANT: embeddings are a function of the FIELD CHECKPOINT. The store
records which ckpt produced them (meta.json); querying with a different
field's F is geometric nonsense -- the loader warns, callers should rebuild.

Index: hnswlib (cosine). Metadata: aligned numpy arrays + ascii FENs, one
directory = {index.bin, meta.npz, meta.json}. Thread-safety: callers hold a
lock (the play server serializes through Engine.lock).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

SOURCES = ("human", "selfplay", "play_ui", "mcts_sim")
_SRC_ID = {s: i for i, s in enumerate(SOURCES)}
RESULT_UNKNOWN = -2


class PositionMemory:
    """Append-able cosine-ANN store of (F, fen, result, certified, source, ply)."""

    def __init__(self, dim: int, capacity: int = 500_000, ckpt_tag: str = "?",
                 ef: int = 64, m: int = 16):
        import hnswlib
        self.dim = dim
        self.ckpt_tag = ckpt_tag
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=capacity, ef_construction=200, M=m)
        self.index.set_ef(ef)
        self._lock = threading.Lock()
        self._fens: list[bytes] = []
        self._result: list[int] = []
        self._cert: list[bool] = []
        self._source: list[int] = []
        self._ply: list[int] = []
        self._dirty = 0                    # adds since last save (autosave hook)

    def __len__(self) -> int:
        return len(self._fens)

    # -- write -------------------------------------------------------------
    def add(self, F: np.ndarray, fens: list[str], results: list[int],
            certified: list[bool], source: str, plies: list[int] | None = None):
        """Append a batch. F: (n, dim) float32 (any scale; cosine index)."""
        n = len(fens)
        assert F.shape == (n, self.dim), (F.shape, n, self.dim)
        sid = _SRC_ID[source]
        plies = plies if plies is not None else [0] * n
        with self._lock:
            start = len(self._fens)
            if start + n > self.index.get_max_elements():
                self.index.resize_index(max(start + n, 2 * self.index.get_max_elements()))
            self.index.add_items(F.astype(np.float32), np.arange(start, start + n))
            self._fens += [f.encode("ascii", "replace") for f in fens]
            self._result += [int(r) for r in results]
            self._cert += [bool(c) for c in certified]
            self._source += [sid] * n
            self._ply += [int(p) for p in (plies or [0] * n)]
            self._dirty += n

    # -- read --------------------------------------------------------------
    def query(self, f: np.ndarray, k: int = 8) -> list[dict]:
        """Nearest stored positions to embedding f (dim,). Returns dicts with
        fen/result/certified/source/ply/dist (cosine distance, 0=identical)."""
        if len(self) == 0:
            return []
        k = min(k, len(self))
        with self._lock:
            labels, dists = self.index.knn_query(f.astype(np.float32)[None], k=k)
        out = []
        for lab, dist in zip(labels[0], dists[0]):
            i = int(lab)
            out.append(dict(fen=self._fens[i].decode(), result=self._result[i],
                            certified=self._cert[i], source=SOURCES[self._source[i]],
                            ply=self._ply[i], dist=round(float(dist), 4)))
        return out

    # -- persistence ---------------------------------------------------------
    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.index.save_index(str(path / "index.bin"))
            np.savez_compressed(
                path / "meta.npz",
                fens=np.array(self._fens, dtype="S128"),
                result=np.array(self._result, dtype=np.int8),
                certified=np.array(self._cert, dtype=bool),
                source=np.array(self._source, dtype=np.uint8),
                ply=np.array(self._ply, dtype=np.int32))
            (path / "meta.json").write_text(json.dumps(
                dict(dim=self.dim, n=len(self._fens), ckpt_tag=self.ckpt_tag)))
            self._dirty = 0

    @classmethod
    def load(cls, path: Path, expect_ckpt_tag: str | None = None) -> "PositionMemory":
        import hnswlib
        path = Path(path)
        info = json.loads((path / "meta.json").read_text())
        if expect_ckpt_tag is not None and info["ckpt_tag"] != expect_ckpt_tag:
            print(f"WARNING: position memory built with field '{info['ckpt_tag']}' "
                  f"but querying with '{expect_ckpt_tag}' -- neighbors will be "
                  f"geometric nonsense; rebuild the memory for this field.")
        mem = cls.__new__(cls)
        mem.dim = info["dim"]
        mem.ckpt_tag = info["ckpt_tag"]
        mem.index = hnswlib.Index(space="cosine", dim=mem.dim)
        mem.index.load_index(str(path / "index.bin"), allow_replace_deleted=False)
        mem.index.set_ef(64)
        mem._lock = threading.Lock()
        z = np.load(path / "meta.npz")
        mem._fens = list(z["fens"])
        mem._result = [int(x) for x in z["result"]]
        mem._cert = [bool(x) for x in z["certified"]]
        mem._source = [int(x) for x in z["source"]]
        mem._ply = [int(x) for x in z["ply"]]
        mem._dirty = 0
        return mem
