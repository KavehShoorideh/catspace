"""catspace/style/dataio.py -- load the M2b feature cache, transparently handling BOTH layouts:
  * legacy single file  cache.npz            (one np.savez of everything)
  * resumable shard dir  cache/  meta.npz + shard_0000.npz, shard_0001.npz, ...
The shard layout (catspace/research/components/planner/approaches/opponent_model/experiments/m2b_cache.py) is crash-safe & resumable: each shard is a contiguous
position range written atomically (temp-then-rename); a restart skips shards that already exist.
Feature arrays concatenate in shard order, which matches the global metadata order in meta.npz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FEATURE_KEYS = ("cand_idx", "cand_logp", "played_slot", "win_prob", "phi", "valid")


def load_cache(path) -> dict:
    """Return a dict of numpy arrays (features + metadata), whichever layout `path` is."""
    p = Path(path)
    if p.is_dir():
        meta = dict(np.load(p / "meta.npz", allow_pickle=True))
        shards = sorted(p.glob("shard_*.npz"))
        if not shards:
            raise FileNotFoundError(f"no shards in {p} (cache incomplete?)")
        feats = {k: [] for k in FEATURE_KEYS}
        for s in shards:
            z = np.load(s)
            for k in FEATURE_KEYS:
                feats[k].append(z[k])
        out = {k: np.concatenate(v) for k, v in feats.items()}
        out.update(meta)
        n = len(out["cand_idx"])
        if int(meta["N"]) != n:
            raise ValueError(f"cache {p} incomplete: meta N={int(meta['N'])} but {n} feature rows "
                             f"across {len(shards)} shards -- resume the precompute to finish it.")
        return out
    return dict(np.load(path, allow_pickle=True))
