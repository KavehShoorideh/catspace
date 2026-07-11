"""
nn/features.py — numpy model-input builders (NO torch import): packed shard
rows -> (N,19,8,8) input planes, and the omega (opponent-model) conditioning
ids: Elo bins + clock bucket. White-POV everywhere; side-to-move is a plane.
"""
from __future__ import annotations

import numpy as np

from catspace.data.encode import decode_planes

N_PLANES = 19          # 12 pieces + stm + 4 castling + ep + halfmove
N_ELO_BINS = 11        # 10 bins of 200 over [800, 2800) + 1 unknown
N_CLOCK_BINS = 8       # 7 log-ish buckets + 1 unknown
_CLOCK_EDGES = np.array([15.0, 30.0, 60.0, 120.0, 300.0, 600.0])


def feature_planes(packed: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """(N,12) packed bitboards + (N,8) meta -> (N,19,8,8) float32 planes."""
    packed = np.atleast_2d(packed)
    meta = np.atleast_2d(meta)
    n = packed.shape[0]
    planes = decode_planes(packed).astype(np.float32)          # (N,12,8,8)
    extra = np.zeros((n, 7, 8, 8), dtype=np.float32)
    extra[:, 0] = meta[:, 0].astype(np.float32)[:, None, None]           # stm (0=W,1=B)
    for i in range(4):                                                    # K,Q,k,q rights
        extra[:, 1 + i] = meta[:, 1 + i].astype(np.float32)[:, None, None]
    ep = meta[:, 5].astype(np.int64)                                      # ep_file+1, 0=none
    has = np.flatnonzero(ep > 0)
    if has.size:
        files = ep[has] - 1
        ranks = np.where(meta[has, 0] == 0, 5, 2)              # W to move -> ep on rank 6
        extra[has, 5, ranks, files] = 1.0
    extra[:, 6] = (np.minimum(meta[:, 6], 100).astype(np.float32) / 100.0)[:, None, None]
    return np.concatenate([planes, extra], axis=1)


def elo_bin(elo: np.ndarray) -> np.ndarray:
    """Elo -> bin id: 200-wide bins over [800, 2800), clipped; 0/unknown -> 10."""
    elo = np.asarray(elo, dtype=np.int64)
    bins = np.clip((elo - 800) // 200, 0, N_ELO_BINS - 2)
    return np.where(elo <= 0, N_ELO_BINS - 1, bins)


def clock_bucket(clock: np.ndarray) -> np.ndarray:
    """Seconds remaining -> bucket id 0..6 (log-ish edges), nan/unknown -> 7."""
    clock = np.asarray(clock, dtype=np.float64)
    buckets = np.searchsorted(_CLOCK_EDGES, np.nan_to_num(clock, nan=0.0))
    return np.where(np.isnan(clock), N_CLOCK_BINS - 1, buckets).astype(np.int64)


def omega_ids(white_elo: np.ndarray, black_elo: np.ndarray, clock: np.ndarray) -> np.ndarray:
    """(N,3) int64 conditioning ids for the F side: white-Elo bin, black-Elo
    bin, clock bucket. The cone is conditioned on WHO is generating the
    dynamics (both players) and the time regime -- README lesson 1 at real
    scale. B never sees omega (goals are board-only by design)."""
    return np.stack([elo_bin(white_elo), elo_bin(black_elo), clock_bucket(clock)], axis=1)


def winprob_cp(cp: np.ndarray) -> np.ndarray:
    """White-POV centipawns -> expected score in [0,1] (lichess logistic,
    k=0.00368208). nan passes through (unannotated positions)."""
    return 1.0 / (1.0 + np.exp(-0.00368208 * np.asarray(cp, dtype=np.float64)))
