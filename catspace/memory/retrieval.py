"""catspace/memory/retrieval.py — composed retrieval distance to a goal / region.

The non-parametric distance-to-goal (Kaveh 2026-07-19): mate / win / draw are
SURFACES, not poles. To estimate d(s -> goal) we look up nearby KNOWN positions
(waypoints g) and compose the trusted SHORT hop d(F(s), B(g)) with g's KNOWN
distance-to-goal d(g -> goal), taking the min over waypoints:

    d_hat(s -> goal) = min_g [ d(F(s), B(g)) + d(g -> goal) ]

The field is trusted only for the short hop; d(g->goal) is grounded truth
(exact tablebase DTM, or 0 for a terminal). This is the triangle inequality made
non-parametric, and it is robust because it never trusts the field's mushy
long-range distance. One primitive, reused by:
  * the DTM training hinge (regress d_hat toward the true dtm),
  * the play readout / navigate engine (value = -d_hat to the goal surface),
  * the propagation diagnostic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from catspace.nn.features import feature_planes


class WaypointBank:
    """A goal SURFACE / region represented as waypoints: the goal-side embeddings
    B(g) plus each waypoint's KNOWN distance-to-goal d(g->goal). refresh() re-embeds
    B as the field drifts during training (the DTMs are constant)."""

    def __init__(self, packed, meta, d_to_goal, label: str = ""):
        self.packed = np.asarray(packed)
        self.meta = np.asarray(meta)
        self.d_to_goal = np.asarray(d_to_goal, dtype=np.float32)
        self.label = label
        self._B = None
        self._d = None

    def __len__(self):
        return len(self.d_to_goal)

    def refresh(self, fb, device="cpu", scale: float = 1.0):
        """(Re)embed the waypoints' B side (detached) and cache d(g->goal)/scale."""
        with torch.no_grad():
            self._B = fb.embed_B(
                torch.from_numpy(feature_planes(self.packed, self.meta)).to(device)
            ).detach()
        self._d = torch.from_numpy(self.d_to_goal).to(device) / scale
        return self

    @property
    def B(self):
        return self._B


def composed_distance(fb, F_query, bank: WaypointBank, k: int | None = None):
    """d_hat(s->goal) = min_g[ d(F(s), B(g)) + d(g->goal) ] over the bank.

    F_query: (n, d) field embeddings. If `k` is given, restrict to the k waypoints
    nearest by the field hop d(F(s), B(g)) BEFORE composing -- "find the top
    neighbors, then the distance to and through them, then the min" (Kaveh). k=None
    composes over the whole bank. Returns (n,) tensor. bank.refresh(fb) first.
    """
    if bank.B is None:
        raise ValueError("call bank.refresh(fb, device) before composed_distance")
    D = fb.distance_matrix(F_query, bank.B)                 # (n, K) short hops
    if k is not None and k < D.shape[1]:
        idx = torch.topk(D, k, dim=1, largest=False).indices  # k nearest neighbours
        D = torch.gather(D, 1, idx)
        dg = bank._d[idx]                                   # (n, k)
    else:
        dg = bank._d[None, :]                              # (1, K) broadcast
    return (D + dg).min(dim=1).values                      # min through the neighbours


def dtm_waypoint_bank(dtm_npz, n: int, seed: int = 0, force_low_frac: float = 0.25):
    """Build a WHITE-WIN surface bank from tablebase DTM data (d(g->mate)=dtm).
    Forces the lowest-dtm positions in as near-mate anchors so the composition
    always has a short path to the mate surface."""
    dz = np.load(dtm_npz) if isinstance(dtm_npz, (str, Path)) else dtm_npz
    rng = np.random.default_rng(seed)
    n = min(n, len(dz["dtm"]))
    low = np.argsort(dz["dtm"])[: max(1, int(n * force_low_frac))]
    perm = rng.permutation(len(dz["dtm"]))
    rest = perm[~np.isin(perm, low)][: n - len(low)]
    idx = np.concatenate([low, rest])
    return WaypointBank(dz["packed"][idx], dz["meta"][idx], dz["dtm"][idx], "W-dtm")


def terminal_bank(packed, meta, label: str = ""):
    """A surface of TERMINAL positions (mates / draws): d(g->goal)=0."""
    packed = np.asarray(packed)
    return WaypointBank(packed, np.asarray(meta), np.zeros(len(packed), np.float32), label)
