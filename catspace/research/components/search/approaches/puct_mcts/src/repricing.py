"""Fast-field re-pricing of a slow distance estimate.

Split out of the old top-level two_field.py in the 2026-08-03 restructure: the
scoring formula is a planner concern (planner/two_perspective_scoring), while
re-pricing a distance from live memory evidence is consumed at search time, so it
lives here as a guidance hook (JOURNAL 2026-07-14 effective-distance override).
"""
from __future__ import annotations

import numpy as np


def effective_distance(d_slow: float, mem: dict | None, lam: float = 8.0,
                       s_min: float = 0.9, floor: float = 1e-3) -> tuple[float, float]:
    """Blend slow distance with fast-field evidence. Returns (d_eff, p_var).
    mem = MemoryField.query() dict or None."""
    if mem is None or mem["support"] < s_min:
        return d_slow, 0.0
    p = max(mem["p_hat"], floor)
    plies = mem["plies"] if mem["plies"] is not None else d_slow * 50.0
    return (plies + lam * (-np.log(p))) / 50.0, mem["p_var"]


def repricing_guidance(base_reach_fn, memory_field, lam: float = 8.0, s_min: float = 0.9):
    """Wrap a reach oracle so its estimate is re-priced wherever memory has support.

    Returns a reach_fn with the same (boards -> np.ndarray) contract MCTS expects,
    plus a `.p_var` attribute holding the per-board uncertainty from the last call
    (high p_var = neither estimate is trusted; the caller should search deeper).
    """
    def reach_fn(boards):
        d = np.asarray(base_reach_fn(boards), dtype=float)
        if memory_field is None:
            reach_fn.p_var = np.zeros(len(d))
            return d
        out = np.empty_like(d)
        pvar = np.empty_like(d)
        for i, b in enumerate(boards):
            out[i], pvar[i] = effective_distance(
                d[i], memory_field.query(b), lam=lam, s_min=s_min)
        reach_fn.p_var = pvar
        return out

    reach_fn.p_var = np.zeros(0)
    return reach_fn
