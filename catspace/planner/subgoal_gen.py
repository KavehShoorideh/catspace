"""catspace/planner/subgoal_gen.py -- M4: the subgoal GENERATOR wiring the M3 pieces together.

Per decision point it produces the two lists the optionality portfolio consumes -- G_me (approach
targets: reachable where THEY err more) and G_opp (avoid zones: reachable where WE err more) --
picks/holds the ACTIVE plan with hysteresis (select_active_plan), and logs intent to the
PlanStore ledger (the M4 steering verdict reads intent vs realization from there).

Distances for the portfolio math: the field speaks probabilities, the portfolio speaks distances;
the bridge is d := -log P(reach) (so soft_reach at beta=1 is log expected-reach -- a Jensen
optionality bonus over the plan set, exactly the built math).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from catspace.planner.optionality import select_active_plan


@dataclass
class PlanContext:
    plan_id: int
    active_cell: int
    cells_me: np.ndarray      # (K,) composite cell ids
    w_me: np.ndarray          # (K,) weights (approach scores)
    cells_opp: np.ndarray
    w_opp: np.ndarray
    p_now_me: np.ndarray      # (K,) reach probs of MY cells from the current position
    p_now_opp: np.ndarray


class SubgoalGenerator:
    def __init__(self, ranker, store, top_k: int = 8, switch_margin: float = 0.15):
        self.rk = ranker; self.store = store
        self.k = top_k; self.margin = switch_margin

    def plan(self, phi_now, game_key: str, ply: int, side: str,
             elo_self: float, elo_oppo: float, z_self=None, z_opp=None, n_obs: int = 0):
        r = self.rk.rank(phi_now, elo_self, elo_oppo, z_self=z_self, z_opp=z_opp,
                         n_obs=n_obs, top=self.k)
        cells_me, cells_opp = r["top"], r["avoid"]
        w_me = np.maximum(r["score"][cells_me], 1e-9)
        w_opp = np.maximum(r["score_avoid"][cells_opp], 1e-9)
        # hysteresis: hold the incumbent unless a challenger clears the margin
        incumbent_cell, _ = self.store.last_active(game_key)
        vals = r["score"][cells_me]
        inc_idx = int(np.where(cells_me == incumbent_cell)[0][0]) \
            if incumbent_cell in cells_me else None
        act_idx = select_active_plan(vals, incumbent=inc_idx, switch_margin=self.margin)
        active = int(cells_me[act_idx])
        reason = ("hold" if incumbent_cell == active else
                  ("cold" if incumbent_cell is None else "switch"))
        plan_id = self.store.log_plan(
            game_key, ply, side, active,
            {"me": cells_me.tolist(), "opp": cells_opp.tolist()},
            {"me": np.round(w_me, 6).tolist(), "opp": np.round(w_opp, 6).tolist()}, reason)
        pr = r["p_reach"]
        return PlanContext(plan_id, active, cells_me, w_me, cells_opp, w_opp,
                           pr[cells_me], pr[cells_opp])

    def neglogp(self, phis, cells, elo_self, elo_oppo, z_self=None, z_opp=None, n_obs=0):
        """d(board -> cell) = -log P(reach cell | board, ctx) for a BATCH of boards.
        phis (B,64), cells (K,) -> (B,K) distances for optionality.move_scores.
        CAUTION (2026-07-29 iter-2 diagnosis): at small p the log AMPLIFIES encoder jitter into
        ±0.2-nat per-successor swings -- use reach_p for move shaping; keep this for retrieval."""
        p = self.rk.p_composite(phis, elo_self, elo_oppo, z_self, z_opp, n_obs)[:, cells]
        return -np.log(np.maximum(p, 1e-9))

    def reach_p(self, phis, cells, elo_self, elo_oppo, z_self=None, z_opp=None, n_obs=0):
        """P(reach cell | board, ctx) (B,K) -- PROBABILITY space: bounded, no log amplification."""
        return self.rk.p_composite(phis, elo_self, elo_oppo, z_self, z_opp, n_obs)[:, cells]
