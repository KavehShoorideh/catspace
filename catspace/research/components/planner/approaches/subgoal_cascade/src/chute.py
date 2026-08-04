"""Planner component: per-move target selection + spell instrumentation.

A Planner owns WHERE to go; the Navigator owns HOW. Implementations register in
PLANNERS and must provide:
    prepare()                      -> None   (build any static structures)
    plan(phi_now, ply)             -> target region id (int)
    new_game() / finish_game(ply)  -> spell bookkeeping
    spells                         -> list[dict] instrument rows (harness reads)
    graph_line()                   -> str for the startup printout
"""
from __future__ import annotations

import numpy as np

from catspace.research.components.planner.approaches.atlas_region_stats.src.region_stats import RegionAtlas
from catspace.research.components.planner.approaches.reach_field.src.region import RegionReach


class ChutePlanner:
    """Chain-of-chutes, threshold-free (Kaveh 2026-07-29): value iteration on the
    absorbing walk over the centroid region graph; per-move plan = argmax
    P(reach g | here) * V(g); replan every our-move, switches instrumented."""

    def __init__(self, reach: RegionReach, atlas: RegionAtlas):
        self.reach = reach; self.atlas = atlas

    def prepare(self):
        a = self.atlas
        p_cc, _ = self.reach.heads(self.reach.bank)         # (G, G) centroid edges
        np.fill_diagonal(p_cc, 0.0)
        stay = np.clip(1.0 - a.fall_opp - a.fall_us, 0.0, 1.0)
        V = a.q_region.copy()
        for _ in range(200):
            cont = (p_cc * V[None, :]).max(1)
            V_new = np.maximum(a.q_region, a.fall_opp + stay * cont)
            if np.abs(V_new - V).max() < 1e-9:
                V = V_new
                break
            V = V_new
        cont = p_cc * V[None, :]
        self.next_hop = np.where(a.fall_opp + stay * cont.max(1) > a.q_region,
                                 cont.argmax(1), -1)        # -1 = convert here
        self.V = V
        self.new_game()

    def new_game(self):
        self.spell = None
        self.spells: list[dict] = []

    def finish_game(self, ply: int):
        if self.spell is not None:
            self.spell.update(outcome="end", actual=ply - self.spell["ply"])
            self.spell = None

    def chain(self, g: int) -> list[int]:
        out = [g]
        while self.next_hop[out[-1]] >= 0 and len(out) < 32:
            out.append(int(self.next_hop[out[-1]]))
        return out

    def plan(self, phi_now, ply: int) -> int:
        cur = self.reach.region_of(phi_now)
        p, plies = self.reach.heads(phi_now[None]); p, plies = p[0], plies[0]
        score = np.log(np.maximum(p, 1e-12)) + np.log(np.maximum(self.V, 1e-12))
        score[cur] = -np.inf                                 # no-op plan excluded
        g1 = int(np.argmax(score))
        if self.spell is not None:                           # close the running spell
            inc = self.spell["target"]
            if cur == inc:
                self.spell.update(outcome="hit", actual=ply - self.spell["ply"])
                self.spell = None
            elif g1 != inc:
                # flapping vs honest abandonment: margin = challenger - incumbent
                # NOW; decay = how much the incumbent fell since adoption
                self.spell.update(outcome="switch", actual=ply - self.spell["ply"],
                                  margin=float(score[g1] - score[inc]),
                                  decay=float(self.spell["loglik"] - score[inc]))
                self.spell = None
        if self.spell is None:
            self.spell = dict(ply=ply, target=g1, chain=len(self.chain(g1)),
                              loglik=float(score[g1]), pred_plies=float(plies[g1]),
                              outcome="open", actual=None)
            self.spells.append(self.spell)
        return g1

    def graph_line(self) -> str:
        a = self.atlas
        chutes = int((self.next_hop >= 0).sum())
        dep = np.array([len(self.chain(g)) for g in range(len(self.V))])
        return (f"graph: V median {np.median(self.V):.3f} p90 "
                f"{np.percentile(self.V, 90):.3f} | {chutes}/{len(self.V)} regions "
                f"continue to a next chute (rest convert in place) | chain depth "
                f"median {np.median(dep):.0f} p90 {np.percentile(dep, 90):.0f} "
                f"max {dep.max()} | fall_opp-fall_us median "
                f"{np.median(a.fall_opp - a.fall_us):+.4f}")


PLANNERS = {"chute": ChutePlanner}
