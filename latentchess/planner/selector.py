"""
planner/selector.py — the PlanSelector protocol: which plan to act on this ply.

GreedyReach is the trivial baseline (current-reach-maximizing, no learning).
RL-based selectors (options/semi-MDP framing, MCTS-over-plans) are deferred to
the M1.5 research phase; the protocol and registry exist now so they plug in
without touching PlanningPolicy.
"""
from __future__ import annotations

from typing import Protocol

from latentchess.planner.plans import Plan, PlanMemory, PlanStatus


class PlanSelector(Protocol):
    def select(self, s: int, memory: PlanMemory) -> Plan | None:
        ...


class GreedyReach:
    """Keep the current active plan while it's ACTIVE; otherwise switch to
    whichever ACTIVE plan currently has the highest reach; otherwise propose a
    fresh plan (returning it only if feasible)."""

    def select(self, s: int, memory: PlanMemory) -> Plan | None:
        if memory.active_id is not None:
            current = memory.plans.get(memory.active_id)
            if current is not None and current.status is PlanStatus.ACTIVE:
                return current

        active_plans = [p for p in memory.plans.values() if p.status is PlanStatus.ACTIVE]
        if active_plans:
            best = max(sorted(active_plans, key=lambda p: p.plan_id),
                       key=lambda p: memory.reach_of(p.goal, s))
            memory.active_id = best.plan_id
            return best

        proposed = memory.propose(s)
        return proposed if proposed.status is PlanStatus.ACTIVE else None


SELECTORS: dict[str, type] = {"greedy_reach": GreedyReach}
