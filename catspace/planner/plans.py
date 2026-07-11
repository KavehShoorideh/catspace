"""
planner/plans.py — plan memory: plans as first-class, persisted objects that
know their feasibility, remember WHY they were blocked, and can be woken when
that reason might no longer hold.

A Plan holds an ORDERED LIST of subgoals (`subgoals`); Phase 5 always uses a
single-subgoal list (the trivial "tree" of depth 1) -- recursive decomposition
into real multi-hop chains is deferred to the M1.5 research phase. Everything
else here (blocking, wake triggers, eviction, persistence) is load-bearing now.

Two wake mechanisms, checked every ply via `PlanMemory.on_ply`:
  - a discrete EVENT INDEX: a plan blocked on a specific MoveIdentity key (the
    refuting reply) is re-checked exactly when that key recurs (or a stratum
    crossing happens) -- O(1) lookup, no scanning of unrelated plans.
  - a continuous DRIFT WATCHER: a plan blocked with a known enabling direction
    Δ is re-checked when the live embedding has drifted far enough along Δ
    since the block -- catches the opponent fixing the problem for us.
Both are hysteretic (wake threshold = tau + wake_margin, strictly above the
block threshold tau) and cooled down (a woken-and-failed plan won't be
re-checked again for `cooldown_plies`), so a plan can't thrash on noise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from catspace.cone.embedding import GoalSpec, QuasimetricEmbedding, reach as reach_fn


class PlanStatus(Enum):
    ACTIVE = "active"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"
    INFEASIBLE = "infeasible"


class PlanEvent(Enum):
    PROGRESS = "progress"
    STALLED = "stalled"
    REPLAN = "replan"
    ACHIEVED = "achieved"
    WOKE = "woke"
    NONE = "none"


@dataclass
class BlockReason:
    rule: str                              # "no_midpoint"|"unlikely_territory"|"dry_out"|"budget"
    feasibility: float                     # reach value at the moment of blocking
    blocked_at_state: int
    blocked_F: np.ndarray | None = None    # F(s_blocked) snapshot -- drift-watcher origin
    delta: np.ndarray | None = None        # enabling direction; None => no drift watcher
    refutation_key: tuple | None = None    # MoveIdentity key; None => no discrete trigger
    detail: str = ""


@dataclass
class PlanStep:
    state: int
    move_id: int | None
    reach: float
    ply: int


@dataclass
class Plan:
    plan_id: str
    subgoals: list             # list[GoalSpec]; length 1 in Phase 5
    active_subgoal: int
    origin_state: int
    feasibility0: float
    status: PlanStatus
    trace: list = field(default_factory=list)          # list[PlanStep]
    block: BlockReason | None = None
    last_wake_ply: int = -10 ** 9

    @property
    def goal(self) -> GoalSpec:
        return self.subgoals[self.active_subgoal]

    def to_json(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "status": self.status.name,
            "origin_state": self.origin_state,
            "feasibility0": self.feasibility0,
            "active_subgoal": self.active_subgoal,
            "subgoals": [{"name": g.name, "region": np.asarray(g.region).tolist()}
                         for g in self.subgoals],
            "trace": [[st.state, st.move_id, st.reach, st.ply] for st in self.trace],
            "block": None if self.block is None else {
                "rule": self.block.rule,
                "feasibility": self.block.feasibility,
                "blocked_at_state": self.block.blocked_at_state,
                "detail": self.block.detail,
            },
        }

    @classmethod
    def from_json(cls, d: dict) -> "Plan":
        subgoals = [GoalSpec(name=sg["name"], region=np.array(sg["region"]), z=None)
                    for sg in d["subgoals"]]
        trace = [PlanStep(state=t[0], move_id=t[1], reach=t[2], ply=t[3]) for t in d["trace"]]
        block = None
        if d["block"] is not None:
            b = d["block"]
            block = BlockReason(rule=b["rule"], feasibility=b["feasibility"],
                                 blocked_at_state=b["blocked_at_state"],
                                 detail=b.get("detail", ""))
        return cls(plan_id=d["plan_id"], subgoals=subgoals, active_subgoal=d["active_subgoal"],
                   origin_state=d["origin_state"], feasibility0=d["feasibility0"],
                   status=PlanStatus[d["status"]], trace=trace, block=block)


_EVICT_PRIORITY = {PlanStatus.ABANDONED: 0, PlanStatus.INFEASIBLE: 1, PlanStatus.ACHIEVED: 2}


class PlanMemory:
    def __init__(self, emb: QuasimetricEmbedding, goals: list, tau: float,
                 drop_delta: float = 0.5, stall_plies: int = 6, k_max: int = 16,
                 wake_margin: float = 0.1, drift_threshold: float = 0.5,
                 cooldown_plies: int = 8):
        self.emb = emb
        self.goals = goals
        self.tau = tau
        self.drop_delta = drop_delta
        self.stall_plies = stall_plies
        self.k_max = k_max
        self.wake_margin = wake_margin
        self.drift_threshold = drift_threshold
        self.cooldown_plies = cooldown_plies

        self.plans: dict[str, Plan] = {}
        self.active_id: str | None = None
        self._counter = 0
        self._ply = 0
        self._event_index: dict[tuple, set] = {}
        self._drift_ids: set = set()

    def reach_of(self, goal: GoalSpec, s: int) -> float:
        return float(reach_fn(self.emb, goal, np.array([s]))[0])

    def _F_of(self, s: int):
        if hasattr(self.emb, "F_of"):
            return self.emb.F_of(np.array([s]))[0]
        return None

    def available(self, s: int) -> list:
        return [(g, r, r >= self.tau) for g, r in
                ((g, self.reach_of(g, s)) for g in self.goals)]

    def propose(self, s: int) -> Plan:
        avail = self.available(s)
        g, r, ok = max(avail, key=lambda t: t[1])
        plan_id = f"p{self._counter:04d}"
        self._counter += 1
        status = PlanStatus.ACTIVE if ok else PlanStatus.INFEASIBLE
        plan = Plan(plan_id=plan_id, subgoals=[g], active_subgoal=0, origin_state=s,
                    feasibility0=r, status=status)
        if not ok:
            plan.block = BlockReason(rule="unlikely_territory", feasibility=r,
                                      blocked_at_state=s, blocked_F=self._F_of(s))
            self._register_block(plan)
        self.plans[plan_id] = plan
        if status is PlanStatus.ACTIVE:
            self.active_id = plan_id
        self._evict()
        return plan

    def update(self, plan: Plan, s: int, move_id: int | None) -> PlanEvent:
        r = self.reach_of(plan.goal, s)
        prior_max = max((st.reach for st in plan.trace), default=plan.feasibility0)
        prior_trace = list(plan.trace)
        plan.trace.append(PlanStep(state=s, move_id=move_id, reach=r, ply=self._ply))

        if bool(np.any(np.asarray(plan.goal.region) == s)):
            plan.status = PlanStatus.ACHIEVED
            if self.active_id == plan.plan_id:
                self.active_id = None
            return PlanEvent.ACHIEVED

        running_max = max(prior_max, r)
        if r < (1.0 - self.drop_delta) * running_max or r < self.tau:
            plan.status = PlanStatus.ABANDONED
            if self.active_id == plan.plan_id:
                self.active_id = None
            return PlanEvent.REPLAN

        if r > prior_max + 1e-12:
            return PlanEvent.PROGRESS

        if len(plan.trace) >= self.stall_plies:
            window = plan.trace[-self.stall_plies:]
            pre_window = prior_trace[:len(prior_trace) - (self.stall_plies - 1)] \
                if len(prior_trace) >= self.stall_plies - 1 else []
            baseline = max((st.reach for st in pre_window), default=plan.feasibility0)
            if max(st.reach for st in window) <= baseline + 1e-12:
                return PlanEvent.STALLED

        return PlanEvent.NONE

    def mark_blocked(self, plan_id: str, reason: BlockReason) -> None:
        plan = self.plans[plan_id]
        plan.status = PlanStatus.INFEASIBLE
        plan.block = reason
        self._register_block(plan)
        if self.active_id == plan_id:
            self.active_id = None

    def _register_block(self, plan: Plan) -> None:
        reason = plan.block
        if reason.refutation_key is not None:
            self._event_index.setdefault(reason.refutation_key, set()).add(plan.plan_id)
        if reason.delta is not None and reason.blocked_F is not None:
            self._drift_ids.add(plan.plan_id)

    def _unregister_block(self, plan_id: str) -> None:
        self._drift_ids.discard(plan_id)
        for listeners in self._event_index.values():
            listeners.discard(plan_id)

    def on_ply(self, s: int, events: list, F_now: np.ndarray | None) -> list:
        """Advance the ply counter and re-check any INFEASIBLE plans whose
        wake condition (a listened-for event, or embedding drift along a known
        enabling direction) just fired. Returns the list of plan_ids woken to
        ACTIVE this call."""
        self._ply += 1
        candidates: set = set()
        for ev in events:
            candidates |= self._event_index.get(ev, set())
        if F_now is not None:
            for pid in self._drift_ids:
                plan = self.plans.get(pid)
                if plan is None or plan.status is not PlanStatus.INFEASIBLE:
                    continue
                b = plan.block
                if b is None or b.delta is None or b.blocked_F is None:
                    continue
                drift = float((F_now - b.blocked_F) @ b.delta)
                if drift >= self.drift_threshold:
                    candidates.add(pid)

        woken = []
        for pid in sorted(candidates):
            plan = self.plans.get(pid)
            if plan is None or plan.status is not PlanStatus.INFEASIBLE:
                continue
            if self._ply - plan.last_wake_ply < self.cooldown_plies:
                continue
            plan.last_wake_ply = self._ply
            r = self.reach_of(plan.goal, s)
            if r >= self.tau + self.wake_margin:
                plan.status = PlanStatus.ACTIVE
                self._unregister_block(pid)
                woken.append(pid)
        return woken

    def _evict(self) -> None:
        while len(self.plans) > self.k_max:
            candidates = [p for p in self.plans.values()
                          if p.status is not PlanStatus.ACTIVE and p.plan_id != self.active_id]
            if not candidates:
                break
            victim = min(candidates, key=lambda p: (_EVICT_PRIORITY.get(p.status, 3), p.feasibility0))
            del self.plans[victim.plan_id]
            self._unregister_block(victim.plan_id)


class PlanStore:
    def __init__(self, dir: Path):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def append(self, plan: Plan) -> None:
        with open(self.dir / "plans.jsonl", "a") as f:
            f.write(json.dumps(plan.to_json()) + "\n")

    def save_goal_z(self, goals: list) -> None:
        arrays = {g.name: g.z for g in goals if g.z is not None}
        if arrays:
            np.savez(self.dir / "z_store.npz", **arrays)

    def load_all(self) -> list:
        """Reloaded plans are records for analysis, not live listeners --
        BlockReason.blocked_F/delta/refutation_key are not serialized (they
        reference the live embedding/identity scheme), so a reloaded blocked
        plan cannot be woken by PlanMemory.on_ply; re-propose it instead."""
        path = self.dir / "plans.jsonl"
        if not path.exists():
            return []
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(Plan.from_json(json.loads(line)))
        return out


def calibrate_tau(reach_live: np.ndarray, won_mask: np.ndarray) -> float:
    """Youden-J-optimal threshold on the WIN/DRAW frontier: maximize
    TPR - FPR of `reach_live >= tau` against the ground-truth `won_mask`."""
    candidates = np.quantile(reach_live, np.linspace(0.01, 0.99, 199))
    best_t, best_j = float(candidates[0]), -np.inf
    for t in candidates:
        tpr = float((reach_live[won_mask] >= t).mean())
        fpr = float((reach_live[~won_mask] >= t).mean())
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t
