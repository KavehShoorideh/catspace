"""
Plan-memory tests: availability/propose/update state machine, block-reason
recording + the two wake mechanisms (discrete event index, continuous drift
watcher) with hysteresis/cooldown, eviction, persistence round-trip, tau
calibration, MoveIdentity keying, and PlanningPolicy parity with a plain
greedy MIN readout (single MATE goal, tau=-inf => must choose identical moves).
"""
import numpy as np
import pytest

from latentchess.chain import exact_P
from latentchess.cone.embedding import GoalSpec, make_goal, reach
from latentchess.cone.tabular import TabularFB
from latentchess.domains import krk
from latentchess.game import play_game
from latentchess.opponents import RandomOpponent
from latentchess.planner.move_identity import RegionPairIdentity, SyntacticIdentity
from latentchess.planner.plans import (
    BlockReason, Plan, PlanEvent, PlanMemory, PlanStatus, PlanStore, PlanStep, calibrate_tau,
)
from latentchess.planner.policy import PlanningPolicy, TablePolicy
from latentchess.planner.readout import ReplyAgg, greedy_policy
from latentchess.scoring import TerminalScores, fill_terminal_state_scores


class StubEmb:
    """Controllable QuasimetricEmbedding: reach/F_of are dictionary lookups
    the test sets up directly, with a call counter to verify short-circuiting
    (e.g. cooldown must skip the reach_of call entirely)."""
    d = 2

    def __init__(self):
        self.scores = {}   # goal name -> np.ndarray over states
        self.F = {}         # state -> np.ndarray
        self.calls = 0

    def reach(self, idx, goal):
        self.calls += 1
        arr = self.scores[goal.name]
        return arr if idx is None else arr[np.asarray(idx)]

    def F_of(self, idx):
        return np.stack([self.F.get(int(i), np.zeros(2)) for i in np.asarray(idx)])


def G(name, region):
    return GoalSpec(name, np.asarray(region), None)


# ---------------------------------------------------------------- available/propose

def test_available_and_propose():
    emb = StubEmb()
    emb.scores["A"] = np.array([0.8])
    emb.scores["B"] = np.array([0.1])
    mem = PlanMemory(emb, [G("A", [99]), G("B", [98])], tau=0.5)

    avail = mem.available(0)
    flags = [ok for _, _, ok in avail]
    assert flags == [True, False]

    plan = mem.propose(0)
    assert plan.status is PlanStatus.ACTIVE
    assert plan.goal.name == "A"
    assert plan.feasibility0 == pytest.approx(0.8)
    assert mem.active_id == plan.plan_id


def test_propose_infeasible_below_tau():
    emb = StubEmb()
    emb.scores["A"] = np.array([0.2])
    emb.scores["B"] = np.array([0.2])
    mem = PlanMemory(emb, [G("A", [99]), G("B", [98])], tau=0.5)

    plan = mem.propose(0)
    assert plan.status is PlanStatus.INFEASIBLE
    assert plan.block is not None
    assert plan.block.rule == "unlikely_territory"
    assert plan.plan_id in mem.plans
    assert mem.active_id is None


# ---------------------------------------------------------------- update()

def test_update_progress_replan_achieved():
    emb = StubEmb()
    emb.scores["G"] = np.array([0.6, 0.7, 0.3])
    mem = PlanMemory(emb, [G("G", [999])], tau=0.0, drop_delta=0.5)
    plan = Plan(plan_id="p0", subgoals=[G("G", [999])], active_subgoal=0,
                origin_state=0, feasibility0=0.5, status=PlanStatus.ACTIVE)
    mem.plans["p0"] = plan
    mem.active_id = "p0"

    assert mem.update(plan, 0, None) is PlanEvent.PROGRESS   # 0.6 > 0.5
    assert mem.update(plan, 1, None) is PlanEvent.PROGRESS   # 0.7 > 0.6
    ev = mem.update(plan, 2, None)                            # 0.3 < 0.5*0.7
    assert ev is PlanEvent.REPLAN
    assert plan.status is PlanStatus.ABANDONED
    assert mem.active_id is None

    # separate plan: landing in the goal region -> ACHIEVED
    emb.scores["H"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9])
    goal_h = G("H", [5])
    mem2 = PlanMemory(emb, [goal_h], tau=0.0)
    plan2 = Plan(plan_id="p1", subgoals=[goal_h], active_subgoal=0,
                 origin_state=0, feasibility0=0.1, status=PlanStatus.ACTIVE)
    mem2.plans["p1"] = plan2
    mem2.active_id = "p1"
    ev2 = mem2.update(plan2, 5, None)
    assert ev2 is PlanEvent.ACHIEVED
    assert plan2.status is PlanStatus.ACHIEVED
    assert mem2.active_id is None


def test_update_stalled():
    emb = StubEmb()
    emb.scores["G"] = np.array([0.6, 0.6, 0.6])
    mem = PlanMemory(emb, [G("G", [999])], tau=0.0, drop_delta=0.5, stall_plies=3)
    plan = Plan(plan_id="p0", subgoals=[G("G", [999])], active_subgoal=0,
                origin_state=0, feasibility0=0.6, status=PlanStatus.ACTIVE)
    mem.plans["p0"] = plan
    mem.active_id = "p0"

    assert mem.update(plan, 0, None) is PlanEvent.NONE
    assert mem.update(plan, 1, None) is PlanEvent.NONE
    assert mem.update(plan, 2, None) is PlanEvent.STALLED


# ---------------------------------------------------------------- block + wake

def test_block_and_discrete_wake():
    emb = StubEmb()
    emb.scores["G"] = np.array([0.7])
    mem = PlanMemory(emb, [G("G", [999])], tau=0.5, wake_margin=0.1)
    plan = Plan(plan_id="p0", subgoals=[G("G", [999])], active_subgoal=0,
                origin_state=0, feasibility0=0.1, status=PlanStatus.ACTIVE)
    mem.plans["p0"] = plan
    mem.mark_blocked("p0", BlockReason(rule="no_midpoint", feasibility=0.1,
                                        blocked_at_state=0, refutation_key=("syn", "Nc3")))
    assert plan.status is PlanStatus.INFEASIBLE

    woken = mem.on_ply(0, [("syn", "Nc3")], None)
    assert woken == ["p0"]
    assert plan.status is PlanStatus.ACTIVE

    # a fresh plan blocked on a different key must not react to an unrelated event
    plan2 = Plan(plan_id="p1", subgoals=[G("G", [999])], active_subgoal=0,
                 origin_state=0, feasibility0=0.1, status=PlanStatus.ACTIVE)
    mem.plans["p1"] = plan2
    mem.mark_blocked("p1", BlockReason(rule="no_midpoint", feasibility=0.1,
                                        blocked_at_state=0, refutation_key=("syn", "Different")))
    woken2 = mem.on_ply(0, [("syn", "Ka1")], None)
    assert woken2 == []
    assert plan2.status is PlanStatus.INFEASIBLE


def test_wake_needs_reach_and_cooldown():
    emb = StubEmb()
    emb.scores["G"] = np.array([0.3])
    mem = PlanMemory(emb, [G("G", [999])], tau=0.5, wake_margin=0.1, cooldown_plies=5)
    plan = Plan(plan_id="p0", subgoals=[G("G", [999])], active_subgoal=0,
                origin_state=0, feasibility0=0.1, status=PlanStatus.ACTIVE)
    mem.plans["p0"] = plan
    mem.mark_blocked("p0", BlockReason(rule="no_midpoint", feasibility=0.1,
                                        blocked_at_state=0, refutation_key=("syn", "X")))

    woken = mem.on_ply(0, [("syn", "X")], None)
    assert woken == []
    assert plan.status is PlanStatus.INFEASIBLE
    assert plan.last_wake_ply == 1
    calls_after_first = emb.calls
    assert calls_after_first == 1

    woken2 = mem.on_ply(0, [("syn", "X")], None)   # within cooldown -> no re-check
    assert woken2 == []
    assert emb.calls == calls_after_first


def test_drift_wake():
    emb = StubEmb()
    emb.scores["G"] = np.array([0.3])
    mem = PlanMemory(emb, [G("G", [999])], tau=0.5, wake_margin=0.1, drift_threshold=0.5)
    plan = Plan(plan_id="p0", subgoals=[G("G", [999])], active_subgoal=0,
                origin_state=0, feasibility0=0.1, status=PlanStatus.ACTIVE)
    mem.plans["p0"] = plan
    mem.mark_blocked("p0", BlockReason(rule="no_midpoint", feasibility=0.1, blocked_at_state=0,
                                        blocked_F=np.array([0.0, 0.0]), delta=np.array([1.0, 0.0])))

    woken = mem.on_ply(0, [], np.array([0.4, 0.0]))   # drift 0.4 < 0.5 -> not a candidate
    assert woken == []
    assert plan.status is PlanStatus.INFEASIBLE

    emb.scores["G"] = np.array([0.9])   # now raise reach for the second check
    woken2 = mem.on_ply(0, [], np.array([2.0, 0.0]))   # drift 2.0 >= 0.5 -> candidate, reach passes
    assert woken2 == ["p0"]
    assert plan.status is PlanStatus.ACTIVE


def test_hysteresis():
    emb = StubEmb()
    tau, margin = 0.5, 0.2
    emb.scores["G"] = np.array([tau + margin / 2])   # above tau, below tau+margin
    mem = PlanMemory(emb, [G("G", [999])], tau=tau, wake_margin=margin)
    plan = Plan(plan_id="p0", subgoals=[G("G", [999])], active_subgoal=0,
                origin_state=0, feasibility0=0.1, status=PlanStatus.ACTIVE)
    mem.plans["p0"] = plan
    mem.mark_blocked("p0", BlockReason(rule="no_midpoint", feasibility=0.1,
                                        blocked_at_state=0, refutation_key=("syn", "X")))
    woken = mem.on_ply(0, [("syn", "X")], None)
    assert woken == []
    assert plan.status is PlanStatus.INFEASIBLE


# ---------------------------------------------------------------- eviction

def test_eviction():
    emb = StubEmb()
    emb.scores["G"] = np.array([0.0])
    mem = PlanMemory(emb, [G("G", [999])], tau=0.0, k_max=3)
    plans = {
        "p0": Plan("p0", [G("G", [999])], 0, 0, 0.5, PlanStatus.ACTIVE),
        "p1": Plan("p1", [G("G", [999])], 0, 0, 0.9, PlanStatus.ABANDONED),
        "p2": Plan("p2", [G("G", [999])], 0, 0, 0.1, PlanStatus.ABANDONED),
        "p3": Plan("p3", [G("G", [999])], 0, 0, 0.5, PlanStatus.INFEASIBLE),
        "p4": Plan("p4", [G("G", [999])], 0, 0, 0.99, PlanStatus.ACHIEVED),
    }
    mem.plans.update(plans)
    mem.active_id = "p0"
    mem._evict()

    assert len(mem.plans) == 3
    assert "p0" in mem.plans
    assert "p1" not in mem.plans and "p2" not in mem.plans   # both ABANDONED evicted first
    assert set(mem.plans) == {"p0", "p3", "p4"}


# ---------------------------------------------------------------- persistence

def test_plan_store_roundtrip(tmp_path):
    g1 = G("MATE", [5, 6])
    g1.z = np.array([1.0, 2.0])
    g2 = G("VIA", [7])
    plan1 = Plan("p0", [g1], 0, 3, 0.4, PlanStatus.ACTIVE,
                 trace=[PlanStep(3, 10, 0.4, 0), PlanStep(4, 11, 0.5, 1)])
    plan2 = Plan("p1", [g2], 0, 2, 0.1, PlanStatus.INFEASIBLE,
                 block=BlockReason(rule="unlikely_territory", feasibility=0.1, blocked_at_state=2))

    store = PlanStore(tmp_path / "plans")
    store.append(plan1)
    store.append(plan2)
    store.save_goal_z([g1, g2])

    loaded = store.load_all()
    assert [p.plan_id for p in loaded] == ["p0", "p1"]
    assert [p.status for p in loaded] == [PlanStatus.ACTIVE, PlanStatus.INFEASIBLE]
    assert loaded[0].subgoals[0].name == "MATE"
    assert np.array_equal(loaded[0].subgoals[0].region, np.array([5, 6]))
    assert len(loaded[0].trace) == 2
    assert loaded[1].block is not None and loaded[1].block.rule == "unlikely_territory"

    z = np.load(tmp_path / "plans" / "z_store.npz")
    assert np.array_equal(z["MATE"], np.array([1.0, 2.0]))


# ---------------------------------------------------------------- calibrate_tau

def test_calibrate_tau():
    rng = np.random.default_rng(0)
    won = rng.normal(1.0, 0.1, size=200)
    lost = rng.normal(0.0, 0.1, size=200)
    reach_live = np.concatenate([won, lost])
    won_mask = np.concatenate([np.ones(200, bool), np.zeros(200, bool)])

    tau = calibrate_tau(reach_live, won_mask)
    assert 0.2 < tau < 0.8
    acc = ((reach_live >= tau) == won_mask).mean()
    assert acc > 0.95


# ---------------------------------------------------------------- MoveIdentity

def _state_of_move(chain, mid):
    return int(np.searchsorted(chain.move_ptr, mid, side="right") - 1)


def test_move_identity_keys():
    chain = krk.build_chain()
    ongoing_mid = int(np.where(chain.move_kind == 0)[0][0])
    mate_mid = int(np.where(chain.move_kind == 1)[0][0])
    s_ongoing = _state_of_move(chain, ongoing_mid)
    s_mate = _state_of_move(chain, mate_mid)

    syn = SyntacticIdentity()
    assert syn.key(chain, s_ongoing, ongoing_mid) == ("syn", chain.move_names[ongoing_mid])

    tokens = np.zeros(chain.n_live, dtype=int)
    region = RegionPairIdentity(tokens)
    assert region.key(chain, s_ongoing, ongoing_mid) == ("rgn", 0, 0)
    key_mate = region.key(chain, s_mate, mate_mid)
    assert key_mate == ("rgn", 0, "T", 1)


# ---------------------------------------------------------------- PlanningPolicy parity

def test_planning_policy_parity():
    chain = krk.build_chain()
    P = exact_P(chain)
    emb = TabularFB.fit(P, gamma=0.98, d=16, seed=0)
    ts = TerminalScores.big()
    goal = make_goal("MATE", np.array([chain.terminals.mate]), emb)

    scores = fill_terminal_state_scores(reach(emb, goal, None), chain, ts)
    table = greedy_policy(scores, chain, ReplyAgg.MIN, ts)
    plain = TablePolicy(table)

    memory = PlanMemory(emb, [goal], tau=-1e18)
    planning = PlanningPolicy(chain, emb, memory, ts, agg=ReplyAgg.MIN, depth=1)

    starts = np.random.default_rng(42).integers(0, chain.n_live, size=20)
    for i, s0 in enumerate(starts):
        rec_plain = play_game(chain, plain, RandomOpponent(), int(s0), cap=80,
                               rng=np.random.default_rng(1000 + i))
        rec_planning = play_game(chain, planning, RandomOpponent(), int(s0), cap=80,
                                  rng=np.random.default_rng(1000 + i))
        assert rec_planning.move_ids == rec_plain.move_ids
