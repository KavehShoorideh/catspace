"""PLAN 56 (1): the chase as a task graph on a priority queue, run by one worker loop. The rules the values travel by (min over replies, best route, kill when every route dies) on a small
synthetic graph; the worker loop's order and its error propagation; and the decisions of 0LRTF and 07Z0P with the scheduler on (the five-target diff is scripts/compare_scheduler.py)."""
from types import SimpleNamespace

import pytest

import catspace.incubator.analyst  # noqa: F401
from catspace.incubator.analyst.discovery import agenda as AG
from catspace.incubator.analyst.discovery import analyse as AN
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import scheduler as SC
from catspace.incubator.analyst.discovery.scheduler import Scheduler, T, best_route, question_killed, route_value


# ---- a synthetic chase: question -> routes -> replies, each reply a leaf value or None (killed: no claim of ours and no material) ----------------------------------------------------

def reply_task(log, name, value):
    log.append(("reply", name))
    return value
    yield                                                                   # (a generator: a leaf task that waits on nothing)


def route_task(log, name, replies):
    """A route: every reply is a task; the route's value is the least (a killed reply kills the route)."""
    log.append(("route", name))
    gains = []
    for i, v in enumerate(replies):
        got = yield T("reply", reply_task(log, f"{name}/{i}", v))
        if got is None:
            return dict(value=0, killed=f"after reply {i}")
        gains.append(got)
    return dict(value=route_value(gains), killed=None)


def question_task(log, routes):
    """A question: its routes are tasks, one after the other; its claim is the best route worth at least 1; killed when none is."""
    done = []
    for name, replies in routes.items():
        res = yield T("route", route_task(log, name, replies))
        if res["killed"] is None and res["value"] >= 1:
            done.append((res["value"], SimpleNamespace(move=SimpleNamespace(uci=lambda n=name: n)), res))
    if question_killed(done):
        return None
    value, r, _ = best_route(done)
    return value, r.move.uci()


def run_question(routes):
    log = []
    sched = Scheduler(record=True)
    out = sched.run(T("ask", question_task(log, routes)))
    return out, log, sched


def test_a_routes_value_is_the_least_over_its_replies():
    out, _, _ = run_question({"Rh8+": [5, 3, 9]})
    assert out == (3, "Rh8+")
    assert route_value([]) == 0


def test_a_questions_claim_is_its_best_route_and_the_tie_is_the_move():
    out, _, _ = run_question({"Qb3": [4, 6], "Qa4": [4, 9], "Qc2": [2, 9]})
    assert out == (4, "Qa4")                                                # Qb3 and Qa4 are worth 4: the uci breaks the tie, never the order asked


def test_a_question_is_killed_when_every_route_is():
    out, log, _ = run_question({"Qb3": [4, None, 9], "Qa4": [None], "Qc2": [0, 7]})
    assert out is None
    assert ("reply", "Qb3/2") not in log                                    # a killed route reads no more replies


def test_one_surviving_route_is_the_claim():
    out, _, _ = run_question({"Qb3": [4, None], "Qa4": [7, 8]})
    assert out == (7, "Qa4")


def test_the_worker_runs_children_before_the_next_sibling_and_resumes_the_parent_with_the_answer():
    out, log, sched = run_question({"A": [1, 2], "B": [3]})
    assert log == [("route", "A"), ("reply", "A/0"), ("reply", "A/1"), ("route", "B"), ("reply", "B/0")]       # depth first: today's order
    assert out == (3, "B") and sched.ready_max == 1                         # one task is ready at a time: a parent needs its child's value before it goes on
    root = sched.root
    assert [c.kind for c in root.children] == ["route", "route"] and [len(c.children) for c in root.children] == [2, 1]
    assert all(t.done for t in (root, *root.children))


def test_ready_tasks_are_popped_by_priority():
    order = []

    def leaf(name):
        order.append(name)
        return name
        yield

    def parent():
        got = yield [T("ask", leaf("c"), prio=(0, 0, 3, 3)), T("ask", leaf("a"), prio=(-5, 0, 1, 1)), T("ask", leaf("b"), prio=(-5, 0, 2, 2))]
        return got

    Scheduler().run(T("node", parent()))
    assert order == ["a", "b", "c"]                                         # the bigger stake first, then the depth / creation sequence: the heap order of the agenda


def test_an_error_travels_up_the_graph_and_runs_the_finally_clauses_of_the_waiting_tasks():
    state = []

    def boom():
        raise ValueError("deep")
        yield

    def mid():
        try:
            yield T("reply", boom())
        finally:
            state.append("mid closed")

    def top():
        try:
            yield T("route", mid())
        finally:
            state.append("top closed")

    with pytest.raises(ValueError):
        Scheduler().run(T("ask", top()))
    assert state == ["mid closed", "top closed"]


# ---- the chase under the flag ------------------------------------------------------------------------------------------------------------------------------------------------------

def decide(puzzle_id, sched):
    AG.SCHEDULER = sched
    try:
        p = next(x for name in ("dev_20", "dev_300", "fresh_300") for x in PA.load_set(name) if x["id"] == puzzle_id)
        r = AN.analyse(p, quiet=True, memory=False, log=lambda *a, **k: None)
    finally:
        AG.SCHEDULER = False
    r.pop("secs", None)
    return r


def test_0lrtf_decides_rh8_with_the_scheduler_and_the_same_counters():
    old, new = decide("0LRTF", False), decide("0LRTF", True)
    print(f"OBS scheduler 0LRTF: decision {new['decision']}, inner agendas {new['chase'][0]}, inner questions {new['chase'][4]}")
    assert new["decision"] == "Rh8+" and new["match"]
    assert old == new                                                       # decision, claims count, kills, gaps, every counter of the chase


def test_07z0p_decides_qxd5_with_the_scheduler_and_the_same_counters():
    old, new = decide("07Z0P", False), decide("07Z0P", True)
    print(f"OBS scheduler 07Z0P: decision {new['decision']}, inner agendas {new['chase'][0]}, inner questions {new['chase'][4]}")
    assert new["decision"] == "Qxd5+" and new["match"]
    assert old == new


def test_the_scheduler_is_off_by_default_and_leaves_no_state_on_the_context():
    assert AG.SCHEDULER is False
    ag = AG.Agenda(AN.state_of("2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 6 62", "w"))
    AG.SCHEDULER = True
    try:
        ag.ask()
    finally:
        AG.SCHEDULER = False
    assert ag.ctx.sched is None and ag.ctx.stats["steps"] > 0 and ag.ctx.stats["node"] >= 1
    assert ag.decision().what == "Rh8+"
