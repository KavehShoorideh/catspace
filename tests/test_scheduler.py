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


# ---- plan 56 (2a): the Rust core (analyst_core.Scheduler) ---------------------------------------------------------------------------------------------------------------------------

import analyst_core as AC  # noqa: E402


@pytest.fixture
def rust_core():
    SC.CORE_RUST = True
    try:
        yield
    finally:
        SC.CORE_RUST = False


def test_core_pop_order_exact():
    s = AC.Scheduler()
    root = s.push("node")
    s.pop()
    c = s.push("ask", root, 0, [0, 0, 3, 3])
    a = s.push("ask", root, 0, [-5, 0, 1, 1])
    b = s.push("ask", root, 0, [-5, 0, 2, 2])
    bg = s.push("ask", root, 1, [-99])
    e1 = s.push("ask", root, 0, [])
    e2 = s.push("ask", root, 0, [])
    got = [s.pop() for _ in range(6)]
    assert got == [e1, e2, a, b, c, bg]                                     # () < (-5, ...) < (0, ...): the tuple order of Python
    assert s.pop() is None and len(s) == 0


def test_core_wait_and_complete_wake_the_parent_only_when_the_last_child_ends():
    s = AC.Scheduler()
    root = s.push("node")
    assert s.pop() == root
    k1, k2 = s.push("ask", root, 0, [1]), s.push("ask", root, 0, [2])
    s.wait(root, k1)
    s.wait(root, k2)
    assert s.state(root) == "waiting" and s.waits(root) == 2 and s.parent(k1) == root and s.depth(k1) == 1
    assert s.pop() == k1
    assert s.complete(k1, "first") is None and s.state(root) == "waiting"
    assert s.pop() == k2
    assert s.complete(k2, "second") == root                                # the last child wakes the parent
    assert s.state(root) == "ready" and s.result(k1) == "first" and s.result(k2) == "second"
    assert s.pop() == root
    assert s.complete(root, 42) is None and s.result(root) == 42 and s.state(root) == "done"
    with pytest.raises(ValueError):
        s.complete(root, 1)


def test_core_reprioritise_moves_a_ready_task():
    s = AC.Scheduler()
    root = s.push("node")
    s.pop()
    a, b, c = (s.push("ask", root, 0, [i]) for i in (1, 2, 3))
    s.reprioritise(c, 0, [0])                                               # c jumps the queue
    s.reprioritise(a, 1, [0])                                               # a falls to the background
    assert [s.pop(), s.pop(), s.pop()] == [c, b, a]
    assert s.pop() is None


def test_core_stats():
    s = AC.Scheduler()
    root = s.push("node")
    s.pop()
    k = [s.push("ask", root, 0, [i]) for i in range(3)]
    for x in k:
        s.wait(root, x)
    st = s.stats()
    assert st["tasks"] == 4 and st["by_kind"] == {"node": 1, "ask": 3} and st["ready"] == 3 and st["ready_max"] == 3 and st["waiting"] == 1 and st["steps"] == 1
    s.pop()
    s.complete(k[0], None)
    st = s.stats()
    assert st["done"] == 1 and st["ready"] == 2 and st["steps"] == 2


def test_core_rules_agree_with_the_python_rules():
    for gains in ([], [5, 3, 9], [-2], [0, 0]):
        SC.CORE_RUST = False
        want = route_value(gains)
        assert AC.sched_route_value(gains) == want
    mk = lambda v, u: (v, SimpleNamespace(move=SimpleNamespace(uci=lambda: u)), None)  # noqa: E731
    done = [mk(4, "b3b4"), mk(4, "a2a4"), mk(2, "c2c3"), mk(4, "a2a4")]
    SC.CORE_RUST = False
    want = best_route(list(done))
    SC.CORE_RUST = True
    try:
        got = best_route(list(done))
        assert got[0] == want[0] and got[1].move.uci() == want[1].move.uci() == "a2a4"
        assert question_killed([]) and not question_killed(done) and route_value([5, 3, 9]) == 3
    finally:
        SC.CORE_RUST = False


@pytest.mark.parametrize("name", ["test_a_routes_value_is_the_least_over_its_replies", "test_a_questions_claim_is_its_best_route_and_the_tie_is_the_move",
                                  "test_a_question_is_killed_when_every_route_is", "test_one_surviving_route_is_the_claim",
                                  "test_the_worker_runs_children_before_the_next_sibling_and_resumes_the_parent_with_the_answer",
                                  "test_ready_tasks_are_popped_by_priority", "test_an_error_travels_up_the_graph_and_runs_the_finally_clauses_of_the_waiting_tasks"])
def test_the_synthetic_graph_gives_the_same_answers_on_the_rust_core(rust_core, name):
    globals()[name]()


def test_0lrtf_on_the_rust_core_is_the_same_as_the_python_scheduler(rust_core):
    new = decide("0LRTF", True)
    SC.CORE_RUST = False
    old = decide("0LRTF", True)
    assert new["decision"] == "Rh8+" and old == new
