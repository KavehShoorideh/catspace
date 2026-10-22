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
    assert old == new          # plan 64.2: the decision itself is asserted under the TIME budget (tests/test_offbuilt64.py): with the chain items ON the count fuse no longer reaches Qxd5+ (the in-check chase of the mate hunts costs ~2x), a stand-in budget



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


# ---- plan 56 (2b): the route reader (analyst_core.read_route) as a handler ----------------------------------------------------------------------------------------------------------

import chess  # noqa: E402

READ_FENS = {
    "07Z0P": "2r2rk1/1p4bp/6p1/pP1qpp2/6PP/P1n1BQ2/2B2R2/5RK1 w - - 1 29",                    # exchanges close; checks among the replies
    "0LRTF": "2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 6 62",                                          # a promotion race: mates in one among the replies
    "back_rank": "3r2k1/5ppp/8/8/8/8/5PPP/R2R2K1 w - - 0 1",                                 # replies that leave us no move (they mate us / stalemate), mates in one, exchanges, open
    "scholar": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "in_check": "4k3/8/8/8/8/8/3PPP2/r3K3 w - - 0 1",                                        # the side to move is in check
    "middlegame": "r3k2r/ppp2ppp/2n5/3q4/3P4/2N2N2/PPP2PPP/R2QK2R w KQkq - 0 1",
}


def reference_rows(fen, uci, white, min_stake=1, need_aware=False, promote=False, target=1, goal_stake=None, skip=()):
    """The reading order and the kinds, derived with the `Pos` methods of plan 54 and agenda.py's rules, in the Python of `_chase_route_body` / `_closed_by_rule`."""
    P0 = AC.Pos(fen)
    P1 = P0.push(uci)
    shift = P1.balance(white) - P0.balance(white)
    b = chess.Board(fen)
    route_capture = b.is_capture(chess.Move.from_uci(uci))
    rows = [(shift + d, u, chk, cap, rec, m1) for u, d, chk, cap, rec, m1 in P1.replies()]
    rows.sort(key=lambda t: (0 if t[4] else 1 if t[2] else 2 if t[3] else 3, t[0], t[1]))
    out = []
    for dbal, u, chk, cap, rec, m1 in rows:
        P2 = P1.push(u)
        kind, best, need, f2 = "open", 0, 0, None
        if u in skip:
            kind = "skip"
        elif P2.legal_count() == 0:
            kind = "dead"
        elif m1:
            kind = "mate1"
        elif not promote and dbal < 1 and not P2.is_check():
            best = P2.exchange()
            if best >= 1 - dbal and not (need_aware and dbal + best < min_stake):
                kind = "exch"
        if kind in ("exch", "open"):
            f2 = P2.fen()
            need = min(8, 1 - dbal) if promote and dbal <= 0 else 8 if promote else target - dbal if dbal < target else 100
        out.append((u, dbal, kind, best if kind == "exch" else 0, chk, f2, need, route_capture and dbal >= 0, route_capture and rec,
                    kind == "exch" and goal_stake is not None and dbal + best < goal_stake))
    return out


@pytest.mark.parametrize("name", list(READ_FENS))
@pytest.mark.parametrize("promote", [False, True])
def test_the_route_reader_gives_every_reply_the_kind_the_python_body_derives(name, promote):
    fen = READ_FENS[name]
    b = chess.Board(fen)
    kinds = set()
    for m in b.legal_moves:
        rr = AC.read_route(fen, m.uci(), b.turn, 3, False, promote, 3, 5, 100, 8, [], [], None)
        assert rr["standing"] is None
        assert [tuple(r) for r in rr["rows"]] == reference_rows(fen, m.uci(), b.turn, 3, False, promote, 3, 5)
        kinds |= {r[2] for r in rr["rows"]}
    if name in ("back_rank", "scholar", "middlegame"):
        print(f"OBS route reader {name} promote={promote}: kinds {sorted(kinds)}")
    if name == "back_rank" and not promote:
        assert {"dead", "mate1", "exch", "open"} <= kinds                       # all four kinds on one position


def test_the_route_reader_need_aware_and_skips_and_the_checks():
    fen = READ_FENS["07Z0P"]
    b = chess.Board(fen)
    m = next(m for m in b.legal_moves if b.san(m) == "Qxd5+")
    first = AC.read_route(fen, m.uci(), True, 6, True, False, 6, None, 100, 8, [], [], None)["rows"]
    assert [tuple(r) for r in first] == reference_rows(fen, m.uci(), True, 6, True, False, 6)
    assert any(r[4] for mv in b.legal_moves for r in AC.read_route(fen, mv.uci(), True, 6, True, False, 6, None, 100, 8, [], [], None)["rows"])      # some reply of theirs gives check
    skip = [first[0][0], first[-1][0]]
    got = AC.read_route(fen, m.uci(), True, 6, True, False, 6, None, 100, 8, [skip[0]], [skip[1]], None)["rows"]
    assert [tuple(r) for r in got] == reference_rows(fen, m.uci(), True, 6, True, False, 6, skip=skip)
    assert got[0][2] == "skip"


def test_the_route_reader_standing_kill_is_the_pythons_rule():
    fen = READ_FENS["middlegame"]
    b = chess.Board(fen)
    P0 = AC.Pos(fen)
    moves = [m.uci() for m in b.legal_moves]
    n_hit = 0
    for a in moves[:12]:
        Pa = P0.push(a)
        for rep in [r for r in Pa.replies()][:6]:
            uci, dbal0 = rep[0], rep[1] + Pa.balance(True) - P0.balance(True)
            for reads in (frozenset(), frozenset(range(64)), None):
                kills = {uci: [(Pa.push(uci), dbal0, reads, "San")]}
                for c_ in moves[12:20]:
                    rr = AC.read_route(fen, c_, True, 1, False, False, 1, None, 100, 8, [], [], kills)
                    P1 = P0.push(c_)
                    shift = P1.balance(True) - P0.balance(True)
                    want = None
                    for u, d, *_ in P1.replies():
                        if u != uci:
                            continue
                        d += shift
                        if reads is None or d >= 1 or dbal0 < d:
                            continue
                        if not (reads & frozenset(kills[uci][0][0].diff(P1.push(u)))):
                            want = (u, d, 0)
                    got = rr["standing"]
                    assert (tuple(got) if got else None) == want
                    n_hit += got is not None
                    assert not got or rr["rows"] == []
    print(f"OBS route reader standing kills matched {n_hit} times")
    assert n_hit > 0


@pytest.fixture
def route_rust():
    SC.CORE_RUST = SC.ROUTE_RUST = True
    try:
        yield
    finally:
        SC.CORE_RUST = SC.ROUTE_RUST = False


@pytest.mark.parametrize("pid", ["0LRTF", "07Z0P"])
def test_the_chase_with_the_rust_route_reader_decides_and_counts_the_same(pid):
    SC.CORE_RUST = True
    try:
        old = decide(pid, True)
        SC.ROUTE_RUST = True
        new = decide(pid, True)
    finally:
        SC.CORE_RUST = SC.ROUTE_RUST = False
    print(f"OBS route reader {pid}: decision {new['decision']}, inner agendas {new['chase'][0]}, replies closed by rule {new['chase'][2]}")
    assert old == new
