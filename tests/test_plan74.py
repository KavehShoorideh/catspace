"""PLAN 74 THE REASONING GRAPH (docs/ANALYST_PLAN.md 74, Kaveh 2026-10-27: "a graph of reasoning ... if I'm analyzing one line and a question pops in my head, I queue it for
later"; "As long as it's generic, I'm happy"). (a) the graph store (QUESTION / LINE / ANSWER / OBSERVATION; AROSE_FROM / SERVES / READS / REFINES), the provenance of every
question and the trace lines of 74.5; (b) the worklist and 74.2b's two clauses (the cheapest question that could keep the status first, then the policy). The new modules name no
game object (the generic test). Count budgets only (a clock is faked where the judge's price is read)."""
import re
from pathlib import Path

from catspace.incubator.analyst.discovery import reasoning as RN
from catspace.incubator.analyst.discovery import worklist as WL
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import State

W0VFO = "2r2rk1/R5pp/5p2/2q1p3/3pQ3/8/2P2PPP/R5K1 w - - 4 30"     # rand8_10: Qb7 threatens mate on g7 (the spec's example line: "arose from Qb7 Rf7 at depth 1")
GENERIC = ("reasoning.py", "worklist.py")                       # the new modules of plan 74 (each stage adds its own)
CHESS_WORDS = r"\b(chess|analyst_core|python-chess|king|queen|rook|bishop|knight|pawn|piece|square|board|mate|check|capture|castl\w*|promot\w*|uci|san|fen|file|rank|" \
              r"KING|QUEEN|ROOK|BISHOP|KNIGHT|PAWN|[a-h][1-8])\b"


def _code(p: Path) -> str:
    """The module's code without its docstrings and comments (as tests/test_plan73_5b.py reads it)."""
    t = p.read_text()
    t = re.sub(r'"""(?:.|\n)*?"""', "", t)
    return "\n".join(ln.split("#")[0] for ln in t.splitlines())


def test_74_the_new_modules_name_no_game_object():
    root = Path(RN.__file__).parent
    for f in GENERIC:
        code = _code(root / f)
        hit = re.search(CHESS_WORDS, code, flags=re.I)
        assert hit is None, (f, hit.group(0) if hit else None)
        assert "import chess" not in code and "analyst_core" not in code, f


def test_74_1_the_graph_store_nodes_edges_provenance_and_serving_by_key():
    g = RN.Graph()
    root = g.line(None, pk="P0")
    q1 = g.question("goal one", ("g", 1), 5, 0, pk="P0", arose=root.id)
    g.asked(q1.id)
    q1.label = "Q1"
    ln = g.line(root.id, steps=("s1", "r1"), names=("S1", "R1"), pk="P1")
    q2 = g.question("goal two", ("g", 2), 3, 1, pk="P1", arose=ln.id, refines=q1.id)
    g.asked(q2.id)
    a = g.answer(7, "goal two: kept", "claim", 3, reads_key=("store", ("g", 2), frozenset({1, 2})), question=q2.id, frame=frozenset({1, 2}))
    assert g.targets(q2.id, RN.AROSE_FROM) == [ln.id] and g.targets(q2.id, RN.REFINES) == [q1.id]
    assert g.targets(a.id, RN.SERVES) == [q2.id] and g.targets(a.id, RN.READS) == [("store", ("g", 2), frozenset({1, 2}))]
    assert g.why(q1.id) == "asked: a root of the position at depth 0"
    assert g.why(q2.id) == "asked: arose from S1 R1 at depth 1 (refines Q1 goal one)"
    # serving by key: a second line reaching a position that agrees on the read set meets the same answer (convergence)
    assert g.lookup(("store", ("g", 2), frozenset({1, 2}))) is a and g.lookup(("store", ("g", 2), frozenset({1}))) is None
    ln2 = g.line(root.id, steps=("s2", "r2"), names=("S2", "R2"), pk="P2")
    q3 = g.question("goal two", ("g", 2), 3, 1, pk="P2", arose=ln2.id, refines=q1.id)
    g.serve(q3.id, 7)
    assert g.nodes[q3.id].state == "served" and q3.id in g.targets(a.id, RN.SERVES) and g.counts["converged"] == 1
    assert g.why(q3.id).startswith("served: the answer #7 of S1 R1, its reads agreeing here (arose from S2 R2 at depth 1")
    t = g.tally()
    assert (t["questions"], t["lines"], t["answers"], t["asked"], t["served"]) == (3, 3, 1, 2, 1)
    assert g.summary().startswith(RN.PREFIX) and all(not x.startswith(RN.PREFIX) for x in g.dump()[:-1])


def test_74_1_a_line_waits_on_its_queued_questions_then_closes():
    g = RN.Graph()
    root = g.line(None, pk="P0")
    ln = g.line(root.id, steps=("s", "r"), names=("S", "R"), pk="P1")
    qs = [g.question(f"q{i}", None, 1, 1, pk="P1", arose=ln.id, state="deferred") for i in range(3)]
    assert all(g.why(q.id).startswith("deferred: arose from S R at depth 1") for q in qs)
    g.activate(qs[0].id, "Q4")
    g.mark_line(ln.id, "pending", on=[qs[0].id])
    assert g.nodes[ln.id].state == "pending" and g.nodes[ln.id].data["pending_on"] == [qs[0].id]
    g.mark_line(ln.id, "closed")
    t = g.tally()
    assert g.nodes[ln.id].state == "closed" and t["lines_pending"] == 1 and t["activated"] == 1 and t["deferred"] == 2


def test_74_5_W0VFO_every_question_records_why_it_was_asked_and_the_decision_shows_its_question_edges():
    ag = Agenda(State(W0VFO, "w", []))
    ag.ask()
    d = ag.decision()
    assert d is not None and d.what == "Qb7" and d.proved
    g = ag.ctx.graph
    qs = [n for n in g.nodes.values() if n.kind == RN.QUESTION]
    assert qs and all(g.line_of(n.id) is not None for n in qs)                    # every question arose from a line
    roots = [n for n in qs if n.depth == 0]
    assert roots and all(not g.line_of(n.id).names for n in roots)                 # the root's arose from the position itself
    deep = [n for n in qs if g.line_name(g.line_of(n.id)) == "Qb7 Rf7"]
    assert deep and all(n.depth == 1 and g.targets(n.id, RN.REFINES) for n in deep)  # the spec's example: arose from Qb7 Rf7 at depth 1, refining the chased root question
    assert any(g.why(n.id).startswith("asked: arose from Qb7 Rf7 at depth 1 (refines Q1 mate on h7") for n in deep)
    tr = ag.trace().splitlines()
    assert "why: Q1 asked: a root of the position at depth 0" in tr
    i = tr.index("why: the question edges of the decision's justification graph (plan 74.5: which question derived each fact, and why it was asked):")
    assert tr[i + 1].startswith("why:   #") and " answers Q1 mate on h7" in tr[i + 1]
    assert any(x.startswith("why:   #") and "arose from Qb7 Rf7 at depth 1" in x for x in tr[i + 1:])
    assert any(x.startswith("why: the reasoning graph: ") for x in tr)


# ---------------------------------------------------------------------------------------------------------------- (b) the worklist and 74.2b's two clauses

def _wl(stakes_prices: list, avoids: tuple = ()):
    w = WL.Worklist()
    f = w.frame()
    its = []
    for k, (stake, price) in enumerate(stakes_prices, 1):
        it = WL.Item(f"q{k}", f, (-stake, 0, 0, k), stake=stake, price=price)
        if k in avoids:
            it.data["avoids"] = True
        its.append(w.push(it))
    return w, f, its


def test_74_2b_clause_one_before_clause_two_on_synthetic_items():
    w, f, (a, b, c, d) = _wl([(100, 5.0), (3, 0.5), (1, None), (5, 0.5)])
    st = dict(balance=-1, line=0)                                  # the need: 1
    cands, clause = w.candidates(f, st, None)                       # (1): no move proved to keep the status -- the cheapest price class of those that could keep it
    assert clause == 1 and set(cands) == {b, d}
    assert w.select(f, st, None)[0] is d                            # ... the policy's order inside the class: the order key (stake 5 before stake 3)
    assert w.select(f, st, None, policy=lambda cs: b if b in cs else None)[0] is b      # ... or the policy's own pick inside it
    cands2, clause2 = w.candidates(f, st, 1)                        # (2): a keeping move is proved (balance -1 + 1 >= line 0): every open item, the policy's order
    assert clause2 == 0 and set(cands2) == {a, b, c, d} and w.select(f, st, 1)[0] is a
    st3 = dict(balance=-2, line=1)                                  # the need 3: the stake-1 item cannot keep it
    assert set(w.candidates(f, st3, None)[0]) == {b, d} and not c.keeps and a.keeps


def test_74_2b_equal_price_class_falls_to_the_policy_order_and_no_status_is_the_old_order():
    w, f, (a, b, c) = _wl([(1, 0.5), (9, 0.5), (5, 0.5)])
    st = dict(balance=0, line=0)                                    # the need 0: every item could keep it, all in one price class: nothing is narrowed
    assert w.candidates(f, st, None) == (w.items(f), 0) and w.select(f, st, None)[0] is b
    w2, f2, (x, y, z) = _wl([(1, None), (9, None), (5, None)])
    assert w2.candidates(f2, None, None) == (w2.items(f2), 0)       # the count budget: no status read -- the order key, the old heap's top
    assert w2.select(f2)[0] is y and w2.head(f2) is y
    w3, f3, (p, q, r) = _wl([(1, 0.1), (9, None), (5, None)], avoids=(1,))
    assert w3.candidates(f3, dict(balance=0, line=5), None) == ([p], 1)      # a loss avoided could keep it whatever its stake; the priced one is the cheapest class


def test_74_2_the_open_set_is_the_old_heap_array():
    import heapq
    import random
    rng = random.Random(742)
    w = WL.Worklist()
    f = w.frame()
    heap, live = [], {}
    for k in range(1, 200):
        if heap and rng.random() < 0.4:
            if rng.random() < 0.5:
                e = heapq.heappop(heap)                              # the old heap's pop ...
                w.take(live.pop(e[3]), pop=True)
            else:
                e = rng.choice(heap)                                 # ... and the policy's remove + heapify
                heap.remove(e)
                heapq.heapify(heap)
                w.take(live.pop(e[3]))
        else:
            o = (-rng.choice((1, 3, 5, 9, 100)), 0, rng.choice((0, 1)), k)
            heapq.heappush(heap, o + ("x",))
            live[k] = w.push(WL.Item(k, f, o, stake=-o[0]))
        assert [e[3] for e in heap] == [i.payload for i in w.items(f)]


def test_74_2b_the_question_price_is_the_judges_price_of_its_routes_read_before():
    import time
    ag = Agenda(State(W0VFO, "w", []))
    q = next(q for q in ag.open_questions() if q.stake == 100)
    assert ag._wl_status() is None and ag._wl_price(q) is None      # the count budget: no status, no rate
    ag.ctx.final = ag.ctx.deadline = time.monotonic() + 100.0       # a clock and a measured rate (0.01 s per cost unit) ...
    ag.ctx.judge_t, ag.ctx.judge_n = 1.0, 100
    assert ag._wl_price(q) is None                                  # ... but its routes are not known before it is asked
    ag.ctx.q_routes[ag._carry_key(q)] = ["e4b7", "e4d5"]
    ag.ctx.route_costs[(ag.pk, "e4b7", q.stake)] = (12, [], False)
    ag.ctx.route_costs[(ag.pk, "e4d5", q.stake)] = (30, [], False)
    assert abs(ag._wl_price(q) - 0.12) < 1e-9                       # the cheapest of its priced routes, cost x the rate
    ag.status_info = dict(balance=-1, line=0, word="losing", threat=0, s0=-1)
    st, best = ag._wl_status()
    assert st is ag.status_info and best is None and abs(q._wi.price - 0.12) < 1e-9
