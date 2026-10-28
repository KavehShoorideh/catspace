"""PLAN 74 THE REASONING GRAPH (docs/ANALYST_PLAN.md 74, Kaveh 2026-10-27: "a graph of reasoning ... if I'm analyzing one line and a question pops in my head, I queue it for
later"; "As long as it's generic, I'm happy"). (a) the graph store (QUESTION / LINE / ANSWER / OBSERVATION; AROSE_FROM / SERVES / READS / REFINES), the provenance of every
question and the trace lines of 74.5. The new modules name no game object (the generic test). Count budgets only."""
import re
from pathlib import Path

from catspace.incubator.analyst.discovery import reasoning as RN
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import State

W0VFO = "2r2rk1/R5pp/5p2/2q1p3/3pQ3/8/2P2PPP/R5K1 w - - 4 30"     # rand8_10: Qb7 threatens mate on g7 (the spec's example line: "arose from Qb7 Rf7 at depth 1")
GENERIC = ("reasoning.py",)                                     # the new modules of plan 74 (each stage adds its own)
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
