"""PLAN 78.64 stage A: the list of events is read from the rules (a second, made-up description gives its own list), no event word is a literal of the code, a taking, a change of kind and a
vanishing are told apart by the tracker, the kept reading cuts a taking that no settled position follows."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from catspace.incubator.analyst.discovery import events_rules as ER
from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.io import paths

CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
CODE = ["events_rules.py", "events_hazard.py", "events_fill.py"]
DISC = Path(__file__).resolve().parents[1] / "catspace" / "incubator" / "analyst" / "discovery"

TOY = """
(role red) (role blue)
(init (lamp red left)) (init (lamp blue left)) (init (count red 0)) (init (turn red))
(<= (next (lamp ?p ?s)) (true (lamp ?p ?s)) (not (broken ?p ?s)))
(<= (next (count ?p ?n)) (true (count ?p ?m)) (succ ?m ?n))
(<= (next (count ?p ?n)) (true (count ?p ?n)) (role ?p))
(<= (next (turn ?o)) (true (turn ?p)) (other ?p ?o))
(<= (goal ?r 10) (role ?r) (won ?r))
(<= (goal ?r 0) (role ?r) (lost ?r))
(<= (goal ?r 5) (role ?r) (tied))
"""


def test_the_fluents_that_can_only_go_false_are_read_from_the_clauses_of_any_description():
    cl = GDL.clauses(GDL.parse(TOY))
    assert ER.persistent_fluents(cl) == ["lamp"]                         # kept with the same arguments in every next clause; `count` and `turn` are made
    assert ER.init_instances(cl, "lamp") == [("blue", "left"), ("red", "left")]
    assert ER.role_words(cl) == ["red", "blue"]
    assert ER.goal_values(cl) == [0, 5, 10]
    assert [w for w, _r in ER.end_outcomes(["red", "blue"], [0, 5, 10])] == ["goal 10 for red", "goal 10 for blue", "no role at the highest goal value"]


def test_the_real_description_gives_its_own_list_from_its_clauses():
    cl = ER.rule_clauses((paths.REPO_ROOT / "rules" / "chess.gdl").read_text())
    fl = ER.persistent_fluents(cl)
    assert len(fl) == 1 and len(ER.init_instances(cl, fl[0])) == 4
    assert ER.goal_values(cl) == [0, 50, 100]


def test_no_event_word_is_a_literal_of_the_code():
    for name in CODE:
        p = DISC / name
        if not p.exists():
            continue
        src = p.read_text()
        tree = ast.parse(src)
        bad = []
        docs = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr)
                and isinstance(getattr(n.body[0], "value", None), ast.Constant) and isinstance(n.body[0].value.value, str)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and re.search(CHESS_WORDS, n.value, flags=re.I):
                bad.append(n.value)
            if isinstance(n, ast.Name) and re.search(CHESS_WORDS, n.id, flags=re.I):
                bad.append(n.id)
        assert not bad, (name, bad)
        code_only = "\n".join(ln.split("#")[0] for ln in src.splitlines())
        for kw in ("fen", "san", "uci", "check", "mate"):
            assert not re.search(rf"\b{kw}\b", re.sub(r'"""[\s\S]*?"""', "", code_only)), (name, kw)


def test_the_tracker_tells_a_taking_a_change_of_kind_and_a_vanishing_apart_with_made_up_words():
    a = [("red", "disc", "p1"), ("red", "ring", "p2"), ("blue", "disc", "p3"), ("blue", "ring", "p4")]
    taken = [("red", "disc", "p3"), ("red", "ring", "p2"), ("blue", "ring", "p4")]            # the red disc moved onto p3 and took the blue disc
    assert ER.census_events(a, taken, "red") == [("taken", "red", "blue", "disc", "disc")]
    changed = [("red", "disc", "p1"), ("red", "star", "p9"), ("blue", "disc", "p3"), ("blue", "ring", "p4")]
    assert ER.census_events(a, changed, "red") == [("changed kind", "red", "red", "ring", "star")]
    gone = [("red", "disc", "p1"), ("red", "ring", "p2"), ("blue", "ring", "p4")]
    assert ER.census_events(a, gone, "red") == [("vanished", "red", "blue", "disc", "")]
    same = a
    assert ER.census_events(a, same, "red") == []                                            # no change in the census, no event
    assert ER.classes_of(("taken", "red", "blue", "disc", "ring")) == [("taken", "red", "disc", "ring"), ("taken", "red", "disc"), ("taken", "red")]
