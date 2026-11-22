"""PLAN 79.29 MEND WHAT 79.28 LEFT: THE COST OF TOUCHING AND THE TEST'S MEANINGLESS GOALS (SYSTEM_SPEC 11 "A slower result is diagnosed and mended, not set aside", "Reached means the goal outlasts the reply"; ledger row 36; piece: tactical reasoner).
Rust holds the tests (rust/engine/tactics/tests/layer4_goals.rs: the answers of the one reading against the slow plain reference, the floors, the nodes). THIS file is what belongs to Python: the scans of the remade fixture (no goal names one unit twice, no negated literal about a unit without
the literal that keeps the unit in place, every kind present), the maker, the reader's print (what a line took and gave up, the judge's value, held at nought plies) and the layer's files naming no object of a game.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "rust" / "engine"
TACTICS = ENGINE / "tactics"
FIXTURES = TACTICS / "tests" / "fixtures"
DISCOVERY = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
LAYER_FILES = [TACTICS / "src" / "meansends.rs", TACTICS / "src" / "reach.rs", TACTICS / "tests" / "layer4_goals.rs"]
GAME_WORDS = r"\b(king\w*|queen\w*|rook\w*|bishop\w*|knight\w*|pawn\w*|castl\w*|promot\w*|en passant|check\w*|mate|stalemate|uci|fen|san|chess\w*|shakmaty)\b"
SQUARE = r"\b[a-h][1-8]\b"


def goals(name: str = "goals.fixture") -> list:
    out = []
    for l in (FIXTURES / name).read_text().splitlines():
        if l.startswith("pos\t"):
            f = l.split("\t")
            out += [(f[1], g["kind"], g["pattern"]) for g in json.loads(f[3])]
    return out


def test_no_goal_of_the_remade_fixture_names_one_unit_twice_or_denies_a_unit_it_does_not_keep_in_place():
    gs = goals()
    kinds: dict = {}
    for _, k, _ in gs:
        kinds[k] = kinds.get(k, 0) + 1
    assert len(kinds) == 5 and set(kinds.values()) == {80}, kinds
    negated = 0
    for pid, k, pat in gs:
        placed = [tuple(l["args"][:3]) for l in pat if l["relation"] == "unit" and not l["neg"]]
        assert len(placed) == len(set(placed)), (pid, "a unit named twice")
        # a unit literal that names a place names that place once among the units of the goal
        places = [a[2] for a in placed if not a[2].startswith("?")]
        assert len(places) == len(set(places)), (pid, "two units on one place")
        for l in pat:
            if l["neg"]:
                negated += 1
                assert l["relation"] == "bear"
                role, kind, place, target = l["args"]
                assert (role, kind, place) in placed, (pid, "a negated literal about a unit without the literal that keeps it in place")
                assert place != target, (pid, "a unit that must not bear on its own place")
                assert (role, kind, target) not in placed, (pid, "the denied unit is another unit than the target")
    assert negated == kinds["may_take"]
    old = (FIXTURES / "goals_79_27.fixture").read_text()
    assert old.count("\npos\t") == 80 and '"differ"' not in old, "the questions of plan 79.27 are kept as they were"
    print(f"OBS 79.29 scan: goals.fixture holds {len(gs)} goals of {len(kinds)} kinds; {negated} negated literals, each with the literal that keeps its unit in place and none on the target's own place")


def test_the_pairs_fixture_is_the_references_and_names_positions_of_the_goals_fixture():
    lines = (FIXTURES / "pairs.fixture").read_text().splitlines()
    assert lines[0].startswith("note\t") and "reference" in lines[0] and "79.29" in lines[0]
    ids = {l.split("\t")[1] for l in (FIXTURES / "goals.fixture").read_text().splitlines() if l.startswith("pos\t")}
    pairs = [l.split("\t") for l in lines if l.startswith("pair\t")]
    assert len(pairs) >= 10 and all(p[1] in ids and p[2].isdigit() and p[3].isdigit() and int(p[2]) < int(p[3]) for p in pairs)
    print(f"OBS 79.29 scan: pairs.fixture holds {len(pairs)} (position, goal, goal) triples the reference found one line to reach")


def test_the_layer_files_name_no_object_of_a_game_and_the_one_reading_is_in_the_body():
    for p in LAYER_FILES:
        txt = p.read_text()
        bad = [m.group(0) for m in re.finditer(GAME_WORDS, txt, flags=re.I)] + [m.group(0) for m in re.finditer(SQUARE, txt)]
        assert not bad, (p.name, bad)
    body = (TACTICS / "src" / "reach.rs").read_text()
    assert "open_t" in body and "Open { strict: open_h, loose: open_t }" in body and "out.strict" in body and "out.loose" in body, "one reading carries both answers (plan 79.32: the walk is layer 0's, asked with the two masks and answering in `Out`)"
    assert "first_nodes" not in body and "cut2" not in body, "the second reading is gone"
    t = (TACTICS / "tests" / "layer4_goals.rs").read_text()
    assert "the_cost_of_the_second_reading" in t and "LAYER4_DUMP" in t
    print("OBS 79.29 scan: the body asks both answers in one tree; the entry, the body and the test name no object of a game")


def test_the_reader_print_says_what_a_line_took_and_gave_up_and_when_a_goal_is_held_at_nought_plies():
    from catspace.incubator.analyst.discovery import layer4_lines_print as lp

    r = {"took": ["rook"], "gave": ["knight", "pawn"], "value": 2}
    s = lp.cost(r)
    assert "TOOK: rook" in s and "GAVE UP: knight, pawn" in s and "VALUE FOR THE SIDE: 2" in s
    assert "TOOK: nothing" in lp.cost({"value": 0})
    rows = [{"status": "reached", "kind": k, "position": f"p{i}", "goal": 0, "plies": 0} for k in ("bears", "on_line") for i in range(3)]
    rows += [{"status": "reached", "kind": "bears", "position": "q", "goal": 0, "plies": 2}]
    root = lp.root_held(rows, 3)
    assert [x["kind"] for x in root] == ["bears", "on_line", "bears"] and all(x["plies"] == 0 for x in root)
    assert lp.root_held(rows, 3) == root, "the rule is a function of the answers"
    assert "whole_79_29" in (DISCOVERY / "layer4_lines_print.py").read_text()
    print("OBS 79.29 scan: the print shows took, gave up and the judge's value, and picks the goals held at nought plies by a fixed rule")


def test_the_maker_is_seeded_and_the_test_is_registered_after_the_one_of_79_28():
    mk = (DISCOVERY / "layer4_goals_print.py").read_text()
    assert "random.Random(" in mk and "others = [" in mk
    gate = (ROOT / "scripts" / "gate.py").read_text()
    assert gate.index("test_plan79_28") < gate.index("test_plan79_29")
    print("OBS 79.29 scan: the maker is seeded and picks the denied unit from the units other than the target; the test is in the gate after 79.28")
