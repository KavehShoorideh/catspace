"""Plan 78.71: goals as facts, the fix as the subgoal, the grading of subgoals (catspace/incubator/analyst/discovery/subgoal_goals.py, the checks in subgoal_measure.py, the print and the game's edit path in
subgoal_print.py). SYSTEM_SPEC 7 "Choosing a subgoal, and getting there". The proposer on hand-made positions (a guard that takes back, a unit in between, an illegal edit rejected), the claims, the small functions of
the measure, the concrete list keeping its dropped claims, and a scan of the files that must name no game object."""
from __future__ import annotations

import re
from pathlib import Path

import cs_engine as CE
import pytest

from catspace.incubator.analyst.discovery import subgoal_goals as SG
from catspace.incubator.analyst.discovery import subgoal_measure as SM
from catspace.incubator.analyst.discovery import subgoal_print as SP
from catspace.incubator.analyst.discovery import threat_concrete as TC

ROOT = Path(__file__).resolve().parents[1]
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
WORTH = {"q": 4343, "b": 1878, "n": 1661, "r": 1605, "p": 1045}
HALF = {"q": 120.0, "b": 606.0, "n": 628.0, "r": 534.0, "p": 508.0}
REC = TC.Record(WORTH, HALF, "synthetic")
PLACES = sorted(f + r for f in "abcdefgh" for r in "12345678")
NAMES = {"places": PLACES, "roles": ["white", "black"], "kinds": ["bishop", "king", "knight", "pawn", "queen", "rook"]}
GUARDED = "4k3/8/3p4/2p5/8/8/8/2Q1K3 w - - 0 1"                  # a unit takes a unit that another unit guards
BLOCKED = "4k3/8/8/2p5/8/2N5/2R5/4K3 w - - 0 1"                  # a unit of ours stands between the bearer and the target


@pytest.fixture(scope="module")
def world():
    facts = SG.Facts("chess")
    facts.goal_name = "king"
    return facts, TC.Lines(REC, threads=1), SP.GameEditor("chess")


def read_taking(lines, text, place):
    r = lines.ask([(text, [])], [(0, 0, 0, 1, PLACES.index(place), 0, 0)])[0]
    return TC.best_taking(r["reads"], REC) if r["ok"] and r["reads"] else None


def goal_of(facts, editor, text, place, best, source):
    row = {"place": place, "place_ix": PLACES.index(place), "two_moves": False, "line": best["names"] if best else [], "verdict": best["verdict"] if best else "DROPPED", "stake": None, "net": best["net"] if best else None}
    return SG.build_goal({"start": text, "moves": [], "names": NAMES}, 0, row, source, facts, editor, text, "white", "black", NAMES)


# ------------------------------------------------------------------------------------------------------------------------------ the goals on hand-made positions

def test_a_guard_that_takes_back_is_the_unmet_fact_and_removing_it_makes_the_goal_work(world):
    facts, lines, ed = world
    best = read_taking(lines, GUARDED, "c5")
    assert best["verdict"] == "REFUTED"
    g = goal_of(facts, ed, GUARDED, "c5", best, "refuted")
    by = {f["name"]: f for f in g["facts"]}
    assert by["bears"]["status"] == "met" and by["unguarded"]["status"] == "unmet"
    assert by["unguarded"]["cause"]["type"] == "guard" and by["unguarded"]["cause"]["stoppers"] == [("black", "pawn", "d6")]
    assert g["edits"] == [("remove", "d6")] and g["unmet"][0] == "unguarded"
    text2, why = ed.apply(GUARDED, g["edits"])
    assert why is None
    after = read_taking(lines, text2, "c5")
    assert after["verdict"] == "CONFIRMED"                         # the fix alone makes the goal work
    null, _ = ed.build(GUARDED, [])
    assert read_taking(lines, null, "c5")["verdict"] == "REFUTED"  # and without the fix it does not


def test_a_unit_in_between_is_named_as_the_stopper_of_the_bearing(world):
    facts, lines, ed = world
    g = goal_of(facts, ed, BLOCKED, "c5", None, "no_bearer_no_move")
    f = g["facts"][0]
    assert f["name"] == "bears" and f["status"] == "unmet" and f["cause"]["type"] == "blocked"
    assert f["cause"]["stoppers"] == [("white", "knight", "c3")] and g["edits"] == [("remove", "c3")]
    text2, why = ed.apply(BLOCKED, g["edits"])
    assert why is None and read_taking(lines, text2, "c5")["verdict"] == "CONFIRMED"


def test_an_edit_that_leaves_no_legal_position_is_rejected_and_counted(world):
    facts, lines, ed = world
    before = dict(ed.count)
    text, why = ed.apply(GUARDED, [("remove", "e8")])              # the goal unit of the other side cannot be taken out
    assert text is None and "legal" in why
    text, why = ed.apply(GUARDED, [("remove", "a5")])              # nothing stands there
    assert text is None
    assert ed.count["rejected"] == before["rejected"] + 2
    assert ed.relocate(GUARDED, "c1", "c1") is None                # a probe is not counted
    assert ed.count["rejected"] == before["rejected"] + 2


def test_a_unit_that_would_have_to_come_to_bear_is_found_by_the_graph_not_a_table(world):
    facts, lines, ed = world
    f = facts.of("4k3/8/8/2p5/8/8/8/4K2R w - - 0 1")               # nothing of ours bears on the target
    assert f.bearers("c5", "white") == []
    fix = SG.bring_fix(f, facts, "white", "c5", ed)
    assert fix is not None and fix["edits"][0][0] == "relocate"
    a, b = fix["edits"][0][1], fix["edits"][0][2]
    text2 = ed.relocate(f.text, a, b)
    g = facts.of(text2)
    assert "c5" in g.bear[g.at[b]]                                  # asked of the engine on the edited text
    assert SG.bring_fix(f, facts, "white", "c5", None) is None      # with no edit path the fix is not invented


def test_the_line_story_names_the_causes_of_plan_78_60(world):
    facts, lines, ed = world
    # a unit of theirs takes the taker's unit elsewhere: the taker unit was the only guard of it
    f = facts.of("4k3/8/8/3b4/8/2p5/1N6/R3K3 w - - 0 1")
    plies = facts.replay(f.text, ["b2d3"])
    assert plies[0]["kind"] == "knight" and plies[0]["from"] == "b2" and plies[0]["to"] == "d3" and plies[0]["took"] is None
    plies = facts.replay("4k3/8/8/3b4/8/2p5/1N6/R3K3 w - - 0 1", ["b2d3", "d5a2", "a1a2"])
    assert plies[1]["took"] is None and plies[2]["took"][1] == "black" and plies[2]["took"][3] == "a2"
    st = SG.line_story(f, "white", "c3", facts.replay(f.text, ["a1a3", "d5a2"]))
    assert st["losses"] == [] or st["losses"][0]["type"] in ("arrives", "loose", "only_guard", "uncovered_line", "guard")


# ------------------------------------------------------------------------------------------------------------------------------ the claims, the sources, the small functions of the measure

def test_the_four_claims_each_with_a_referee_that_is_our_own_reasoning():
    g = {"edits": [("remove", "d6")], "key": "unguarded:guard"}
    cl = SG.subgoal_claims(g)
    assert [c["name"] for c in cl] == ["worth", "right fix", "reachable", "best use of the moves"]
    assert all(c["referee"] and "oracle" not in c["referee"].lower() or "no oracle" in c["referee"].lower() for c in cl)
    assert not any("stockfish" in c["referee"].lower() for c in cl)


def test_the_sources_of_goals_from_the_concrete_list():
    assert SG._source_of({"verdict": "EXCHANGE"}) == "exchange"
    assert SG._source_of({"verdict": "UNREAD"}) == "unread"
    assert SG._source_of({"verdict": "CONFIRMED", "two_moves": True}) == "two_moves"
    assert SG._source_of({"verdict": "CONFIRMED", "two_moves": False}) is None            # it works: not a goal
    assert SG._source_of({"verdict": "DROPPED", "why": "refuted"}) == "refuted"
    c = {"mover": [{"owner": 0, "verdict": "CONFIRMED"}], "other": [{"owner": 1, "verdict": "UNREAD"}, {"owner": 1, "verdict": "CONFIRMED"}], "exchange": [{"owner": 1, "verdict": "EXCHANGE"}],
         "dropped_rows": [{"owner": 1, "verdict": "DROPPED", "why": "refuted"}, {"owner": 0, "verdict": "DROPPED", "why": "no_taking_legal"}]}
    gs, ws = SG.rows_for(c, 0)                                      # the threats of side 0 are the rows against side 1
    assert sorted(s for _r, s in gs) == ["exchange", "refuted", "unread"] and len(ws) == 1


def test_the_hindsight_control_is_exact_and_the_table_is_smoothed_and_crossfitted():
    assert SM.hyper_hit(4, 1, 1) == pytest.approx(0.25)
    assert SM.hyper_hit(4, 1, 3) == pytest.approx(0.75)
    assert SM.hyper_hit(3, 0, 3) == 0.0 and SM.hyper_hit(3, 3, 1) == 1.0
    gs = [(0, {}, {"reach": {4: True}, "untested": [], "key": "k", "testable": ["unguarded"]}) for _ in range(30)] + [(1, {}, {"reach": {4: False}, "untested": [], "key": "k", "testable": ["unguarded"]}) for _ in range(30)]
    t0 = SM.table_of(gs, 0)                                         # fitted on the games of the OTHER fold only
    assert t0[0]["k"] == [30, 0]
    g = {"key": "k", "testable": ["unguarded"]}
    assert SM.chance_of(g, t0) == pytest.approx(1 / 32)
    assert SM.chance_of({"key": "zz", "testable": []}, t0) == pytest.approx(1 / 32)       # no tabled key: the pooled share


def test_the_facts_of_a_goal_at_a_position_of_the_game(world):
    facts, lines, ed = world
    f = facts.of(GUARDED)
    assert SM.facts_at(f, "white", "c5") == {"bears": True, "unguarded": False, "may_take": True}
    assert SM.facts_at(f, "white", None) == {}


def test_the_concrete_list_keeps_the_claims_it_drops_with_the_line_and_why():
    src = (ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "threat_concrete.py").read_text()
    assert '"dropped_rows"' in src and 'code = ("two_move_not_ahead"' in src and '"refuted")' in src


# ------------------------------------------------------------------------------------------------------------------------------ the scan: the files that must name no game object

def test_the_new_game_free_files_name_no_game_object():
    base = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
    for f in ("subgoal_goals.py", "subgoal_measure.py"):
        text = (base / f).read_text()
        hits = [(i + 1, m.group(0)) for i, line in enumerate(text.splitlines()) for m in re.finditer(CHESS_WORDS, line, re.IGNORECASE)]
        assert not hits, (f, hits[:5])
