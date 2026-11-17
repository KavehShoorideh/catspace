"""Plan 78.74: the proposer looks at every unit (catspace/incubator/analyst/discovery/subgoal_units.py, subgoal_goals.py, the checks in subgoal_measure.py). SYSTEM_SPEC section 7 "Choosing a subgoal, and getting there".
A goal for every unit of the other side (the line read for the units the threat list had no row for), the goal unit as a target, the fix as a set with its order, the two steps where nothing bears, the facts read from the line and not by counting
the units that bear, and the exact control of the grade."""
from __future__ import annotations

import re
from pathlib import Path

import cs_engine as CE
import pytest

from catspace.incubator.analyst.discovery import subgoal_goals as SG
from catspace.incubator.analyst.discovery import subgoal_measure as SM
from catspace.incubator.analyst.discovery import subgoal_print as SP
from catspace.incubator.analyst.discovery import subgoal_units as SU
from catspace.incubator.analyst.discovery import threat_concrete as TC

ROOT = Path(__file__).resolve().parents[1]
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
WORTH = {"q": 4343, "b": 1878, "n": 1661, "r": 1605, "p": 1045}
HALF = {"q": 120.0, "b": 606.0, "n": 628.0, "r": 534.0, "p": 508.0}
REC = TC.Record(WORTH, HALF, "synthetic")
PLACES = sorted(f + r for f in "abcdefgh" for r in "12345678")
NAMES = {"places": PLACES, "roles": ["white", "black"], "kinds": ["bishop", "king", "knight", "pawn", "queen", "rook"]}
GUARDED = "4k3/8/3p4/2p5/8/8/8/2Q1K3 w - - 0 1"                  # a unit takes a unit that another unit guards
KING_GUARD = "4k3/3p4/8/8/8/8/8/3QK3 w - - 0 1"                  # the only unit that takes back is the goal unit itself
ONE_ROOK = "4k3/8/8/8/8/8/8/3RK3 w - - 0 1"                      # the goal unit of the other side, and one unit of ours that covers two of its places
NOTHING_BEARS = "4k3/8/8/2p5/8/8/8/4K2R w - - 0 1"


def empty_c() -> dict:
    return {"mover": [], "other": [], "exchange": [], "goal": [], "dropped": {}, "dropped_rows": [], "resp_read": 0}


def item_of(text: str) -> dict:
    return {"start": text, "moves": [], "raw": {"mover": 0, "board": []}, "names": NAMES, "list": {"mover": [], "other": []}}


@pytest.fixture(scope="module")
def world():
    facts = SG.Facts("chess")
    facts.goal_name = "king"
    return facts, TC.Lines(REC, threads=1), SP.GameEditor("chess")


# ------------------------------------------------------------------------------------------------------------------------------ A: a line is read for every unit the list had no row for

def test_a_row_is_read_for_every_unit_of_the_other_side_but_the_goal_unit(world):
    facts, lines, ed = world
    c = empty_c()
    info = SU.every_unit([item_of(GUARDED)], [[0]], [c], facts, lines, [None])
    rows = c["mover"] + c["other"] + c["exchange"] + c["dropped_rows"]
    assert sorted(r["place"] for r in rows) == ["c5", "d6"] and info["units"] == 2
    assert all(r["every"] for r in rows) and not any(r["kind"] == "king" for r in rows)
    by = {r["place"]: r for r in rows}
    assert by["c5"]["verdict"] == "REFUTED" and by["c5"] in c["dropped_rows"] and by["c5"]["why"] == "refuted"      # the taking is read to its end by the line reader: the guard takes back
    assert by["c5"]["line"][0] == "c1c5"


def test_units_that_already_have_a_row_are_not_read_again(world):
    facts, lines, ed = world
    c = empty_c()
    c["dropped_rows"].append({"owner": 1, "place": "c5", "verdict": "DROPPED", "why": "refuted"})
    info = SU.every_unit([item_of(GUARDED)], [[0]], [c], facts, lines, [None])
    assert info["units"] == 1 and [r["place"] for r in c["mover"] + c["other"] + c["exchange"] + c["dropped_rows"] if r.get("every")] == ["d6"]


def test_a_unit_nothing_bears_on_gets_the_line_of_a_quiet_move_and_its_row_says_so(world):
    facts, lines, ed = world
    c = empty_c()
    SU.every_unit([item_of(NOTHING_BEARS)], [[0]], [c], facts, lines, [None])
    rows = c["mover"] + c["other"] + c["exchange"] + c["dropped_rows"]
    r = [x for x in rows if x["place"] == "c5"][0]
    assert r["two_moves"] is True and r["line"] and r["verdict"] in ("CONFIRMED", "DROPPED")


def test_goals_are_built_from_the_added_rows_and_name_every_stopper(world):
    facts, lines, ed = world
    c = empty_c()
    it = item_of(GUARDED)
    SU.every_unit([it], [[0]], [c], facts, lines, [None])
    gl = SG.goals(it, 0, c, facts, ed)
    g = [x for x in gl if x["target"][2] == "c5"][0]
    by = {f["name"]: f for f in g["facts"]}
    assert by["unguarded"]["status"] == "unmet" and by["unguarded"]["cause"]["basis"] == "line"        # a line said it, not a count of the units that bear
    assert g["stoppers"] == [("black", "pawn", "d6")] and g["set_edits"] == [("remove", "d6")]
    assert g["every"] is True and g["n_unmet"] == len(g["unmet"])


# ------------------------------------------------------------------------------------------------------------------------------ the goal unit, and the units that cannot be taken out

def test_the_goal_unit_is_a_target_with_the_places_it_may_go_to_and_who_covers_them(world):
    facts, lines, ed = world
    it = item_of(ONE_ROOK)
    gl = SG.goals(it, 0, empty_c(), facts, ed)
    g = [x for x in gl if x["source"] == "goal_unit"][0]
    assert g["target"] == ("black", "king", "e8") and g["stake"] is None
    esc = {e["place"]: e["covered"] for e in g["escapes"]}
    assert esc == {"d7": True, "d8": True, "e7": False, "f7": False, "f8": False}                     # the unit on the d file covers two of the five places
    cov = [f for f in g["facts"] if f["name"] == "covered" and f["status"] == "unmet"]
    assert len(cov) == 3 and any(f["fix"] is not None for f in cov) and all(("fix" in f) for f in cov)
    assert g["facts"][-1]["name"] == "forced_end" and g["facts"][-1]["status"] == "unmet"
    assert any(f["name"] == "borne_on" and f["status"] == "unmet" for f in g["facts"])


def test_a_stopper_of_the_goal_kind_is_named_but_left_out_of_the_edits(world):
    facts, lines, ed = world
    c = empty_c()
    it = item_of(KING_GUARD)
    SU.every_unit([it], [[0]], [c], facts, lines, [None])
    g = [x for x in SG.goals(it, 0, c, facts, ed) if x["target"][2] == "d7"][0]
    assert ("black", "king", "e8") in g["stoppers"]
    assert g["fix_set"][0]["removable"] is False and g["set_edits"] == []
    text, why = ed.apply(KING_GUARD, [("remove", "e8")])
    assert text is None                                                                         # the rules do not let it be taken out: that was the rejected edit of plan 78.71


# ------------------------------------------------------------------------------------------------------------------------------ B: the fix as an ordered set

def fact(name, stoppers, status="unmet", fix=None):
    return {"name": name, "vars": {}, "status": status, "cause": {"type": "guard", "stoppers": stoppers, "victim": None, "victims": [], "words": "", "basis": "line"}, "fix": fix}


def test_the_set_is_ordered_by_which_single_removal_leaves_the_fewest_facts_unmet():
    a, b, d = ("w", "k1", "x1"), ("w", "k2", "x2"), ("w", "k3", "x3")
    fx = [fact("f1", [a]), fact("f2", [a, b]), fact("f3", [a]), fact("f4", [d])]
    s = SG.fix_set_of(fx, "kg")
    assert [e["unit"] for e in s] == [a, d, b]                                                  # the unit that stops three facts alone or in part leaves the fewest unmet
    assert s[0]["left"] == 2 and s[0]["resolves"] == ["f1", "f3"] and s[1]["left"] == 3
    assert all(e["kind"] == "remove" and e["edit"] == ("remove", e["unit"][2]) for e in s)
    s2 = SG.fix_set_of([fact("f1", [("w", "kg", "x9")])], "kg")
    assert s2[0]["removable"] is False


def test_where_nothing_bears_the_goal_has_two_steps_a_unit_brought_by_the_moves_of_the_graph(world):
    facts, lines, ed = world
    c = empty_c()
    it = item_of(NOTHING_BEARS)
    row = {"place": "c5", "place_ix": PLACES.index("c5"), "two_moves": False, "line": [], "verdict": "DROPPED", "stake": None, "net": None, "owner": 1}
    g = SG.build_goal(it, 0, row, "no_bearer_no_move", facts, ed, NOTHING_BEARS, "white", "black", NAMES)
    assert g["steps"] and g["steps"][0]["step"] == "bring" and g["steps"][0]["moves"] in (1, 2, None)
    st = g["steps"][0]
    assert st["route"][0] == "h1" and st["route"][-1] in PLACES and len(st["route"]) == (2 if st["moves"] in (1, None) else 3)
    by = {f["name"]: f for f in g["facts"]}
    assert by["bears"]["cause"]["type"] == "no_bearer" and by["bears"]["cause"]["basis"] == "graph"
    assert g["fix_set"][0]["kind"] == "bring" and g["set_edits"][0][0] == "relocate"
    f = facts.of(NOTHING_BEARS)
    assert SG.route_of(f, facts, ed, "unit:white:rook:h1", "h5")[0] == 1                       # the rules let it move there now
    assert SG.route_of(f, facts, ed, "unit:white:king:e1", "e3")[0] in (1, 2, None)


# ------------------------------------------------------------------------------------------------------------------------------ C: the grade

def test_the_ranking_controls_are_exact_with_ties_in_random_order():
    assert SM.top_hit([0, 0, 0, 0], [True, False, False, False], 1) == pytest.approx(0.25)
    assert SM.top_hit([0, 0, 0, 0], [True, False, False, False], 3) == pytest.approx(0.75)
    assert SM.top_hit([1, 0, 2], [True, False, False], 1) == 0.0 and SM.top_hit([1, 0, 2], [True, False, False], 2) == 1.0
    assert SM.top_hit([(1, -0.5), (1, -0.2), (0, 0)], [False, True, False], 3) == 1.0 and SM.top_hit([(1, -0.5), (1, -0.2), (0, 0)], [False, True, False], 2) == 0.0          # the proposer's key: fewest unmet first, then the larger chance
    assert SM.top_hit([0, 0], [False, False], 3) == 0.0


def test_the_four_claims_are_unchanged():
    cl = SG.subgoal_claims({"edits": [("remove", "d6")], "key": "unguarded:guard"})
    assert [x["name"] for x in cl] == ["worth", "right fix", "reachable", "best use of the moves"]


def test_the_variants_of_the_fix_the_b_stage_tries():
    g = {"edits": [("remove", "d6"), ("move", "x")], "set_edits": [("remove", "d6"), ("remove", "c2")],
         "fix_set": [{"removable": True, "edit": ("remove", "d6")}, {"removable": False, "edit": ("remove", "e8")}, {"removable": True, "edit": ("remove", "c2")}]}
    v = SM.variants_of(g)
    assert v == {"fix": [("remove", "d6")], "set": [("remove", "d6"), ("remove", "c2")], "first": [("remove", "d6")], "last": [("remove", "c2")]}


# ------------------------------------------------------------------------------------------------------------------------------ the scan: the files that must name no game object

def test_the_game_free_files_of_plan_78_74_name_no_game_object():
    base = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
    for f in ("subgoal_goals.py", "subgoal_measure.py", "subgoal_units.py"):
        text = (base / f).read_text()
        hits = [(i + 1, m.group(0)) for i, line in enumerate(text.splitlines()) for m in re.finditer(CHESS_WORDS, line, re.IGNORECASE)]
        assert not hits, (f, hits[:5])
