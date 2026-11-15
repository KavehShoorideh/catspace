"""Plan 78.45: the sheet on puzzles (catspace/incubator/analyst/discovery/puzzle_sheet_measure.py): SEES, PICKS and each class of miss on small hand-built sheet data, the target of a line,
the tables, the source scan (the script names no game object) and one end-to-end run on two fixed positions written here."""
import random
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import puzzle_sheet_measure as PM

ROOT = Path(__file__).resolve().parents[1]
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"

LEGAL = ["a1a2", "a1b1", "b2b5", "b2c2", "c3c4", "d1d8", "e1e2", "f1f8"]
ENDS = lambda m: (m[:2], m[2:4])
KINDS = ("live0", "lost", "confirmed")


def finding(level, mode, status, detail):
    return {"level": level, "mode": mode, "status": status, "detail": detail}


def entry(way, status, move=None, route=None, line=None, stops=None):
    return {"way": way, "status": status, "move": move, "route": route, "line": line or [], "stops": stops or [], "time": None, "answer": None, "obstacles": [], "without_stopper": None}


def question(target, kind, stake, findings, asker="us"):
    return {"asker": asker, "target": target, "key": ["theirs", kind, "1", "clear"], "stake": stake, "level": 3, "state": "open", "findings": findings, "goal": False}


def frame(entries):
    return finding(2, "frame", "live", {"results": entries})


def lv1(routes):
    return finding(1, "level1", "live", {"routes": routes, "ways": [r[0] for r in routes]})


W1 = (1, "x", "a1", "a2", "strike")
W2 = (1, "x", "b2", "b5", "strike")
W3 = (2, "x", "c3", "d8", "strike")


def route(unit_at, places):
    return ("x", unit_at, places, [], "strike", [], [], "f")


def test_a_confirmed_way_that_begins_with_the_answer_is_seen_and_picked():
    q = question(("them", "x", "a2"), "confirmed", 8.0, [lv1([(W1, [route("a1", ["a2"])])]), frame([entry(W1, "confirmed", move="a1a2")])])
    s = PM.seeing([q], "a1a2", LEGAL, ENDS, KINDS)
    assert s["sees"] and s["picks"] and s["via"] == ["confirmed"] and s["pick"] == "a1a2"


def test_the_first_move_of_a_way_is_its_move_else_its_line_else_its_routes_first_step():
    legal = LEGAL
    assert PM._first_move(entry(W1, "x", move="a1a2"), legal, ENDS) == "a1a2"
    assert PM._first_move(entry(W1, "x", line=["a1b1", "z9z9"]), legal, ENDS) == "a1b1"
    assert PM._first_move(entry(W1, "x", line=["my unit on c3"], route=["a2"]), legal, ENDS) == "a1a2"        # a line of words is not a move
    assert PM._first_move(entry(W1, "x"), legal, ENDS) is None


def test_a_refuted_or_waiting_way_is_no_sight_and_a_confirmed_way_of_another_first_move_is_not_the_answers():
    q = question(("them", "x", "a2"), "confirmed", 8.0, [frame([entry(W1, "refuted", move="a1a2"), entry(W2, "confirmed", move="b2b5")])])
    assert PM.seeing([q], "a1a2", LEGAL, ENDS, KINDS)["sees"] is False
    s = PM.seeing([q], "b2b5", LEGAL, ENDS, KINDS)
    assert s["sees"]


def test_a_unit_standing_lost_is_seen_by_any_taking_of_its_place_and_picked_by_the_first_move_of_its_line():
    q = question(("them", "x", "b5"), "lost", 6.0, [frame([entry(None, "confirmed", line=["b2b5"])])])
    ends = lambda m: (m[:2], m[2:4])
    legal = LEGAL + ["c3b5"]
    s = PM.seeing([q], "c3b5", legal, ends, KINDS)
    assert s["sees"] and s["via"] == ["lost"] and s["pick"] == "b2b5" and not s["picks"]           # seen (a taking of the place) but outranked by the line's own first move
    assert PM.seeing([q], "b2b5", legal, ends, KINDS)["picks"]
    assert PM.seeing([q], "c3c4", legal, ends, KINDS)["sees"] is False


def test_the_chooser_takes_the_top_claim_by_stake_and_ties_by_the_sheets_order():
    q1 = question(("them", "x", "a2"), "confirmed", 8.0, [frame([entry(W1, "confirmed", move="a1a2")])])
    q2 = question(("them", "x", "b5"), "confirmed", 14.0, [frame([entry(W2, "confirmed", move="b2b5")])])
    q3 = question(("them", "x", "c4"), "confirmed", 14.0, [frame([entry(W3, "confirmed", move="c3c4")])])
    s = PM.seeing([q1, q2, q3], "a1a2", LEGAL, ENDS, KINDS)
    assert s["sees"] and s["pick"] == "b2b5" and not s["picks"] and s["picks_any"]
    assert PM.seeing([q3, q2, q1], "a1a2", LEGAL, ENDS, KINDS)["pick"] == "c3c4"          # the sheet's order breaks the tie


def test_a_claim_against_the_mover_or_not_in_danger_is_no_claim():
    mine = question(("us", "x", "a1"), "confirmed", 8.0, [frame([entry(W1, "confirmed", move="a1a2")])], asker="them")
    waiting = question(("them", "x", "a2"), "waiting", 8.0, [frame([entry(W1, "waiting", move="a1a2")])])
    s = PM.seeing([mine, waiting], "a1a2", LEGAL, ENDS, KINDS)
    assert not s["sees"] and not s["picks_any"] and s["pick"] is None


def test_a_live0_claim_sees_through_the_first_steps_of_its_level1_routes():
    q = question(("them", "x", "a2"), "live0", 3.0, [lv1([(W1, [route("a1", ["a2"]), route("a1", ["b1"])])])])
    assert PM.seeing([q], "a1b1", LEGAL, ENDS, KINDS)["sees"]
    assert not PM.seeing([q], "d1d8", LEGAL, ENDS, KINDS)["sees"]


# ------------------------------------------------------------------------------------------------------------------------------- the classes

def cls(questions, answer, target, took=(), line=("a1a2", "z", "z")):
    return PM.classify(questions, answer, target, LEGAL, ENDS, list(took), list(line), KINDS)


def test_class_T_the_target_is_not_a_question():
    q = question(("them", "x", "a2"), "waiting", 8.0, [lv1([(W1, [route("a1", ["a2"])])])])
    assert cls([q], "a1a2", ("them", "x", "h8"))["cls"] == "T"
    assert cls([q], "a1a2", None)["cls"] == "other"


def test_class_R_no_way_begins_with_the_answer_and_its_sub_kinds():
    q = question(("them", "x", "a2"), "waiting", 8.0, [lv1([(W1, [route("a1", ["a2"])])]), frame([entry(W1, "waiting", route=["a2"])])])
    got = cls([q], "d1d8", ("them", "x", "a2"), took=[("them", "y", "d8")])
    assert got["cls"] == "R" and "another unit" in got["sub"]
    got = cls([q], "d1d8", ("them", "x", "a2"))
    assert got["cls"] == "R" and "quiet" in got["sub"]
    q2 = question(("them", "x", "a2"), "waiting", 8.0, [lv1([((2, "x", "a1", "c4", "strike"), [route("a1", ["b1", "d8"])])])])
    assert "lands on a place" in cls([q2], "d1d8", ("them", "x", "a2"))["sub"]
    q0 = question(("them", "x", "a2"), "closed1", 8.0, [lv1([])])
    assert "no way at all" in cls([q0], "d1d8", ("them", "x", "a2"))["sub"]
    q3 = question(("them", "x", "a2"), "waiting", 8.0, [finding(1, "level1", "live", {"routes": [], "ways": [(2, "x", "a1", "c4", "strike")]})])
    got = cls([q3], "d1d8", ("them", "x", "a2"))
    assert got["cls"] == "other" and "did not list" in got["sub"]                                  # a way of two moves with no routes read: it cannot be said


def test_class_X_a_refuted_way_with_the_stop_and_whether_the_line_plays_the_stops_answer():
    stop = (True, "b1a1", 1, None, "unit_taken", [], ("b1", "a1"))
    q = question(("them", "x", "a2"), "refuted", 8.0, [lv1([(W1, [route("a1", ["a2"])])]), frame([entry(W1, "refuted", move="a1a2", stops=[stop])])])
    got = cls([q], "a1a2", ("them", "x", "a2"), line=("a1a2", "b1a1", "c3c4"))
    assert got["cls"] == "X" and got["stops"][0]["reason"] == "unit_taken" and got["stops"][0]["answer_played_in_line"] is True
    got = cls([q], "a1a2", ("them", "x", "a2"), line=("a1a2", "e1e2", "c3c4"))
    assert got["stops"][0]["answer_played_in_line"] is False


def test_class_X_the_exchange_where_the_unit_stands_refuted_when_the_answer_takes_it():
    q = question(("them", "x", "b5"), "waiting", 6.0, [frame([entry(None, "refuted", line=["b2b5", "c3b5"])])])
    got = cls([q], "b2b5", ("them", "x", "b5"))
    assert got["cls"] == "X" and got["way"][1] == "stand"


def test_class_W_a_waiting_way():
    q = question(("them", "x", "a2"), "waiting", 8.0, [lv1([(W1, [route("a1", ["a2"])])]), frame([entry(W1, "waiting", route=["a2"], stops=[(False, None, 0, 1, "line_shut", ["c2"], None)])])])
    got = cls([q], "a1a2", ("them", "x", "a2"))
    assert got["cls"] == "W" and got["stops"][0]["reason"] == "line_shut"


def test_class_N_not_settled_and_not_reached():
    q = question(("them", "x", "a2"), "live1", 8.0, [lv1([(W1, [route("a1", ["a2"])])])])
    assert cls([q], "a1a2", ("them", "x", "a2"))["cls"] == "N"                                  # a route begins with it, no deep reading
    q2 = question(("them", "x", "a2"), "unsettled", 8.0, [lv1([(W1, [route("a1", ["a2"])])]), frame([entry(W1, "not settled", route=["a2"])])])
    assert cls([q2], "a1a2", ("them", "x", "a2"))["cls"] == "N"
    q3 = question(("them", "x", "a2"), "refuted", 8.0, [finding(2, "search", "idle", {"ways": [W1], "results": [(None, "open", 0.0, None)]})])
    assert cls([q3], "a1a2", ("them", "x", "a2"))["cls"] == "N" or cls([q3], "a1a2", ("them", "x", "a2"))["cls"] == "R"


def test_a_search_finding_gives_the_first_move_of_each_way_by_index():
    q = question(("them", "x", "a2"), "confirmed", 28.0, [finding(2, "search", "live", {"ways": [W1, W2], "results": [("a1b1", "open", 0.0, None), ("b2b5", "confirmed", 1e9, None)]})])
    assert PM.seeing([q], "b2b5", LEGAL, ENDS, KINDS)["sees"] and not PM.seeing([q], "a1b1", LEGAL, ENDS, KINDS)["sees"]
    q = question(("them", "x", "a2"), "refuted", 28.0, [finding(2, "search", "idle", {"ways": [W1], "results": [("a1a2", "refuted", -14.0, "b1a1")]})])
    got = cls([q], "a1a2", ("them", "x", "a2"))
    assert got["cls"] == "X" and got["stops"][0]["answer"] == "b1a1"


def test_the_target_of_a_line():
    units = [("r", "b5", 8.0), ("q", "c4", 14.0)]
    assert PM.line_target(units, ["a1b5"], ENDS) == ("them", "r", "b5")
    # the solver takes the queen after it moved: followed through the other side's move
    assert PM.line_target(units, ["a1a2", "c4c5", "a2c5"], ENDS) == ("them", "q", "c4")
    assert PM.line_target(units, ["a1a2", "z1z2"], ENDS) is None
    assert PM.line_target(units, ["a1b5"], ENDS, None, True, ("them", "g", "h8")) == ("them", "g", "h8")        # a won game outranks the material on the way


def test_the_tables_and_the_pick_of_misses():
    def row(i, cls_, sees, picks, a, b, theme, length):
        return {"id": str(i), "set": "s", "class": cls_, "themes": [theme], "length": length, "answer_verdict": "SAFE", "safe_moves": 3, "seconds_default": 0.1, "seconds_big": 0.5, "search_a": a, "search_b": b,
                "seen": {"default": {"sees": sees, "picks": picks, "picks_any": sees, "via": ["confirmed"] if sees else []}, "big": {"sees": sees, "picks": picks}}}
    rows = [row(1, "SEEN-PICKED", True, True, True, True, "t1", "short"), row(2, "O", True, False, False, True, "t1", "long"), row(3, "R", False, False, True, True, "t2", "short"),
            row(4, "X", False, False, False, False, "t2", "short"), row(5, "T", False, False, False, True, "t1", "long")]
    tab = PM.tabulate(rows)
    assert tab["n"] == 5 and tab["seen"] == 2 and tab["picked"] == 1
    assert dict(tab["classes"]) == {"O": 1, "R": 1, "X": 1, "T": 1}
    assert tab["four"]["a"][(True, False)] == 1 and tab["four"]["a"][(False, True)] == 1 and tab["sheet_yes_search_a_no"] == ["2"]
    assert tab["by_theme"]["t1"]["SEEN"] == 1 and tab["by_theme"]["t1"]["O"] == 1 and tab["by_length"]["long"]["T"] == 1
    got = PM.pick_misses(rows, 3, 7)
    assert len(got) == 3 and len({r["class"] for r in got}) == 3 and PM.pick_misses(rows, 3, 7) == got
    assert any("OBS puzzle sheet x: classes" in line and "T 1" in line for line in PM.obs_lines("x", tab))


def test_the_script_names_no_game_object_in_code_or_comments():
    text = (ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "puzzle_sheet_measure.py").read_text()
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    hit = re.search(CHESS_WORDS, text, flags=re.I)
    assert hit is None, hit.group(0) if hit else None


# two fixed positions: a one-move mate (the answer takes the board) and a rook that a queen takes where it stands
POSITIONS = [
    {"id": "a", "set": "t", "pos": "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", "answer": "d1d8", "themes": ["mate", "mateIn1", "oneMove"]},
    {"id": "b", "set": "t", "pos": "6k1/5ppp/8/3r4/8/8/5PPP/3Q2K1 w - - 0 1", "answer": "d1d5", "themes": ["hangingPiece", "short"]},
]


def test_an_end_to_end_measure_of_two_fixed_positions():
    PM._init("chess")
    rows = [PM.measure(p, {"line_ply": 3, "line_seconds": 0.05}) for p in POSITIONS]
    for r in rows:
        assert "error" not in r, r
        assert set(r["seen"]) == {"default", "big"} and r["class"] in ("SEEN-PICKED", "O", "T", "R", "X", "W", "N", "other")
        assert r["answer_verdict"] in ("SAFE", "LOSES", "NOT SETTLED") and r["safe_moves"] >= 1 and r["seconds_default"] > 0
        assert r["line"][0] == r["answer"] and r["search_b"] in (True, False)
    a, b = rows
    assert a["length"].startswith("1 ") and a["line_decided"] is True and a["target"][1] == "king"       # a game won by the line: the target is the goal unit
    assert b["target"] == ["them", "rook", "d5"]                                                           # the unit the answer takes
    assert b["seen"]["default"]["sees"] is True                                                            # a unit standing lost is seen by the taking of its place
    tab = PM.tabulate(rows)
    assert tab["n"] == 2 and tab["seen"] >= 1
