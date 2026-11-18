"""Plan 78.87: the verdict's two faults (SYSTEM_SPEC 5, the line verdict). (1) The settled tail is a reading by the judge: the end of the game is counted as the search counts it, the side to move stands when every taking
leaves it worse, and a tail that reaches the end of the game says so (`tail_end`; the Rust test of case 8 of 78.86 is rust/engine/tactics/tests/tail_end.rs). (2) A line with nothing taken, no kind change and net zero is QUIET,
not REFUTED. These tests read the verdict on toy reads, the callers that branch on the word, the binding's new key, the measure's counts and its rule for the printed positions, and scan the measure's source for names of the game."""
from __future__ import annotations

import re
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import symbols_play as SP
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import verdict_measure as VM

ROOT = Path(__file__).resolve().parents[1]
WORTH = {"q": 4343, "b": 1878, "n": 1661, "r": 1605, "p": 1045}
HALF = {"q": 120.0, "b": 606.0, "n": 628.0, "r": 534.0, "p": 508.0}
REC = TC.Record(WORTH, HALF, "synthetic")
NAMES = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
CASE_8 = "6k1/ppp3pp/4P3/2b5/8/5R2/P1q3PP/1q3R1K w - - 0 54"


def _line(events, value, changes=(), end=0, cut=False, depth=6, **kw):
    return dict({"events": events, "changes": list(changes), "value": value, "end": end, "cut": cut, "depth": depth, "names": ["m1", "m2", "m3"], "settled": 3, "nodes": 10, "first": (1, 2), "lead": None, "prefix": []}, **kw)


# ------------------------------------------------------------------------------------------------------------------------------ the word QUIET

def test_a_line_with_nothing_taken_no_kind_change_and_net_zero_is_quiet():
    v = TC.verdict_of(_line([], 0), REC)
    assert v["verdict"] == "QUIET" and v["net"] == 0 and "QUIET" in TC.VERDICTS and v["ended"] == TC.TL.ENDED[1]


def test_quiet_is_only_that_line_every_other_is_the_word_it_was():
    assert TC.verdict_of(_line([(True, "p")], -1045), REC)["verdict"] == "REFUTED"                      # a unit lost: refuted as before
    assert TC.verdict_of(_line([(False, "p")], 1045), REC)["verdict"] == "CONFIRMED"
    assert TC.verdict_of(_line([(False, "n"), (True, "n")], 0), REC)["verdict"] == "EXCHANGE"
    assert TC.verdict_of(_line([], 0, changes=[(True, "p", "p")]), REC)["verdict"] == "REFUTED"        # a kind change with net zero is not a line with nothing happening: the old word
    assert TC.verdict_of(_line([], 0, cut=True), REC)["verdict"] == "UNREAD"
    assert TC.verdict_of(_line([], 0, unsettled=True), REC)["verdict"] == "QUIET"


def test_the_callers_that_branch_on_refuted_treat_quiet_as_they_treated_refuted():
    # symbols_play.gives_up needs units lost and none taken: a quiet line never had them, so it never gave up a unit and still does not
    assert not SP.gives_up(TC.verdict_of(_line([], 0), REC)) and not SP.gives_up(TC.verdict_of(_line([], 0), REC), True)
    for f in ("threat_concrete.py", "subgoal_units.py"):
        assert 'in ("REFUTED", "QUIET")' in (ROOT / "catspace/incubator/analyst/discovery" / f).read_text(), f      # the rows are dropped as refuted, as before
    assert '"REFUTED", "EXCHANGE", "QUIET"' in (ROOT / "catspace/incubator/analyst/discovery/subgoal_measure.py").read_text()


# ------------------------------------------------------------------------------------------------------------------------------ the tail's end of the game

def test_a_tail_that_ends_the_game_says_so_in_the_verdict():
    lost = TC.verdict_of(_line([(False, "q"), (True, "r")], 4343 - 1605, tail_end=2), REC)
    assert lost["verdict"] == "REFUTED" and lost["end"] == 2 and lost["tail_end"] == 2 and "tail" in lost["reason"] and lost["ended"] == TC.TL.ENDED[0]
    won = TC.verdict_of(_line([(False, "q")], -5, tail_end=1), REC)
    assert won["verdict"] == "CONFIRMED" and won["end"] == 1 and "tail" in won["reason"]
    assert TC.verdict_of(_line([(True, "p")], -1045, tail_end=0), REC)["tail_end"] == 0
    assert not SP.gives_up(lost) and not SP.gives_up(lost, True)                              # the game's end is not a settled end: no unit "given up for nothing"


def test_the_search_own_end_still_decides_as_before():
    assert TC.verdict_of(_line([(False, "p"), (True, "p")], 1 << 40, end=1), REC)["verdict"] == "CONFIRMED"
    assert TC.verdict_of(_line([], -(1 << 40), end=2), REC)["verdict"] == "REFUTED"


def test_the_binding_carries_the_tail_end_and_a_read_of_a_quiet_line_agrees_with_its_value():
    rec = TC.Record.load(None)
    res = CE.read_lines("chess", [(CASE_8, [])], [(0, 2, 0, 1, 0, 0, 0)], rec.worth, 4, 20000, 7, 100000, 1)
    reads = res[0]["reads"]
    assert reads and all(r["tail_end"] in (0, 1, 2) for r in reads)
    for r in reads:
        v = TC.verdict_of(r, rec)
        if not r["cut"] and not r["unsettled"] and v["end"] == 0:
            assert not (v["verdict"] == "CONFIRMED" and v["net_search"] < -v["resolution"]), (r["names"], v["net"], v["net_search"])


# ------------------------------------------------------------------------------------------------------------------------------ the measure

def _v(word, net=0, ns=0, tail=0, names=("m",), uns=False):
    return {"verdict": word, "net": net, "net_search": ns, "tail_end": tail, "unsettled": uns, "full_names": list(names)}


def test_the_counts_of_the_from_to_table_the_signs_and_the_tails():
    b = [("1", _v("REFUTED")), ("2", _v("CONFIRMED", 800, -590)), ("3", _v("EXCHANGE", 0, 0)), ("4", _v("UNSETTLED", -5, -5, uns=True))]
    a = [("1", _v("QUIET")), ("2", _v("EXCHANGE", 0, -590, tail=0)), ("3", _v("EXCHANGE", 0, 0, tail=1)), ("4", _v("UNSETTLED", -5, -5, uns=True))]
    s = VM.stats(b, a)
    assert s["n"] == 4 and s["changed"] == 2 and s["table"][("REFUTED", "QUIET")] == 1 and s["table"][("CONFIRMED", "EXCHANGE")] == 1
    assert s["opp"] == (1, 0) and s["diff"] == (1, 1) and s["tail_end"] == (0, 1) and s["unsettled"] == (1, 1)


def test_the_rule_for_the_ten_takes_quiet_and_tail_first_then_file_order():
    rows = []
    for i in range(14):
        quiet = i in (4, 6, 8, 11)
        tail = i in (1, 2, 5, 9, 12)
        rows.append({"set": "A", "id": str(i), "before": _v("REFUTED", names=("a",)), "after": _v("QUIET" if quiet else "EXCHANGE", names=("a", "b") if tail else ("a",))})
    got = [r["id"] for r in VM.pick(rows)]
    assert got == [str(i) for i in range(10)]                    # quiet 4, 6, 8; tail 1, 2, 5; then the first of the rest, 0, 3, 7, 9
    assert sum(1 for r in VM.pick(rows) if r["after"]["verdict"] == "QUIET") >= 3
    assert sum(1 for r in VM.pick(rows) if r["after"]["verdict"] != "QUIET" and r["before"]["full_names"] != r["after"]["full_names"]) >= 3
    assert [r["id"] for r in VM.pick(rows)] == sorted((r["id"] for r in VM.pick(rows)), key=int)         # shown in file order
    assert VM.pick(rows[:2]) == rows[:2]                                                                 # fewer than ten: all


def test_the_output_constant_and_the_sources_name_no_game_object():
    assert VM.OUT == ("verdict_78_87",) and Path(VM.out_root()).parts[-1] == "verdict_78_87"
    text = (ROOT / "catspace/incubator/analyst/discovery/verdict_measure.py").read_text()
    assert not re.search(NAMES, text, re.I), re.search(NAMES, text, re.I).group(0)
