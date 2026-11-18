"""Plan 78.86: the ladder read again with the repaired verdict (SYSTEM_SPEC 9 the ladder, 5 the line verdict). ladder_reread.py is a re-measurement; these tests read its comparison table, the rule for the
printed cases and the verdict counts on a toy input, and scan its sources for names of the game."""
from __future__ import annotations

import re
from pathlib import Path

from catspace.incubator.analyst.discovery import ladder_reread as LR

ROOT = Path(__file__).resolve().parents[1]
NAMES = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|[a-h][1-8])\b"


def _run(rungs: dict, verdicts: "dict | None" = None) -> dict:
    out = {}
    for i, r in rungs.items():
        deep = {"verdict": verdicts[i]} if verdicts and i in verdicts else None
        out[i] = {"id": i, "result": {"rung": r, "deep": deep}}
    return out


OLD = _run({"a": 1, "b": 0, "c": 3, "d": 0, "e": 2, "f": 0, "g": 0, "h": 3})
NEW = _run({"a": 1, "b": 3, "c": 0, "d": 3, "e": 1, "f": 3, "g": 0, "h": 0})


def test_the_table_of_changes_counts_each_direction():
    t = LR.rung_table(OLD, NEW)
    assert t == {(0, 3): 3, (3, 0): 2, (2, 1): 1}
    assert sum(t.values()) == 6                                              # a and g did not change


def test_division_counts_the_rungs_in_the_order_one_two_three_open():
    assert LR.division(OLD) == (1, 1, 2, 4)
    assert LR.division(NEW) == (2, 0, 3, 3)


def test_the_rule_for_the_cases_takes_directions_by_size_round_and_round():
    order = ["a", "b", "c", "d", "e", "f", "g", "h"]
    assert LR.pick_changed(order, OLD, NEW, 20) == ["b", "c", "e", "d", "h", "f"]      # (0->3) holds 3, (3->0) 2, (2->1) 1: one of each, then the next, then the last
    assert LR.pick_changed(order, OLD, NEW, 4) == ["b", "c", "e", "d"]
    assert LR.pick_changed(order, OLD, OLD, 20) == []


def test_the_verdict_counts_unsettled_and_the_kind_change_in_the_net():
    v = lambda w, k, u=False, wk=None: {"verdict": w, "kind_net": k, "unsettled": u, "without_kind": wk or w}
    run = _run({"a": 3, "b": 0, "c": 0, "d": 3}, {"a": v("CONFIRMED", 800, False, "EXCHANGE"), "b": v("UNSETTLED", 0, True), "c": v("REFUTED", -590, False, "REFUTED"), "d": None})
    c = LR.verdict_counts(run)
    assert c["lines"] == 3 and c["unsettled"] == 1 and c["kind_net"] == 2 and c["kind_net_changes_word"] == 1
    assert c["words"] == {"CONFIRMED": 1, "UNSETTLED": 1, "REFUTED": 1}


def test_the_run_is_the_last_one_of_78_77_and_goes_to_its_own_folder():
    assert LR.RUN == "selfplay_78_58@on" and LR.EARLIER == ("ladder_78_77", "selfplay_78_58_on")
    from catspace.incubator.analyst.discovery import ladder_measure as LM
    assert Path(LM.out_root("full", LR.RUN)).parts[-2:] == ("ladder_78_86", "selfplay_78_58_on")


def test_the_reread_sources_name_no_game_object():
    for f in ("catspace/incubator/analyst/discovery/ladder_reread.py", "catspace/incubator/analyst/discovery/ladder_measure.py"):
        text = (ROOT / f).read_text()
        assert not re.search(NAMES, text, re.I), (f, re.search(NAMES, text, re.I).group(0))
