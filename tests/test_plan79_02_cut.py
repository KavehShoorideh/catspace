"""PLAN 79.02 WHAT THE EQUAL SET HOLDS WHEN THE READING IS CUT OR PRUNED, AND THE MEND (SYSTEM_SPEC 4 the equal reading and the verified choice, 12 the debt of 78.84). Tests of the pure parts of the count (how a held move is held, whether the played move was asked before the clock stopped, the moves dropped), of the measure of the mend (put out, the pick's label, the four conditions of the rule,
the refusal to read CONFIRM before the rule is written), of the wall (no oracle call in these modules, no play path loads them), of the switch in the binding (off by default: the row's first fifteen elements are what they were and the sixteenth is empty; on: it counts) and of the printer. The layer files name no game object."""
import json
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import cut_count as CC
from catspace.incubator.analyst.discovery import cut_mend as CM
from catspace.incubator.analyst.discovery import cut_play as CP

DIR = Path(CC.__file__).parent
LAYER = ("cut_count.py", "cut_mend.py", "cut_play.py", "cut_out.py")
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")


def test_the_layer_files_name_no_game_object():
    for f in LAYER:
        text = (DIR / f).read_text().lower()
        assert not re.search(r"\b[a-h][1-8]\b", text), f
        assert "sys.path.insert" not in text, f
        for w in BANNED:
            assert not re.search(rf"\b{w}\b", text), (f, w)


def test_this_test_file_names_no_game_object_either():
    assert not re.search(r"\b[a-h][1-8]\b", Path(__file__).read_text().lower())


def test_the_wall_no_module_of_this_plan_calls_the_oracle_and_no_play_path_loads_them():
    for f in LAYER:
        text = (DIR / f).read_text()
        assert not re.search(r"\bLO\.(analyse|analyse_after|label)\b", text), f
        assert "ladder_oracle" not in text, f
    for p in DIR.glob("*.py"):
        if p.name.startswith("cut_") or p.name in ("ladder_print.py",):
            continue
        assert not re.search(r"^\s*(from|import) .*\bcut_(count|mend|play|out)\b", p.read_text(), flags=re.M), p.name
    play = (DIR / "cut_play.py").read_text()
    assert not re.search(r"^\s*(from|import) .*\b(leaf_\w+|cut_count|cut_mend|ladder_oracle|prior_equal|stack_room)\b", play, flags=re.M)


def test_the_match_module_does_not_load_the_oracle_wrapper_even_through_what_it_imports():
    import subprocess
    import sys
    code = ("import sys; import catspace.incubator.analyst.discovery.cut_play; "
            "bad = [m for m in sys.modules if m.endswith('ladder_oracle') or m.endswith('leaf_ask') or m.endswith('cut_count') or m.endswith('cut_mend')]; "
            "print('LOADED', bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr


# ---- part one: how a held move is held

def row(warm, plain=None, no_late=None, no_null=None, exact=None):
    return {"warm": warm, "plain": plain, "no_late": no_late, "no_null": no_null, "exact": exact}


def test_a_move_the_search_holds_is_held_by_the_plain_answer_or_only_by_a_reduced_one():
    v = 10
    assert CC.hold_kind(row(10, plain=10), v) == "plain"
    assert CC.hold_kind(row(30, plain=12), v) == "plain"
    assert CC.hold_kind(row(10, plain=0, no_late=0, no_null=10), v) == "late"
    assert CC.hold_kind(row(10, plain=0, no_late=10, no_null=0), v) == "null"
    assert CC.hold_kind(row(10, plain=0, no_late=0, no_null=0), v) == "both"
    assert CC.hold_kind(row(10, plain=0, no_late=10, no_null=10), v) == "joint"
    assert CC.hold_kind(row(9, plain=0), v) == "none"                      # the search's own reading does not hold it: not in the set
    assert CC.hold_kind(row(10, plain=None), v) == "unread" and CC.hold_kind(None, v) == "unread"


def test_the_played_move_is_unshown_only_when_the_reading_was_cut_and_no_later_member_shows_it_was_passed():
    order = ["a", "b", "c", "d"]
    assert CC.first_state(order, ["a", "c"], "c", False) == "asked"        # not cut: every move was asked
    assert CC.first_state(order, ["a", "c"], "a", True) == "asked"        # cut, but the reading reached c after a
    assert CC.first_state(order, ["a", "c"], "c", True) == "unshown"      # cut, c is the last member: nothing shows it was reached
    assert CC.first_state(order, ["c"], "c", True) == "unshown"           # cut, the set is the played move alone
    assert CC.first_state(order, ["x"], "x", True) == "unknown"


def test_the_dropped_moves_are_those_the_untimed_reading_holds_that_the_set_does_not_hold():
    rows = [{"name": "a", "warm": 5}, {"name": "b", "warm": 5}, {"name": "c", "warm": 1}, {"name": "d", "warm": None}]
    assert CC.dropped(rows, 5, ["a"]) == ["b"]
    assert CC.dropped(rows, 5, ["a", "b"]) == []


def test_every_equal_move_gets_its_class_its_labels_and_how_it_was_held():
    rec = {"id": "1_5", "ord": 1, "band": "mid", "first": "m0", "tied": ["m0", "m1", "m2"], "cut": True, "cls": 0, "cm": [0, -1, 0]}
    aud = {"value": 7, "rows": [{"name": "m0", "warm": 7, "plain": 7, "no_late": 7, "no_null": 7, "exact": 7}, {"name": "m1", "warm": 7, "plain": 0, "no_late": 0, "no_null": 7, "exact": 0}, {"name": "m2", "warm": 9, "plain": 9, "no_late": 9, "no_null": 9, "exact": 9}, {"name": "m3", "warm": 8, "plain": 8, "no_late": 8, "no_null": 8, "exact": 8}]}
    ms = CC.equal_moves(rec, aud, ["keep", "lose", None])
    assert [m["kind"] for m in ms] == ["first-asked", "other-late", "other-plain"]
    assert [m["label"] for m in ms] == ["keep", "lose", None] and [m["l1"] for m in ms] == ["keep", "lose", "keep"]
    assert all(m["cut"] for m in ms)
    assert CC.equal_moves(dict(rec, cut=False), aud, None)[0]["label"] is None


# ---- part three: the measure of the mend

def test_a_move_is_put_out_when_it_is_in_the_unmended_set_and_not_in_the_other_and_unlabelled_moves_are_counted_apart():
    labels = {"a": "keep", "b": "lose", "c": "lose", "d": None}
    po = CM.put_out(["a", "b", "c", "d", "z"], ["a", "c"], labels)
    assert po["lose"] == (1, 2) and po["keep"] == (0, 1) and po["unlabelled"] == 2
    assert CM.put_out([], ["a"], labels) == {"lose": (0, 0), "keep": (0, 0), "unlabelled": 0}


def test_the_state_of_a_pick_is_its_firm_label_or_unlabelled():
    labels = {"a": "keep", "b": "lose", "c": None}
    assert [CM.pick_state(x, labels) for x in ("a", "b", "c", "z", None)] == ["keep", "lose", "neither", "unlabelled", "unlabelled"]


def _pos(game, a_set, m_set, n_set, pick_a, pick_m, labels, sec=(0.2, 0.2, 0.2)):
    return {"id": f"{game}_1", "game": game, "band": "mid", "part": "find", "labels": labels, "sets": {"a": a_set, "m": m_set, "n": n_set},
            "reads": {"a": {"seconds": sec[0]}, "m": {"seconds": sec[1]}, "n": {"seconds": sec[2]}}, "pick": {"a": CM.pick_state(pick_a, labels), "m": CM.pick_state(pick_m, labels), "n": CM.pick_state(pick_a, labels)},
            "first": {"a": "keep", "m": "keep", "n": "keep"}}


def test_the_four_conditions_of_the_rule_pass_for_a_mend_that_puts_out_losing_moves_at_no_price_and_fail_for_one_that_costs_keeping_moves():
    labels = {"k": "keep", "l1": "lose", "l2": "lose"}
    good = [_pos(g, ["k", "l1", "l2"], ["k"], ["k", "l1", "l2"], "l1", "k", labels) for g in range(1, 41)]
    c = CM.conditions(good)
    assert c["pick_pass"] and c["over_pass"] and c["price_pass"] and c["time_pass"]
    assert c["pick_gain"][0] == pytest.approx(1.0) and c["price"] == 0.0
    bad = [_pos(g, ["k", "l1", "l2"], ["l1"], ["k", "l1", "l2"], "k", "l1", labels) for g in range(1, 41)]
    c = CM.conditions(bad)
    assert not c["pick_pass"] and not c["price_pass"] and c["price"] == 1.0
    slow = [_pos(g, ["k", "l1"], ["k"], ["k", "l1"], "l1", "k", labels, sec=(0.2, 0.35, 0.2)) for g in range(1, 41)]
    assert not CM.conditions(slow)["time_pass"]
    noisy = [_pos(g, ["k", "l1"], ["k"], ["k"], "l1", "k", labels) for g in range(1, 41)]
    assert not CM.conditions(noisy)["over_pass"]                              # the clock alone put out as many


def test_confirm_is_not_read_before_the_rule_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(CC, "out_root", lambda tag: tmp_path)
    monkeypatch.setattr(CM.CC, "out_root", lambda tag: tmp_path)
    with pytest.raises(RuntimeError):
        CM.cmd_count("x", "confirm", lambda s: None)
    said = []
    CM.cmd_rule("x", said.append)
    rule = json.loads((tmp_path / "rule.json").read_text())
    assert rule["price_max"] == CM.PRICE_MAX and "before CONFIRM" in said[0]


def test_a_firmly_losing_move_that_left_the_mends_set_left_by_the_unpruned_answer_or_by_the_cut_and_the_order():
    labels = {"k": "keep", "l1": "lose", "l2": "lose", "l3": "lose"}
    aud = {"1_1": {"audit": {"value": 5, "rows": [{"name": "l1", "warm": 5, "plain": 0, "no_late": 0, "no_null": 5, "exact": 5}, {"name": "l2", "warm": 5, "plain": 5, "no_late": 5, "no_null": 5, "exact": 5},
                                                  {"name": "l3", "warm": 5, "plain": 0, "no_late": 5, "no_null": 0, "exact": 5}]}}}
    r = _pos(1, ["k", "l1", "l2", "l3"], ["k", "l3"], ["k", "l1", "l2", "l3"], "k", "k", labels)
    r["id"] = "1_1"
    assert CM.put_out_reasons([r], aud) == {"reasked": 1, "dropped": 1, "no_audit": 0}          # l1 left with an unpruned answer under v, l2 is held by the plain answer too
    assert CM.put_out_reasons([r], aud, "n") == {"reasked": 0, "dropped": 0, "no_audit": 0}      # the noise arm kept all of them
    assert CM.put_out_reasons([r], {}) == {"reasked": 0, "dropped": 0, "no_audit": 2}


def test_the_clock_of_the_mend_measure_is_the_play_clock_unless_a_supplement_says_otherwise():
    from catspace.incubator.analyst.discovery import tiebreak_measure as TM
    assert CM.CLOCK == TM.SECONDS


def test_the_three_arms_are_read_in_the_order_switch_off_on_off():
    assert CM.ARMS == ("a", "m", "n")


# ---- the match

def test_the_two_sides_of_the_match_differ_in_the_switch_alone():
    a, b = CP.player_spec("mend"), CP.player_spec("stack")
    assert a.pop("reask") is True and b.pop("reask") is False
    assert a == b and a["verify_choice"] is True and a["choice_mode"] == "slot"
    specs = CP.match_specs("mend", "stack", "A", 2, 0.2, Path("/nonexistent"))
    assert len(specs) == 4 and {s["players"][0]["reask"] for s in specs} == {True, False}
    assert all(s["max_plies"] == 120 and s["seconds"] == 0.2 for s in specs)


def test_the_movers_pass_the_switch_only_when_the_spec_asks():
    from catspace.incubator.analyst.discovery import tiebreak_measure as TM
    m = TM.Mover67.__new__(TM.Mover67)
    m.verify_choice, m.choice_mode, m.reask = True, "slot", False
    assert m.choice_kw() == {"verify_choice": True, "choice_mode": "slot"}
    m.reask = True
    assert m.choice_kw() == {"verify_choice": True, "choice_mode": "slot", "reask": True}
    m.verify_choice, m.reask = False, False
    assert m.choice_kw() == {}


# ---- the switch in the binding

MOVES = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]


def _call(**kw):
    import cs_engine as CE
    from catspace.incubator.analyst.discovery import record_play as RP
    return CE.solve_equals_batch([("chess", None, list(MOVES))], 2.0, worth=dict(RP.worth_of("human")), ours_first=False, valuation="odds record", threads=1, ledger_share=0.4, verify_choice=True, choice_mode="slot", **kw)[0]


def test_the_binding_row_keeps_its_first_fifteen_elements_and_the_switch_off_adds_an_empty_sixteenth():
    r = _call()
    if len(r) < 16:
        pytest.skip("the extension in this environment was built before plan 79.02")
    assert len(r) == 16 and list(r[15]) == []
    assert r[4] is None and isinstance(r[5], list) and r[10] > 0 and r[12] in (0, 1, 2, 3, 4)


def test_the_switch_on_counts_what_the_second_asking_did_and_the_set_is_inside_the_first_reading():
    r0 = _call()
    if len(r0) < 16:
        pytest.skip("the extension in this environment was built before plan 79.02")
    r = _call(reask=True)
    assert len(r) == 16 and len(r[15]) == 6
    asked, held, reasked, out, unreached, first_cut = r[15]
    assert asked >= held >= out >= 0 and reasked + unreached <= held and first_cut in (0, 1)
    assert len(r[5]) <= max(held, 1) + 1 and r[0] in r[5]                    # the move played is in the set


def test_the_printer_has_the_drawing_mode():
    from catspace.incubator.analyst.discovery import ladder_print as LP
    assert callable(LP.cut02_cases)
