"""PLAN 79.03 THE SEARCH'S VALUE UNDER THE BEST MOVE IT COULD READ AT ITS OWN DEPTH, AND THE MEND (SYSTEM_SPEC 4 the tactical reasoner's deepening and its prunings, 12 the debt of plan 78.84). Tests of the pure parts of the count (the weakest unit, the kind of a gap, a position's row, which pruning the best move's reading passed through), of the measure of the mend
(the class of a move, the kind of a fault, the conditions of the rule, the refusal to read CONFIRM before the rule is written, the arms), of the wall (no oracle call in these modules, no play path loads them), of the switch in the binding (off by default; on it changes what it names) and of the printer. The layer files name no game object."""
import json
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import best_cases as BCS
from catspace.incubator.analyst.discovery import best_count as BC
from catspace.incubator.analyst.discovery import best_mend as BM
from catspace.incubator.analyst.discovery import best_play as BP
from catspace.incubator.analyst.discovery import best_which as BW

DIR = Path(BC.__file__).parent
LAYER = ("best_count.py", "best_which.py", "best_mend.py", "best_play.py", "best_out.py", "best_cases.py")
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
        if p.name.startswith("best_") or p.name in ("ladder_print.py",):
            continue
        assert not re.search(r"^\s*(from|import) .*\bbest_(count|which|mend|play|out|cases)\b", p.read_text(), flags=re.M), p.name
    play = (DIR / "best_play.py").read_text()
    assert not re.search(r"^\s*(from|import) .*\b(leaf_\w+|cut_count|cut_mend|best_count|best_mend|best_which|best_cases|ladder_oracle|prior_equal|stack_room)\b", play, flags=re.M)


def test_the_match_module_does_not_load_the_oracle_wrapper_even_through_what_it_imports():
    import subprocess
    import sys
    code = ("import sys; import catspace.incubator.analyst.discovery.best_play; "
            "bad = [m for m in sys.modules if m.endswith('ladder_oracle') or m.endswith('leaf_ask') or m.endswith('cut_count') or m.endswith('cut_mend') or m.endswith('best_count') or m.endswith('best_mend')]; "
            "print('LOADED', bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_statement_of_what_the_code_does_names_the_root_the_reduction_and_the_null_move():
    t = BC.STATEMENT
    for w in ("AT THE ROOT", "TWO PLIES SHALLOWER", "read again at the full depth", "NOT above alpha", "THE NULL MOVE", "never tried at the root", "NOT tried one ply under the root"):
        assert w in t, w


# ---- part one

def test_the_weakest_unit_is_the_smallest_positive_worth():
    assert BC.weakest_unit({"a": 475, "b": 1216, "c": 0}) == 475


def test_a_gap_is_none_reverse_less_unit_or_the_games_end():
    u = 475
    assert BC.gap_kind(0, u, False) == "none" and BC.gap_kind(None, u, False) == "none"
    assert BC.gap_kind(-10, u, False) == "reverse"
    assert BC.gap_kind(10, u, False) == "less" and BC.gap_kind(474, u, False) == "less"
    assert BC.gap_kind(475, u, False) == "unit" and BC.gap_kind(5000, u, False) == "unit"
    assert BC.gap_kind(5000, u, True) == "ends" and BC.gap_kind(0, u, True) == "none"


def _rec(first="m0", tied=("m0", "m1"), cm=(0, 1), cls=0):
    return {"id": "7_30", "ord": 7, "band": "mid", "first": first, "tied": list(tied), "cm": list(cm), "cls": cls, "cut": False}


def _aud(value, exact):
    return {"value": value, "finished": 6, "rows": [{"name": n, "exact": e} for n, e in exact]}


def test_a_position_row_holds_the_gap_the_played_move_and_the_best_moves_stored_class():
    r = BC.position_row(_rec(), _aud(100, [("m0", 100), ("m1", 700), ("m2", 50)]), 475)
    assert r["gap"] == 600 and r["kind"] == "unit" and r["above"] == 1 and r["best_move"] == "m1" and r["best_in_set"]
    assert r["played_under_best"] and r["played_keeps"] is True and r["best_keeps"] is True
    r = BC.position_row(_rec(cm=(-1, 1)), _aud(100, [("m0", 100), ("m1", 100)]), 475)
    assert r["kind"] == "none" and r["played_keeps"] is False and not r["played_under_best"]
    r = BC.position_row(_rec(), _aud(100, [("m0", 100), ("m9", 700)]), 475)
    assert r["kind"] == "unit" and not r["best_in_set"] and r["best_keeps"] is None             # the best move has no stored label
    assert BC.position_row(_rec(), _aud(100, [("m0", None)]), 475) is None
    r = BC.position_row(_rec(), _aud(100, [("m0", BC.WIN - 3)]), 475)
    assert r["kind"] == "ends"


# ---- part one, second half

def test_the_pruning_a_best_moves_reading_passed_through_is_the_one_whose_removal_lifts_it():
    t = 100
    f = lambda nn, nl, pl: {"other_no_null": nn, "other_no_late": nl, "other_plain": pl}
    assert BW.cause_of(t, f(50, 100, 100)) == "late"
    assert BW.cause_of(t, f(100, 50, 100)) == "null"
    assert BW.cause_of(t, f(100, 120, 120)) == "either"
    assert BW.cause_of(t, f(50, 50, 100)) == "joint"
    assert BW.cause_of(t, f(50, 50, 60)) == "neither"
    assert BW.cause_of(t, f(None, 100, 100)) == "unread"


# ---- part three

def test_the_class_of_a_move_is_its_stored_label_or_unlabelled():
    lab = {"a": True, "b": False}
    assert [BM.class_of(x, lab) for x in ("a", "b", "z", None)] == ["keep", "lose", "unlabelled", "unlabelled"]
    assert BM.labels_of({"tied": ["a", "b"], "cm": [1, -1], "cls": 0}) == {"a": True, "b": False}


def test_a_fault_is_a_gap_of_a_unit_or_more_or_the_games_end_and_an_unread_audit_has_none():
    u = 475
    assert BM.fault_of({"value": 0, "best_exact": 475}, u) == "unit" and BM.is_fault("unit") and BM.is_fault("ends")
    assert BM.fault_of({"value": 0, "best_exact": 100}, u) == "less" and not BM.is_fault("less")
    assert BM.fault_of({"value": 0, "best_exact": None}, u) == "unread" and BM.fault_of(None, u) == "unread"
    assert BM.fault_of({"value": 0, "best_exact": BC.WIN - 4}, u) == "ends"


def _pos(game, fault_a, fault_m, pick_a, pick_m, depth=(7.0, 7.0), sec=(0.2, 0.2), band="mid", part="find"):
    return {"id": f"{game}_1", "game": game, "band": band, "part": part, "pick": {"a": pick_a, "m": pick_m, "n": pick_a}, "first": {"a": pick_a, "m": pick_m, "n": pick_a},
            "fault": {"a": fault_a, "m": fault_m, "n": fault_a}, "reads": {"a": {"depth": depth[0], "seconds": sec[0]}, "m": {"depth": depth[1], "seconds": sec[1]}, "n": {"depth": depth[0], "seconds": sec[0]}}}


def test_the_four_conditions_pass_for_a_mend_that_lowers_the_fault_and_lifts_the_pick_at_no_great_cost_and_fail_otherwise():
    good = [_pos(g, "unit", "none", "lose", "keep", depth=(7.0, 6.5)) for g in range(1, 41)]
    c = BM.conditions(good)
    assert c["pick_pass"] and c["fault_pass"] and c["depth_pass"] and c["time_pass"]
    assert c["fault_change"][0] == pytest.approx(-1.0) and c["pick_gain"][0] == pytest.approx(1.0)
    flat = [_pos(g, "unit", "none", "keep", "keep") for g in range(1, 41)]
    assert not BM.conditions(flat)["pick_pass"]                                   # the fault falls and the pick does not move: the rule is not met
    deep = [_pos(g, "unit", "none", "lose", "keep", depth=(7.0, 5.5)) for g in range(1, 41)]
    assert not BM.conditions(deep)["depth_pass"]                                   # the horizon was bought
    slow = [_pos(g, "unit", "none", "lose", "keep", sec=(0.2, 0.35)) for g in range(1, 41)]
    assert not BM.conditions(slow)["time_pass"]
    same = [_pos(g, "none", "none", "lose", "keep") for g in range(1, 41)]
    assert not BM.conditions(same)["fault_pass"]
    unl = [_pos(g, "unit", "none", "unlabelled", "keep") for g in range(1, 41)]
    assert BM.conditions(unl)["pick_gain"][0] != BM.conditions(unl)["pick_gain"][0]       # no labelled pair: nan


def test_confirm_is_not_read_before_the_rule_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(BM, "out_root", lambda tag: tmp_path)
    with pytest.raises(RuntimeError):
        BM.cmd_count("x", "confirm", lambda s: None)
    said = []
    BM.cmd_rule("x", said.append)
    rule = json.loads((tmp_path / "rule.json").read_text())
    assert rule["depth_loss_max"] == BM.DEPTH_LOSS_MAX and "before CONFIRM" in said[0]


def test_the_three_arms_are_read_in_the_order_switch_off_on_off_and_only_the_second_has_the_switch():
    assert BM.ARMS == ("a", "m", "n") and BM.SWITCH == {"a": False, "m": True, "n": False}


def test_the_clock_of_the_mend_measure_is_the_play_clock_unless_a_supplement_says_otherwise():
    from catspace.incubator.analyst.discovery import tiebreak_measure as TM
    assert BM.CLOCK == TM.SECONDS


def test_the_judged_positions_are_position_two_of_79_01_and_the_first_fourteen_of_kind_unit_in_the_hash_order():
    from catspace.incubator.analyst.discovery import prior_equal as PE
    rows = [{"id": f"{g}_{p}", "game": g, "kind": ("unit" if g % 2 else "less")} for g in range(1, 80) for p in (10, 20)]
    ids = BCS.pick_ids(rows, "5_10")
    assert ids[0] == "5_10" and len(ids) == 15 and len(set(ids)) == 15 and "5_10" not in ids[1:]
    assert all(next(r for r in rows if r["id"] == i)["kind"] == "unit" for i in ids)
    key = [PE.order_key(int(i.split("_")[0]), int(i.split("_")[1])) for i in ids[1:]]
    assert key == sorted(key)


# ---- the match

def test_the_two_sides_of_the_match_differ_in_the_switch_alone():
    a, b = BP.player_spec("root"), BP.player_spec("stack")
    assert a.pop("root_full") is True and b.pop("root_full") is False
    assert a == b and a["verify_choice"] is True and a["choice_mode"] == "slot"
    specs = BP.match_specs("root", "stack", "A", 2, 0.2, Path("/nonexistent"))
    assert len(specs) == 4 and {s["players"][0]["root_full"] for s in specs} == {True, False}
    assert all(s["max_plies"] == 120 and s["seconds"] == 0.2 for s in specs)


def test_the_movers_pass_the_switch_only_when_the_spec_asks():
    from catspace.incubator.analyst.discovery import tiebreak_measure as TM
    m = TM.Mover67.__new__(TM.Mover67)
    m.verify_choice, m.choice_mode, m.reask, m.root_full = True, "slot", False, False
    assert m.choice_kw() == {"verify_choice": True, "choice_mode": "slot"}
    m.root_full = True
    assert m.choice_kw() == {"verify_choice": True, "choice_mode": "slot", "root_full": True}
    m.verify_choice, m.root_full = False, False
    assert m.choice_kw() == {}


# ---- the switch in the binding

MOVES = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]


def _call(**kw):
    import cs_engine as CE
    from catspace.incubator.analyst.discovery import record_play as RP
    return CE.solve_equals_batch([("chess", None, list(MOVES))], 2.0, worth=dict(RP.worth_of("human")), ours_first=False, valuation="odds record", threads=1, ledger_share=0.4, verify_choice=True, choice_mode="slot", **kw)[0]


def test_the_binding_row_keeps_its_sixteen_elements_with_the_switch_off_and_on():
    import cs_engine as CE
    if not hasattr(CE, "gap_audit"):
        pytest.skip("the extension in this environment was built before plan 79.03")
    off, on = _call(), _call(root_full=True)
    assert len(off) == 16 and len(on) == 16 and list(off[15]) == [] and list(on[15]) == []
    assert off[4] is None and on[4] is None and off[10] > 0 and on[10] == off[10] and on[12] in (0, 1, 2, 3, 4)


def test_the_gap_audit_reads_the_deepening_and_every_root_move_with_the_switch_as_a_parameter():
    import cs_engine as CE
    from catspace.incubator.analyst.discovery import record_play as RP
    from catspace.incubator.analyst.discovery import odds_measure as OM
    if not hasattr(CE, "gap_audit"):
        pytest.skip("the extension in this environment was built before plan 79.03")
    w = dict(RP.worth_of("human"))
    a = CE.gap_audit(OM.VARIANT, list(MOVES), w, 2_000_000, 5, 2_000_000, True, False)
    b = CE.gap_audit(OM.VARIANT, list(MOVES), w, 2_000_000, 5, 2_000_000, True, True)
    assert a["depth"] == 5 and b["depth"] == 5 and a["first"] and b["first"] and not a["exact_cut"]
    assert len(a["rows"]) == len(b["rows"]) > 5 and {r["name"] for r in a["rows"]} == {r["name"] for r in b["rows"]}
    assert a["rows"] == b["rows"]                                                  # the exact readings do not depend on the switch
    again = CE.gap_audit(OM.VARIANT, list(MOVES), w, 2_000_000, 5, 2_000_000, True, False)
    assert (again["value"], again["run_nodes"], again["first"]) == (a["value"], a["run_nodes"], a["first"])          # no clock: it repeats


def test_the_printer_has_the_drawing_mode():
    from catspace.incubator.analyst.discovery import ladder_print as LP
    assert callable(LP.best03_cases)
