"""PLAN 79.01 A FIRM LABEL FOR THE ROOM, AND THE RESULT-FITTED VALUE AT THE END OF THE LINE (SYSTEM_SPEC 4 the equal reading, 5 and 6 the value fitted on results, 9 the oracle in sleep, 12 the debt of 78.84). Tests of the pure parts: the split by game, the depth inside a class, the firm label and its two sides, the rule that chooses the
margin on FIND (and the refusal to read anything before it is written), the ties of the highest chance, the scores of the five rules, the rule of the judged positions, the estimator's reading through its existing interface, the ways a move is put out in part three, and the wall (only the second asking calls the oracle; no
play path loads these modules). The layer files name no game object."""
import json
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import leaf_ask as LA
from catspace.incubator.analyst.discovery import leaf_debt as LD
from catspace.incubator.analyst.discovery import leaf_firm as LF
from catspace.incubator.analyst.discovery import leaf_line as LL
from catspace.incubator.analyst.discovery import leaf_measure as LM
from catspace.io import paths

DIR = Path(LF.__file__).parent
LAYER = ("leaf_firm.py", "leaf_ask.py", "leaf_line.py", "leaf_measure.py", "leaf_debt.py")
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


def test_the_wall_only_the_second_asking_calls_the_oracle_and_no_play_path_loads_these_modules():
    for f in LAYER:
        text = (DIR / f).read_text()
        calls = re.search(r"\bLO\.(analyse|analyse_after|label)\b", text)
        assert bool(calls) == (f == "leaf_ask.py"), f
    for p in DIR.glob("*.py"):
        if p.name.startswith("leaf_") or p.name in ("ladder_print.py",):
            continue
        assert not re.search(r"^\s*(from|import) .*\bleaf_(ask|firm|line|measure|debt)\b", p.read_text(), flags=re.M), p.name


# ---- the split and the firm label

def test_the_split_is_by_game_and_both_parts_exist():
    parts = {o: LF.part_of(o) for o in range(2000)}
    assert set(parts.values()) == {"find", "confirm"}
    assert all(LF.part_of(o) == parts[o] for o in range(2000))
    assert 700 < sum(1 for v in parts.values() if v == "find") < 1300


def test_the_depth_inside_a_class_is_the_distance_to_its_nearest_cut():
    w = LF.FL.WIN_CP
    assert LF.depth_inside(1, w + 30) == 30 and LF.depth_inside(-1, -w - 40) == 40
    assert LF.depth_inside(0, 0) == w and LF.depth_inside(0, w - 10) == 10 and LF.depth_inside(0, -w + 15) == 15
    assert LF.depth_inside(1, None) == float("inf") and LF.depth_inside(-1, None) == float("inf")


def test_a_label_is_firm_when_both_askings_agree_and_are_inside_the_class_by_the_margin():
    w = LF.FL.WIN_CP
    assert LF.firm_class(1, w + 60, 1, w + 80, 50) == 1
    assert LF.firm_class(1, w + 60, 1, w + 20, 50) is None              # the second asking is near the edge
    assert LF.firm_class(1, w + 60, 0, w - 5, 0) is None                # the askings disagree
    assert LF.firm_class(-1, None, -1, None, 1000) == -1                # a forced end is always inside


def rec(cls, cm, cp, tied=None):
    tied = tied or [f"m{i}" for i in range(len(cm))]
    return {"id": "1_10", "ord": 1, "cls": cls, "cm": cm, "cp": cp, "tied": tied, "depth": 5, "band": "mid", "ply": 10, "units": 20}


def test_a_move_firmly_keeps_or_firmly_loses_against_the_class_before_it():
    w = LF.FL.WIN_CP
    r = rec(0, [0, -1, 1, 0], [0, -w - 100, w + 100, 150])
    a = {"cls2": [0, -1, 1, 0], "cp2": [10, -w - 90, w + 120, 190]}
    v = LF.firm_vec(r, a, 50)
    assert v == ["keep", "lose", "keep", None]                           # the last: inside class 0 by 10 and 50 only against the cut at 200
    assert LF.is_firm(v) and not LF.is_firm(["keep", None]) and not LF.is_firm(["lose", "lose"])
    a_flip = {"cls2": [0, 0, 1, 0], "cp2": [10, -w + 5, w + 120, 190]}        # the second asking flips a class
    assert LF.firm_vec(r, a_flip, 50)[1] is None


def test_the_room_is_some_keeping_and_some_not_with_two_or_more_moves():
    assert LF.in_room(rec(0, [0, -1], [0, -300]))
    assert not LF.in_room(rec(0, [0, 1], [0, 300]))
    assert not LF.in_room(rec(0, [-1, -1], [-300, -300]))
    assert not LF.in_room(dict(rec(0, [0], [0]), depth=None))


# ---- the margin is chosen on FIND, before anything of CONFIRM is read

def test_the_margin_is_the_smallest_grid_value_with_enough_agreement_and_enough_moves():
    pairs = [(10, 0, 0)] * 90 + [(10, 0, 1)] * 10 + [(60, 0, 0)] * 300 + [(110, 1, 1)] * 300
    m, table = LF.choose_margin(pairs, grid=(0, 25, 50, 100), agree=0.99, min_moves=200)
    assert m == 25 and [t[1] for t in table] == [700, 600, 600, 300]
    m2, _t = LF.choose_margin(pairs[:100], grid=(0, 25), agree=0.99, min_moves=200)
    assert m2 == 25                                                       # nothing qualifies: the largest of the grid


def test_nothing_of_the_tables_runs_before_the_margin_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(LF, "out_root", lambda tag: tmp_path)
    with pytest.raises(RuntimeError):
        LF.load_margin("x")
    with pytest.raises(RuntimeError):
        LL.items_of("x")
    LF.save_margin("x", 25, [(0, 10, 9)], 7)
    assert LF.load_margin("x") == 25.0
    assert json.loads((tmp_path / "margin.json").read_text())["chosen_on"] == "find"


# ---- ties and the five rules

def test_the_highest_chance_shares_its_weight_over_ties_and_a_missing_value_gives_none():
    assert LF.weights([0.2, 0.7, 0.7 + 1e-12]) == [0.0, 0.5, 0.5]
    assert LF.weights([0.1, 0.5]) == [0.0, 1.0]
    assert LF.weights([0.1, None]) is None


def test_the_scores_of_the_five_rules_weigh_the_firm_labels_of_the_moves_each_takes():
    vec = ["lose", "keep", None, "keep"]
    s = LF.rule_scores(vec, 0, 1, [0.3, 0.3, 0.2, 0.3], [0.5, 0.5, 0.5, 0.5])
    assert s["first"] == (0.0, 1.0) and s["pick"] == (1.0, 0.0)
    assert s["random"] == (0.5, 0.25)
    assert s["end"] == pytest.approx((2 / 3, 1 / 3))                      # a three-way tie among moves 0, 1 and 3
    assert s["one"] == s["random"] or s["one"] == (0.5, 0.25)             # no difference: every move shares the weight


# ---- the second asking

def test_the_room_positions_to_ask_again_come_in_the_order_of_the_hash():
    recs = [rec(0, [0, -1], [0, -300]), dict(rec(0, [0, 1], [0, 300]), id="2_10", ord=2), dict(rec(0, [0, -1], [0, -300]), id="3_10", ord=3)]
    games = {1: {"moves": list("abcdefghijkl")}, 2: {"moves": list("abcdefghijkl")}, 3: {"moves": list("abcdefghijkl")}}
    items = LA.items_of(recs, games, 0)
    assert sorted(r["ord"] for r, _m in items) == [1, 3] and all(len(m) == 10 for _r, m in items)
    assert len(LA.items_of(recs, games, 1)) == 1
    a = LA.ask_record(recs[0], [0, -1], [5, None], 0.5)
    assert a["calls"] == 2 and a["cls2"] == [0, -1] and a["cp2"] == [5, None] and a["tied"] == recs[0]["tied"]


# ---- the estimator through its existing interface

REAL = Path(paths.experiment(*LL.RECORD))


@pytest.mark.skipif(not REAL.exists(), reason="the human-results estimator record is an untracked artifact")
def test_the_estimator_is_the_human_result_head_of_kinds_and_reads_for_the_root_mover():
    ch = LL.Chance()
    kinds = ch.kinds
    assert kinds and "human games" in ch.source
    flat = {k: [1, 1] for k in kinds}
    ahead = dict(flat, **{kinds[0]: [3, 1]})
    behind = dict(flat, **{kinds[0]: [1, 3]})
    assert ch.of(ahead, True) > ch.of(flat, True) > ch.of(behind, True)
    assert ch.of(ahead, False) == pytest.approx(1.0 - ch.of(ahead, True))
    assert 0.0 < ch.of(flat, True) < 1.0
    # the side to move's bonus: at equal units the two sides' chances do not add to one when each is read as the side to move; the symmetrised reading does
    assert ch.sym(flat) == pytest.approx(0.5)
    swapped = {k: [v[1], v[0]] for k, v in ahead.items()}
    assert ch.sym(ahead) + ch.sym(swapped) == pytest.approx(1.0) and ch.sym(ahead) > 0.5


def test_the_symmetrised_rule_is_scored_only_when_its_values_are_given():
    assert LF.rule_scores(["keep", "lose"], 0, 0, [0.1, 0.2], [0.1, 0.1])["end_sym"] is None
    assert LF.rule_scores(["keep", "lose"], 0, 0, [0.1, 0.2], [0.1, 0.1], [0.7, 0.2])["end_sym"] == (1.0, 0.0)


# ---- the table of part two and the judged positions

class FakeChance:
    def sym(self, inputs):
        return inputs["v"]

    def of(self, inputs, root):
        return inputs["v"] if root else 1.0 - inputs["v"]


def line_row(v_end, v_one, root=True):
    return {"ok": True, "end": {"v": v_end}, "end_is_root": root, "one": {"v": v_one}, "one_is_root": False}


def test_the_rule_table_reads_the_same_positions_for_every_rule_and_leaves_out_unread_lines():
    w = LF.FL.WIN_CP
    recs = {"1_10": dict(rec(0, [0, -1], [0, -w - 100]), pick="m0", band="few"), "2_10": dict(rec(0, [0, -1], [0, -w - 100]), id="2_10", ord=2, pick="m1"), "3_10": dict(rec(0, [0, -1], [0, -w - 100]), id="3_10", ord=3, pick="m0")}
    asks = {i: {"cls2": [0, -1], "cp2": [0, -w - 100]} for i in recs}
    lines = {"1_10": {"rows": [line_row(0.4, 0.5), line_row(0.2, 0.5)]},            # the chance at the end of the line picks the keeping move
             "2_10": {"rows": [line_row(0.2, 0.5), line_row(0.6, 0.5)]},            # it picks the losing move (its chance, for the root mover, is the higher)
             "3_10": {"rows": [{"ok": False}, line_row(0.1, 0.1)]}}
    t = LM.rule_table(recs, asks, lines, FakeChance(), 25)
    assert [r["id"] for r in t["rows"]] == ["1_10", "2_10"] and t["skipped"] == 1
    r1, r2 = t["rows"]
    assert r1["scores"]["end"] == (1.0, 0.0) and r2["scores"]["end"] == (0.0, 1.0)
    assert r1["end_unique"] and r1["one_flat"] and r1["end_argmax"] == [0]
    assert r2["scores"]["pick"] == (0.0, 1.0) and r2["end_argmax"] == [1]
    assert LM.pick_cases(t["rows"], 15) == [1]                                      # a unique highest end-of-line chance on a move that is not the first move


def test_the_rows_of_the_table_for_the_judged_positions_keep_the_order_they_came_in():
    rows = [{"end_unique": True, "end_argmax": [0]}, {"end_unique": True, "end_argmax": [2]}, {"end_unique": False, "end_argmax": [1, 2]}, {"end_unique": True, "end_argmax": [1]}]
    assert LM.pick_cases(rows, 15) == [1, 3] and LM.pick_cases(rows, 1) == [1]


# ---- part three

def test_a_move_is_put_out_when_its_answer_is_under_the_value_and_a_cut_read_gives_no_answer():
    assert LD.put_out_audit({"plain": 5}, 10) is True and LD.put_out_audit({"plain": 10}, 10) is False and LD.put_out_audit(None, 10) is None
    assert LD.put_out_exact({"exact": -3}, 0) is True and LD.put_out_exact({"exact": None}, 0) is None
    ans = {"first": "m0", "equals": ["m1"], "cut": False, "error": None}
    assert LD.put_out_clock(["m0", "m1", "m2"], ans) == [False, False, True]
    assert LD.put_out_clock(["m0"], dict(ans, cut=True)) == [None] and LD.put_out_clock(["m0"], None) == [None]


def test_the_drawing_mode_exists_in_the_printer():
    from catspace.incubator.analyst.discovery import ladder_print as LP
    assert callable(LP.leaf01_cases)
