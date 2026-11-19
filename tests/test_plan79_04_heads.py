"""PLAN 79.04 THE RESULT-FITTED HEADS WE ALREADY HAVE, WITH THE GRAPH'S STATEMENTS AS INPUTS, AT THE EQUAL MOVES (SYSTEM_SPEC 5 and 6 the value fitted on results and its heads, 4 the leaf, 9 the oracle in sleep). Tests of the pure parts: the record's heads and their column layout, the view from both sides (two lines of different parity ending in the same position get the same value, the raw reading does not),
the ties, the carriers of a difference, the count of part two, the refusal to read CONFIRM before the best head is chosen on FIND, and the wall (no oracle call; no play path loads these modules). The layer files name no game object."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from catspace.incubator.analyst.discovery import heads_measure as HM
from catspace.incubator.analyst.discovery import heads_read as HRD
from catspace.io import paths

DIR = Path(HRD.__file__).parent
LAYER = ("heads_read.py", "heads_measure.py")
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")
RECORD = Path(paths.experiment("estimator_78_59", "full", "estimator_heads_human.json"))
needs_record = pytest.mark.skipif(not RECORD.exists(), reason="the record of plan 78.59 is not in this checkout")


def test_the_layer_files_name_no_game_object():
    for f in LAYER:
        text = (DIR / f).read_text().lower()
        assert not re.search(r"\b[a-h][1-8]\b", text), f
        assert "sys.path.insert" not in text, f
        for w in BANNED:
            assert not re.search(rf"\b{w}\b", text), (f, w)


def test_this_test_file_names_no_game_object_either():
    assert not re.search(r"\b[a-h][1-8]\b", Path(__file__).read_text().lower())


def test_the_wall_no_oracle_call_and_no_play_path_loads_these_modules():
    for f in LAYER:
        assert not re.search(r"\bLO\.(analyse|analyse_after|label)\b", (DIR / f).read_text()), f
    for p in DIR.glob("*.py"):
        if p.name.startswith(("leaf_", "cut_", "best_", "heads_")) and p.name != "best_play.py" or p.name == "ladder_print.py":
            continue
        assert not re.search(r"^\s*(from|import) .*\bheads_(read|measure)\b", p.read_text(), flags=re.M), p.name


def test_the_drawing_mode_exists_in_the_printer():
    text = (DIR / "ladder_print.py").read_text()
    assert "def heads04_cases" in text and "--heads04" in text and "a.heads04" in text


# ---- the record and the layout

@needs_record
def test_every_head_of_the_spec_is_in_the_record_and_its_layout_has_the_columns_of_its_model():
    rec = HRD.Record()
    n = 7
    blocks = {"K": np.arange(n * 5, dtype=float).reshape(n, 5), "S": np.ones((n, 26)), "C": np.ones((n, 64))}
    for name in rec.heads:
        x = rec.columns(name, blocks, 40)
        h = rec.head(name)
        want = len(h["transform"]["scale"]) if h.get("transform") else (len(h["model"]["mean"]) if rec.kind(name) == "value" else np.asarray(h["model"]["coef"]).shape[1])
        assert x.shape == (n, want), name
        assert rec.stm(name, blocks, 40).shape == (n,)


@needs_record
def test_ratings_are_an_average_pair_and_the_ply_is_the_one_given():
    rec = HRD.Record()
    blocks = {"K": np.zeros((2, 5)), "S": np.zeros((2, 26))}
    x = rec.columns("result_human_RKS", blocks, 33)
    h = rec.head("result_human_RKS")
    assert x[0, 0] == pytest.approx(h["transform"]["centre"][0]) and x[0, 1] == 0.0 and x[0, 2] == 33.0
    y = rec.columns("arrival_human_KS_R", blocks, 21)
    m = rec.head("arrival_human_KS_R")["model"]["mean"]
    assert y[0, -3] == pytest.approx(m[-3]) and y[0, -2] == 0.0 and y[0, -1] == 21.0


# ---- the view from both sides

def random_blocks(rng, n=1):
    return {"K": rng.integers(-3, 4, (n, 5)).astype(float), "S": rng.normal(0, 3, (n, 26)), "C": rng.integers(-2, 3, (n, 64)).astype(float)}


@needs_record
def test_two_lines_of_different_parity_ending_in_the_same_position_get_the_same_value_from_both_sides_view():
    rec = HRD.Record()
    rng = np.random.default_rng(3)
    for name in rec.heads:
        need = set(rec.letters(name)) - {"R"}
        for _ in range(5):
            full = random_blocks(rng)
            x = {k: v for k, v in full.items() if k in need}
            a = rec.view(name, x, 30, np.array([True]), True)          # the root mover is to move at the end of this line: the inputs are the root mover's
            b = rec.view(name, swapped_of(x), 30, np.array([False]), True)   # the same position at the end of a line one ply longer: the other side is to move, its inputs are the negatives
            assert a[0] == pytest.approx(b[0], abs=1e-12), name


def swapped_of(x):
    return HRD.swapped(x)


@needs_record
def test_the_raw_reading_orders_the_same_two_lines_by_their_parity_and_the_both_sides_reading_does_not():
    rec = HRD.Record()
    rng = np.random.default_rng(5)
    x = {"K": rng.integers(-1, 2, (1, 5)).astype(float)}
    raw_a = rec.view("result_human_K", x, 30, np.array([True]), False)[0]
    raw_b = rec.view("result_human_K", swapped_of(x), 30, np.array([False]), False)[0]
    assert abs(raw_a - raw_b) > 1e-6                              # the side to move's bonus
    sym_a = rec.view("result_human_K", x, 30, np.array([True]), True)[0]
    sym_b = rec.view("result_human_K", swapped_of(x), 30, np.array([False]), True)[0]
    assert sym_a == pytest.approx(sym_b, abs=1e-12)


@needs_record
def test_an_equal_position_gets_the_neutral_value_in_both_sides_view():
    rec = HRD.Record()
    z = {"K": np.zeros((1, 5)), "S": np.zeros((1, 26)), "C": np.zeros((1, 64))}
    assert rec.view("result_human_RKS", {"K": z["K"], "S": z["S"]}, 30, np.array([True]), True)[0] == pytest.approx(0.5, abs=1e-9)
    assert rec.view("arrival_human_KS", {"K": z["K"], "S": z["S"]}, 30, np.array([True]), True)[0] == pytest.approx(0.0, abs=1e-9)


@needs_record
def test_a_value_is_for_the_root_mover_a_probability_one_less_and_an_arrival_value_negated_when_the_other_side_is_to_move():
    rec = HRD.Record()
    rng = np.random.default_rng(9)
    x = {"K": rng.integers(-2, 3, (2, 5)).astype(float), "S": rng.normal(0, 2, (2, 26))}
    for name, op in (("result_human_RKS", lambda v: 1.0 - v), ("arrival_human_KS", lambda v: -v)):
        a = rec.view(name, x, 30, np.array([True, True]), False)
        b = rec.view(name, x, 30, np.array([False, False]), False)
        assert np.allclose(b, op(a))


# ---- the ties, the carriers, the count

def test_the_highest_value_shares_its_weight_and_the_rule_scores_the_firm_labels_of_the_moves_it_takes():
    s = HM.scores_of(["keep", "lose", "keep"], [0.5, 0.5, 0.2])
    assert s["keep"] == pytest.approx(0.5) and s["lose"] == pytest.approx(0.5) and s["sep"] == 1.0 and s["uniq"] == 0.0 and s["argmax"] == [0, 1]
    flat = HM.scores_of(["keep", "lose"], [0.3, 0.3])
    assert flat["sep"] == 0.0 and flat["keep"] == pytest.approx(0.5)
    one = HM.scores_of(["keep", "lose"], [0.1, 0.9])
    assert one["uniq"] == 1.0 and one["lose"] == 1.0
    assert HM.fixed_scores(["keep", "lose", None], 0) == {"keep": 1.0, "lose": 0.0}
    r = HM.fixed_scores(["keep", "lose", None, "lose"], None)
    assert r["keep"] == pytest.approx(0.25) and r["lose"] == pytest.approx(0.5)


@needs_record
def test_the_carriers_of_an_arrival_head_add_up_to_the_difference_of_its_two_sides_values():
    rec = HRD.Record()
    rng = np.random.default_rng(11)
    a, b = random_blocks(rng), random_blocks(rng)
    names = HM.statement_names()
    need = {"K", "S"}
    xa, xb = ({k: v[0] for k, v in blk.items() if k in need} for blk in (a, b))
    car = HM.carriers(rec, "arrival_human_KS", xa, True, xb, False, 30, names, top=31)
    va = rec.view("arrival_human_KS", {k: v for k, v in a.items() if k in need}, 30, np.array([True]), True)[0]
    vb = rec.view("arrival_human_KS", {k: v for k, v in b.items() if k in need}, 30, np.array([False]), True)[0]
    assert sum(d for _n, d in car) == pytest.approx(va - vb, abs=1e-9)


def test_within_gives_the_keepers_less_the_losers_and_nothing_without_both():
    x = np.array([[1.0, 5.0], [3.0, 5.0], [0.0, 5.0]])
    d, v = HM.within(x, ["keep", "keep", "lose"])
    assert d.tolist() == [2.0, 0.0] and v[1] == 0.0
    assert HM.within(x, ["keep", "keep", None]) is None


def test_the_effect_of_a_statement_is_its_mean_difference_over_its_within_position_spread():
    diffs = np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    var = np.array([[4.0, 1.0]] * 4)
    games = np.array([0, 1, 2, 3])
    eff, bands, above, nz = HM.effects(diffs, var, games)
    assert eff[0] == pytest.approx(1.0) and eff[1] == pytest.approx(0.0)
    assert bands[0][1] <= 1.0 <= bands[0][2]
    assert above[0] == 1.0 and np.isnan(above[1]) and nz.tolist() == [1.0, 0.0]


# ---- the order of the stages

def test_nothing_of_the_tables_runs_before_the_best_head_is_chosen_on_find(tmp_path, monkeypatch):
    monkeypatch.setattr(HRD, "out_root", lambda tag: tmp_path)
    with pytest.raises(RuntimeError):
        HM.load_best("x")
    (tmp_path / "best.json").write_text(json.dumps({"head": "h", "reading": "end"}))
    assert HM.load_best("x")["head"] == "h"


def test_the_blocks_of_a_position_say_whose_turn_it_is_at_each_reading():
    row = {"K": [0] * 5, "S": [0.0] * 26, "C": None, "quiet": True, "ok": True}
    pos = {"id": "p", "ply": 5, "rows": [{"one": row, "end": row, "end_is_root": True}, {"one": row, "end": row, "end_is_root": False}]}
    _b, one = HM.blocks_of(pos, "one")
    _b, end = HM.blocks_of(pos, "end")
    assert one.tolist() == [False, False] and end.tolist() == [True, False]


def test_the_short_wire_starts_at_the_roots_position_text_and_a_head_list_is_complete():
    assert HRD.NAMES[1] == HRD.BASELINE and len(set(HRD.NAMES)) == len(HRD.NAMES)
    assert all(set(s[3]) <= set("RKSC") for s in HRD.SPEC)
