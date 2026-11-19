"""PLAN 78.98 WHY GIVING UP A STRONGER UNIT FOR A WEAKER ONE IS HELD EQUAL (SYSTEM_SPEC 4 the tactical reasoner's equal reading, 5 the value, 11 the layers are game-free, 12 the debt of plan 78.84): the causes of part one (the order they apply in), the
exchange read off a line's ledger, the counts' helpers. The layer files name no game object. (Part two's Rust test is in rust/engine/tactics/tests.)"""
import re
from pathlib import Path

from catspace.incubator.analyst.discovery import equal_measure as EM
from catspace.incubator.analyst.discovery import equal_trace as ET

DIR = Path(ET.__file__).parent
LAYER = ("equal_trace.py", "equal_measure.py")
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


def row(warm=0, exact=0, plain=0):
    return {"warm": warm, "exact": exact, "plain": plain}


def test_a_move_the_rule_does_not_hold_is_not_counted_among_the_causes():
    assert ET.cause_of(10, row(warm=9), None) == "not held"
    assert ET.cause_of(10, None, None) == "not held"


def test_cause_b_is_an_exact_reading_under_the_value_or_the_same_depth_reading_under_its_best():
    assert ET.cause_of(10, row(warm=10, exact=4), None) == "b"
    assert ET.cause_of(10, row(warm=10, plain=4), None) == "b"
    deep = {"other_d": 10, "best_d": 30, "other": 10, "best": 10}
    assert ET.cause_of(10, row(warm=10, exact=10, plain=10), deep) == "b"                 # the value v itself is below the best of the unpruned reading


def test_cause_c_is_a_loss_the_longer_reading_shows_and_cause_a_is_none_at_any_depth_read():
    same = {"other_d": 10, "best_d": 10}
    assert ET.cause_of(10, row(warm=10, exact=10, plain=10), dict(same, other=3, best=10)) == "c"
    assert ET.cause_of(10, row(warm=10, exact=10, plain=10), dict(same, other=10, best=10)) == "a"
    assert ET.cause_of(10, row(warm=10, exact=10, plain=10), dict(same, other=None, best=10)) == "unread"
    assert ET.cause_of(10, row(warm=10, exact=None, plain=10), dict(same, other=10, best=10)) == "unread"
    assert ET.cause_of(10, row(warm=10, exact=10, plain=10), None) == "unread"


def test_an_exchange_is_read_from_the_ledger_with_the_movers_side_first():
    ex = ET_exchange([(True, "x"), (False, "y"), (False, "z")], [(True, "a", "b"), (False, "c", "d")])
    assert ex == {"gave": ["x"], "took": ["y", "z"], "changed": [("a", "b")]}


def ET_exchange(events, changes):
    return ET.exchange_of(events, changes)


def test_the_record_makes_an_exchange_a_gain_a_loss_or_nothing():
    worth = {"x": 5, "y": 3}
    assert EM.net_by_record({"gave": ["x"], "took": ["y"]}, worth) == -2
    assert EM.net_by_record({"gave": ["y"], "took": ["x"]}, worth) == 2
    assert EM.net_by_record({"gave": [], "took": []}, worth) == 0
    assert [EM.sign_word(v) for v in (2, 0, -2)] == ["more than nothing", "nothing", "less than nothing"]


def test_the_game_of_a_position_is_the_first_part_of_its_id():
    assert EM.game_of({"id": "123_45"}) == 123
