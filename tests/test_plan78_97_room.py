"""PLAN 78.97 WHY A LOSING MOVE IS HELD EQUAL, AND WHAT THE VERIFIED CHOICE TAKES OF THE ROOM (SYSTEM_SPEC 5 the equal reading and the verified choice, 9 the oracle in sleep and the wall, 11 the layers are game-free): the room, the leaving of a move from a set, the stuck set, the split of a loss
into tactical and slow, the kind of the missing statement, the picks' set form, the tables' bands, and the wall (the picks' module never imports the oracle wrapper). The layer files name no game object."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import room_lines as RL
from catspace.incubator.analyst.discovery import room_measure as RM
from catspace.incubator.analyst.discovery import room_picks as RP
from catspace.incubator.analyst.discovery import room_reads as RR

DIR = Path(RR.__file__).parent
LAYER = ("room_reads.py", "room_picks.py", "room_lines.py", "room_measure.py")
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


def test_only_the_sleep_stage_calls_the_oracle_wrapper():
    """THE WALL (SYSTEM_SPEC 9): room_lines is the one file that imports the oracle wrapper and calls it; the reads, the picks and the tables never name it and make no call of the kind (they load the games through future_labels, as the measure files of 78.96 do),
    and the fitted prior they load (prior_pick) still imports no oracle in a fresh process."""
    for f in ("room_reads.py", "room_picks.py", "room_measure.py"):
        text = (DIR / f).read_text()
        assert not re.search(r"^\s*(from|import) .*ladder_oracle", text, flags=re.M), f
        assert "analyse" not in text, f
    assert re.search(r"^from .* import ladder_oracle as LO", (DIR / "room_lines.py").read_text(), flags=re.M)
    code = "import sys; import catspace.incubator.analyst.discovery.prior_pick as P; print('LOADED' if any(m.endswith('ladder_oracle') for m in sys.modules) else 'CLEAN')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.stdout.strip().endswith("CLEAN"), (out.stdout, out.stderr[-500:])


def rec(rid="1_10", cls=1, cm=(1, 0, 1, 0), tied=("a", "b", "c", "d"), ord_=1, ply=10):
    return {"id": rid, "ord": ord_, "ply": ply, "cls": cls, "cm": list(cm), "tied": list(tied), "part": "find", "units": 20, "band": "many"}


def row(first="a", equals=("a", "b"), cut=False, error=None, state=0, depth=6):
    return {"first": first, "equals": list(equals), "cut": cut, "error": error, "state": state, "depth": depth, "seconds": 0.1}


def test_the_room_is_the_positions_where_some_equal_moves_keep_the_class_and_some_do_not():
    a = rec("1_10", 1, (1, 0, 1))
    b = rec("2_10", 1, (1, 1, 1), ord_=2)                   # every move keeps it: no room
    c = rec("3_10", 1, (0, -1, 0), ord_=3)                  # none keeps it: no room (the search's part, not the choice's)
    d = rec("4_10", 1, (1,), ord_=4, tied=("a",))           # one move: no choice
    e = rec("5_10", -1, (-1, -1, 0), ord_=5)                # a losing position has nothing to keep
    out = RR.room_ids([b, c, d, e, a])
    assert [r["id"] for r in out] == ["1_10"]


def test_the_room_is_in_a_fixed_order():
    rs = [rec(f"{k}_10", 1, (1, 0, 1), ord_=k) for k in range(1, 30)]
    assert RR.room_ids(rs) == RR.room_ids(list(reversed(rs)))


def test_a_move_leaves_a_set_when_it_is_not_in_it_and_a_cut_or_failed_run_has_no_answer():
    r = rec()                                               # moves a b c d; a and c keep, b and d do not
    lv = RR.leaves(r, row("a", ("a", "c")))
    assert lv == {"lose": [True, True], "keep": [False, False]}
    lv = RR.leaves(r, row("c", ("a", "b")))                 # the run's own move counts as held
    assert lv == {"lose": [False, True], "keep": [False, False]}
    assert RR.leaves(r, row(cut=True)) is None
    assert RR.leaves(r, row(error="x")) is None
    assert RR.leaves(r, row(first=None, equals=())) is None


def test_the_stuck_moves_are_class_losing_moves_held_at_every_clock_and_need_every_run_read():
    r = rec()
    rows = {"x5": row("a", ("a", "b", "c", "d")), "x25": row("a", ("a", "b", "c")), "off": row("a", ("a", "b", "d"))}
    assert RR.stuck_of(r, rows) == ["b"]                    # d left at 25 times, b stayed everywhere
    rows["x25"] = row("a", ("a", "b", "c"), cut=True)
    assert RR.stuck_of(r, rows) == []                       # a cut run is no answer: nothing is called stuck on it
    assert RR.stuck_of(r, {"x5": rows["x5"]}) == []


def test_the_settings_are_the_play_clock_five_times_it_twenty_five_times_it_and_the_prunings_off():
    s = RR.settings(0.2)
    assert set(s) == set(RR.SETTINGS)
    assert s["x5"][0] == 1.0 and s["x25"][0] == 5.0 and s["r1"][0] == s["off"][0] == s["on"][0] == 0.2
    assert s["off"][1] == {"null_move": False, "late_reduce": False}
    assert s["on"][1]["verify_choice"] is True and "verify_choice" not in s["r1"][1]


def test_the_picks_read_the_set_with_the_runs_own_move_first():
    assert RP.tied_of(row("c", ("a", "b", "c"))) == ["c", "a", "b"]
    assert RP.tied_of(row("z", ("a", "b"))) == ["z", "a", "b"]


def test_a_loss_is_tactical_when_units_go_or_the_game_ends_within_six_plies():
    assert RL.is_tactical(1000, None, 6)
    assert not RL.is_tactical(0, None, 6) and not RL.is_tactical(-500, None, 6)         # even or gained: slow
    assert RL.is_tactical(0, -3, 5) and not RL.is_tactical(0, -4, 6) and not RL.is_tactical(0, 2, 6)


def test_the_missing_statement_is_the_first_that_applies():
    assert RL.kind_of(False, "CONFIRMED", True) == "absent"
    assert RL.kind_of(True, "UNREAD", True) == "unconfirmed"
    assert RL.kind_of(True, "CONFIRMED", False) == "unheld"
    assert RL.kind_of(True, "CONFIRMED", True) == "all present"
    assert set(RL.KINDS) == {"absent", "unconfirmed", "unheld", "all present"}


def test_the_classes_needed_are_only_the_picked_moves_the_labels_do_not_hold():
    labels = {"1_10": rec()}
    reads = [{"id": "1_10"}]
    picks = {"1_10": {"r1": {"first": "a", "pattern": "z", "prior": "b"}, "on": {"first": "y", "pattern": "z", "prior": "a"}}}
    assert sorted(RL.needed_classes(reads, picks, labels)) == [("1_10", "y"), ("1_10", "z")]
    assert RL.needed_classes(reads, {}, labels) == []


def test_the_groups_cover_find_and_confirm_apart_and_together():
    reads = [{"part": p, "band": b, "cls": c} for p in ("find", "confirm") for b in RM.BAND_NAMES for c in (1, 0) for _ in range(2)]
    g = dict(RM.group_masks(reads, {}))
    assert int(g["both all"].sum()) == len(reads) == int(g["find all"].sum()) + int(g["confirm all"].sum())
    assert int(g["both mid class neither"].sum()) == 4 and int(g["find many class winning"].sum()) == 2


def test_a_share_of_moves_has_a_band_over_games():
    vals = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0] * 20)
    game = np.repeat(np.arange(12), 10)
    v, lo, hi = RM.ratio(vals, game)
    assert lo <= v <= hi and abs(v - vals.mean()) < 1e-9


def test_a_move_the_labels_hold_is_read_from_them_and_another_from_the_asked_classes():
    labels = {"1_10": rec()}
    assert RM.class_of_move("1_10", "b", labels, {}) == 0
    assert RM.move_kept("1_10", "a", labels, {}) is True and RM.move_kept("1_10", "b", labels, {}) is False
    assert RM.class_of_move("1_10", "q", labels, {}) is None
    assert RM.move_kept("1_10", "q", labels, {RL.key_of("1_10", "q"): 1}) is True
