"""PLAN 79.00 WHAT THE THREE THINGS ARE WORTH TOGETHER, AND WHAT ROOM IS LEFT (SYSTEM_SPEC 4 the equal reading and the verified choice, 5 the measured records, 7 the pick among equal moves, 9 the oracle in sleep and the wall): the players (record, verification of the choice and pick are
parameters of the spec; the defaults of the engine and of the movers are untouched; a player may carry its own clock), the rule for a further match, the fresh sample, the shares, the cause-(a) line kinds, the rule for the judged positions, and the wall (the match player never loads the
oracle wrapper). The layer files name no game object."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import stack_causes as SC
from catspace.incubator.analyst.discovery import stack_measure as SM
from catspace.incubator.analyst.discovery import stack_play as SPL
from catspace.incubator.analyst.discovery import stack_room as SR
from catspace.incubator.analyst.discovery import tiebreak_measure as TM

DIR = Path(SPL.__file__).parent
LAYER = ("stack_play.py", "stack_room.py", "stack_measure.py", "stack_causes.py")
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


def test_the_match_player_never_imports_the_oracle_wrapper():
    """THE WALL (SYSTEM_SPEC 9): the match player's module and everything it loads at run time leave the oracle wrapper out."""
    text = (DIR / "stack_play.py").read_text()
    assert not re.search(r"^\s*(from|import) .*(ladder_oracle|prior_equal|future_labels|stack_room)", text, flags=re.M)
    code = "import sys; import catspace.incubator.analyst.discovery.stack_play as P; print('LOADED' if any(m.endswith('ladder_oracle') for m in sys.modules) else 'CLEAN')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.stdout.strip().endswith("CLEAN"), (out.stdout, out.stderr[-500:])


# ---- the players: the three things are parameters of the spec, nothing else changes

def test_a_player_is_the_record_the_verification_and_the_pick_and_nothing_else():
    p = SPL.player_spec("human+v+pick")
    assert p["verify_choice"] is True and p["choice_mode"] == "slot" and p["kind"] == "sy_reasons" and p["charge"] is False
    q = SPL.player_spec("human")
    assert q["verify_choice"] is False and q["kind"] == "tb_control" and q["worth"] == p["worth"]
    o = SPL.player_spec("odds")
    assert o["worth"] != p["worth"] and o["verify_choice"] is False
    assert "seconds" not in p and SPL.player_spec("odds@5x")["seconds"] == 5 * TM.SECONDS
    assert SPL.player_spec("human+pick", charge=True)["charge"] is True


def test_the_default_of_a_mover_leaves_the_verification_off():
    m = TM.Mover67({"kind": "tb_control", "worth": {"p": 1}}, 0.2, "s", "t")
    assert m.verify_choice is False and m.choice_kw() == {}                    # the engine's default call is exactly what it was
    v = TM.Mover67({"kind": "tb_control", "worth": {"p": 1}, "verify_choice": True}, 0.2, "s", "t")
    assert v.choice_kw() == {"verify_choice": True, "choice_mode": "slot"}


def test_a_player_may_carry_its_own_clock():
    a = SPL.Mover79({"kind": "tb_control", "worth": {"p": 1}, "seconds": 1.0}, 0.2, "s", "t")
    b = SPL.Mover79({"kind": "tb_control", "worth": {"p": 1}}, 0.2, "s", "t")
    assert a.seconds == 1.0 and b.seconds == 0.2


def test_the_games_of_a_match_alternate_colours_and_carry_each_sides_clock(tmp_path):
    specs = SPL.match_specs("human+v+pick", "odds@5x", "A", 3, TM.SECONDS, tmp_path)
    assert len(specs) == 6
    assert specs[0]["players"][0]["verify_choice"] is True and specs[1]["players"][1]["verify_choice"] is True
    assert specs[0]["players"][1]["seconds"] == specs[1]["players"][0]["seconds"] == 5 * TM.SECONDS
    assert len({s["gid"] for s in specs}) == 6


def test_the_rule_for_a_further_match_is_every_bands_lower_end_above_a_half():
    assert SPL.ahead([(0.62, 0.55, 0.69), (0.60, 0.52, 0.68)]) is True
    assert SPL.ahead([(0.62, 0.55, 0.69), (0.60, 0.49, 0.68)]) is False
    assert SPL.ahead([]) is False


def test_the_verifications_work_is_counted_per_side():
    def g(a, b):
        return {"colour": 0, "stats": [dict({"decisions": 10}, **a), dict({"decisions": 10}, **b)]}
    games = [g({"vc_asked": 8, "vc_held": 6, "vc_replaced": 1, "vc_cut": 1}, {}), g({"vc_asked": 2, "vc_held": 2}, {})]
    r = SPL.replacements(games, 0)
    assert (r["vc_asked"], r["vc_held"], r["vc_replaced"], r["vc_cut"], r["decisions"]) == (10, 8, 1, 1, 20)
    assert SPL.replacements(games, 1)["vc_asked"] == 0


# ---- the fresh sample

def labelled(ord_, part, n, cls=1, units=20):
    return {"ord": ord_, "part": part, "positions": [{"ply": 10 + 2 * i, "units": units, "cls": cls} for i in range(n)]}


def test_the_fresh_sample_takes_confirm_only_caps_a_game_and_puts_the_seen_last():
    recs = [labelled(1, "confirm", 30), labelled(2, "confirm", 30), labelled(3, "find", 30)]
    seen = {f"1_{10 + 2 * i}" for i in range(20)}
    s = SR.sample_fresh(recs, seen, cell=40, cap=25)
    assert all(i["part"] == "confirm" for i in s) and len(s) == 40
    assert max(sum(1 for i in s if i["ord"] == o) for o in (1, 2)) <= 25
    first_seen = [i for i in s if i["seen96"]]
    unseen = [i for i in s if not i["seen96"]]
    assert len(unseen) >= 35 and len(first_seen) <= 5                         # 40 unseen positions exist, the cap of a game leaves a few seen ones in
    assert s == SR.sample_fresh(recs, seen, cell=40, cap=25)                  # the same order every time


def test_the_labelled_record_carries_the_pick_the_state_and_whether_78_96_sampled_it():
    item = {"ord": 3, "part": "confirm", "ply": 12, "units": 20, "band": "many", "cls": 0, "seen96": True}
    search = {"first": "m1", "equals": ["m2", "m3"], "depth": 6, "error": None, "cut": False, "state": 2}
    r = SR.label_record(item, search, [0, -1, 1], "m9", None, [None, None, None], {"search": 0.1, "pick": 0.0, "oracle": 1.0}, "m3")
    assert r["tied"] == ["m1", "m2", "m3"] and r["pick"] == "m3" and r["pick_cls"] == 1 and r["state"] == 2 and r["seen96"] is True and r["calls"] == 4


# ---- the shares

def rec(i, cls, cm, tied, pick=None, best_cls=None, band="mid"):
    return {"id": f"{i}_10", "ord": i, "cls": cls, "cm": cm, "tied": tied, "pick": pick, "best_cls": best_cls, "band": band}


def test_the_shares_count_the_room_the_first_move_and_the_pick():
    recs = [rec(1, 0, [0, -1, 1], ["a", "b", "c"], pick="c", best_cls=1),         # the room; the first keeps; the pick keeps
            rec(2, 0, [-1, 0], ["a", "b"], pick="a", best_cls=0),                 # the room; the first loses; the pick loses
            rec(3, 1, [0, -1], ["a", "b"], pick="a", best_cls=1),                 # no equal move keeps it, the oracle's move does
            rec(4, 0, [0], ["a"], pick="a", best_cls=0)]                          # one equal move, it keeps
    m = SM.metrics(recs, {})
    assert list(m["room"]) == [1, 1, 0, 0] and list(m["nokeep"]) == [0, 0, 1, 0]
    assert list(m["first"]) == [1, 0, 0, 1] and list(m["pick"]) == [1, 0, 0, 1]
    assert list(m["searchs"]) == [0, 0, 1, 0] and list(m["ge2"]) == [1, 1, 1, 0]
    assert list(m["differ_class"]) == [1, 1, 1, 0]


def test_a_record_without_its_own_pick_takes_the_stored_one_or_the_first_move():
    recs = [rec(1, 0, [0, 1], ["a", "b"]), rec(2, 0, [0, 1], ["a", "b"]), rec(3, 0, [0, 1], ["a", "b"])]
    m = SM.metrics(recs, {"1_10": "b", "2_10": "zz"})                          # a pick not in the set is the first move
    assert list(m["pick"]) == [1, 1, 1] and list(m["first"]) == [1, 1, 1]
    m2 = SM.metrics([rec(1, 1, [0, 1], ["a", "b"]), rec(2, 1, [0, 1], ["a", "b"])], {"1_10": "b"})
    assert list(m2["pick"]) == [1, 0]


def test_the_twenty_judged_positions_are_the_first_room_positions_where_the_first_move_loses_the_class():
    recs = [rec(1, 0, [-1, 0], ["a", "b"]), rec(2, 0, [0, 0], ["a", "b"]), rec(3, 0, [0, -1], ["a", "b"]), rec(4, 0, [-1, -1], ["a", "b"]), rec(5, 0, [-1, 1], ["a", "b"])]
    assert [r["ord"] for r in SM.pick_cases(recs)] == [1, 5]
    assert [r["ord"] for r in SM.pick_cases(recs, 1)] == [1]


# ---- part three

def test_the_room_moves_to_trace_are_the_class_losing_equal_moves_with_a_keeper_and_capped_a_position():
    games = {1: {"moves": [f"m{i}" for i in range(40)]}}
    r = {"id": "1_10", "ord": 1, "ply": 10, "cls": 0, "band": "many", "depth": 7, "cm": [0, -1, -1, -1, 1], "tied": ["a", "b", "c", "d", "e"]}
    it = SC.items_of([r], games, cap_moves=2)
    assert [i["stuck"] for i in it] == ["b", "c"] and all(i["keeper"] == "a" and i["depth"] == 7 and len(i["moves"]) == 10 for i in it)
    assert SC.items_of([dict(r, cm=[-1, -1, -1, -1, -1])], games) == []         # no keeper: not the room
    assert SC.items_of([dict(r, depth=None)], games) == []
    assert len(SC.items_of([r], games, cap_moves=3, limit=2)) == 2


def test_the_kind_of_a_cause_a_line():
    w = {"p": 475, "n": 1216, "b": 1301, "r": 1945, "q": 3583}
    assert SC.line_kind({"gave": [], "took": []}, w) == "no taking"
    assert SC.line_kind({"gave": ["p"], "took": []}, w) == "weakest kind for nothing"
    assert SC.line_kind({"gave": ["n"], "took": []}, w) == "other kinds for nothing"
    assert SC.line_kind({"gave": ["r"], "took": ["n"]}, w) == "unlike exchange"
    assert SC.line_kind({"gave": ["r", "p"], "took": ["p", "r"]}, w) == "like exchange"
    assert SC.line_kind({"gave": ["r"], "took": ["n", "p"]}, w) == "other"


def test_the_tracer_judges_by_the_human_record():
    t = SC.StackTracer()
    assert t.worth == SPL.RP.worth_of("human") and t.worth != SPL.RP.worth_of("odds")


def test_the_stacks_shares_are_numbers_with_a_band():
    v = SM.PMS.share(np.array([1.0, 0.0, 1.0, 1.0]), np.array([1, 1, 2, 3]))
    assert 0.0 <= v[1] <= v[0] <= v[2] <= 1.0
