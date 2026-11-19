"""PLAN 78.99 THE JUDGE'S RECORD OF WORTHS, MEASURED IN THE ROOM AND IN PLAY (SYSTEM_SPEC 4 the tactical reasoner's equal reading, 5 the value and its measured records, 11 the layers are game-free): the class of a move and its absence, the leaving counts, the rule for the
judged room positions, the players (the record is a parameter; the default record is untouched; the early stop waits for sixty opening pairs) and the wall (the match player never imports the oracle wrapper). The layer files name no game object."""
import re
import subprocess
import sys
from pathlib import Path

from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import record_measure as RM
from catspace.incubator.analyst.discovery import record_play as RP
from catspace.incubator.analyst.discovery import record_reads as RDS
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import tiebreak_measure as TM

DIR = Path(RP.__file__).parent
LAYER = ("record_reads.py", "record_measure.py", "record_play.py")
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
    text = (DIR / "record_play.py").read_text()
    assert not re.search(r"^\s*(from|import) .*(ladder_oracle|prior_equal|future_labels)", text, flags=re.M)
    code = "import sys; import catspace.incubator.analyst.discovery.record_play as P; print('LOADED' if any(m.endswith('ladder_oracle') for m in sys.modules) else 'CLEAN')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.stdout.strip().endswith("CLEAN"), (out.stdout, out.stderr[-500:])


# ---- the class of a move, what leaves

REC = {"id": "1_20", "ord": 1, "cls": 0, "tied": ["m1", "m2", "m3"], "cm": [0, -1, 1], "first": "m1", "best": "m9", "best_cls": 1}


def test_a_move_has_the_class_the_labels_read_or_none():
    assert RDS.class_of(REC, "m2") == -1
    assert RDS.class_of(REC, "m9") == 1                  # the oracle's own move
    assert RDS.class_of(REC, "zz") is None               # not read: no class, never guessed
    assert RDS.class_of(REC, None) is None


def test_a_move_keeps_the_class_when_it_is_not_lower():
    assert RDS.first_keeps(REC, "m1") is True
    assert RDS.first_keeps(REC, "m2") is False
    assert RDS.first_keeps(REC, "m3") is True
    assert RDS.first_keeps(REC, "zz") is None


def test_leaving_is_counted_per_losing_and_keeping_move_and_a_cut_reading_has_no_answer():
    row = {"first": "m3", "equals": ["m1"], "depth": 5, "cut": False, "error": None}
    lv = RDS.leaves(REC, row)
    assert lv == {"lose": [True], "keep": [False, False]}      # m2 is gone; m1 (a first move or held) and m3 (the first move) stay
    assert RDS.leaves(REC, dict(row, cut=True)) is None
    assert RDS.leaves(REC, dict(row, error="x")) is None


def reads_of(first_by_record, room=True, rid="1_20", band="mid"):
    rows = {n: {"first": f, "equals": [], "depth": 5, "cut": False, "error": None} for n, f in first_by_record.items()}
    return {"id": rid, "ord": 1, "part": "find", "band": band, "room": room, "rows": rows}


def test_the_room_counts_first_moves_with_and_without_a_class():
    reads = [reads_of({"gen2": "m1", "human_78_59": "m2"}), reads_of({"gen2": "m1", "human_78_59": "zz"}, rid="2_20")]
    labels = {"1_20": REC, "2_20": dict(REC, id="2_20")}
    c = RM.room_counts(reads, labels, "human_78_59")
    assert c["noclass"] == 1                                   # counted apart
    assert [x[3] for x in c["first"]] == [False]               # the one with a class loses it
    assert [x[3] for x in c["differs"]] == [True, True]


def test_the_price_outside_the_room_is_read_where_the_odds_first_move_keeps_the_class():
    reads = [reads_of({"gen2": "m1", "human_78_59": "m2"}, room=False), reads_of({"gen2": "m2", "human_78_59": "m1"}, room=False, rid="2_20"),
             reads_of({"gen2": "m1", "human_78_59": "zz"}, room=False, rid="3_20")]
    labels = {"1_20": REC, "2_20": dict(REC, id="2_20"), "3_20": dict(REC, id="3_20")}
    c = RM.price_counts(reads, labels, "human_78_59")
    assert c["base"] == 2                                      # the second position's odds first move loses the class: not in the base
    assert sum(x[3] for x in c["loses"]) == 1 and sum(x[3] for x in c["none"]) == 1 and sum(x[3] for x in c["keeps"]) == 0


def test_the_ten_room_positions_are_the_first_where_the_human_first_move_differs():
    reads = [reads_of({"gen2": "a", "human_78_59": "a"}), reads_of({"gen2": "a", "human_78_59": "b"}, rid="2_20"), reads_of({"gen2": "a", "human_78_59": "b"}, room=False, rid="3_20"),
             reads_of({"gen2": "a", "human_78_59": "c"}, rid="4_20")]
    assert [r["id"] for r in RM.room_cases(reads, 10)] == ["2_20", "4_20"]
    assert [r["id"] for r in RM.room_cases(reads, 1)] == ["2_20"]


# ---- the players

def test_the_three_records_are_read_as_stored_and_the_default_is_untouched():
    odds = RP.worth_of("odds")
    assert odds == dict(OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")["worth"])      # the judge of every match since 78.61, as it was
    assert odds == dict(TC.Record.load(None).worth)
    for n in ("selfplay", "human"):
        w = RP.worth_of(n)
        assert w and w != odds
    assert RP.worth_of("human") == dict(TC.Record.load("human_78_59").worth)
    assert TM.WORTH_GEN == 3                                   # the default of the other matches did not move


def test_a_player_carries_the_record_as_a_parameter_and_the_pick_changes_only_the_kind():
    p0, p1 = RP.player_spec("human", False), RP.player_spec("human", True)
    assert p0["kind"] == TM.KIND["control"] and p1["kind"] != p0["kind"]
    assert p0["worth"] == p1["worth"] == RP.worth_of("human")


def test_a_match_has_both_sets_colours_alternated_and_the_record_of_each_side():
    sp = RP.match_specs("human", "odds", False, "A", 3, 0.2, Path("x"))
    assert len(sp) == 6 and [s["colour"] for s in sp] == [0, 1] * 3
    s0, s1 = sp[0], sp[1]
    assert s0["players"][0]["worth"] == RP.worth_of("human") and s0["players"][1]["worth"] == RP.worth_of("odds")
    assert s1["players"][0]["worth"] == RP.worth_of("odds") and s1["players"][1]["worth"] == RP.worth_of("human")
    assert s0["start"] == s1["start"] and s0["opening"] == s1["opening"]
    assert s0["seconds"] == 0.2 and s0["max_plies"] == TM.PLY_CAP
    assert {tuple(s["opening"]) for s in RP.match_specs("human", "odds", False, "B", 3, 0.2, Path("x"))}.isdisjoint({tuple(s["opening"]) for s in sp})


def test_the_early_stop_is_not_asked_before_sixty_whole_pairs():
    games = [{"start": f"s{i}", "colour": c, "outcome": 100 if c == 0 else 0} for i in range(40) for c in (0, 1)]       # forty pairs, all won
    assert TM.settled(games, min_pairs=30) == "clear"                 # a stop after thirty pairs would read high
    assert RP.stop_of()(games) is None
    more = [{"start": f"s{i}", "colour": c, "outcome": 100 if c == 0 else 0} for i in range(64) for c in (0, 1)]
    assert RP.stop_of()(more) == "clear"
