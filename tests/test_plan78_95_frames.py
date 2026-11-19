"""PLAN 78.95 WHAT PRECEDES A WIN, COUNTED FAIRLY; ONE THREAT AND ITS CHANCE; THE TWO PICKS TOGETHER IN PLAY (SYSTEM_SPEC 7 the future imagination layer, 9 the oracle in sleep and the wall, 11 the layers are game-free): the complete reply test of a goal, the stable points, the kinds
of frame and their stake, the kind targets, the weighted chooser, and the wall (the match player never imports the oracle wrapper). The layer files name no game object."""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from catspace.incubator.analyst.discovery import future_choose as FC
from catspace.incubator.analyst.discovery import future_frames as FF
from catspace.incubator.analyst.discovery import future_goals as FG
from catspace.incubator.analyst.discovery import future_judge as FJ
from catspace.incubator.analyst.discovery import future_labels as FL
from catspace.incubator.analyst.discovery import future_measure as FM
from catspace.incubator.analyst.discovery import threat_concrete as TC

DIR = Path(FG.__file__).parent
LAYER = ("future_goals.py", "future_labels.py", "future_judge.py", "future_choose.py", "future_measure.py", "future_frames.py", "future_play.py")
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
    """THE WALL (SYSTEM_SPEC 9): the match player's module and everything it loads at run time leave the oracle wrapper out; the judge it loads was fitted on stored labels."""
    for f in ("future_play.py", "future_choose.py", "future_judge.py"):
        text = (DIR / f).read_text()
        assert not re.search(r"^\s*(from|import) .*ladder_oracle", text, flags=re.M), f
        assert "analyse(" not in text, f
    code = "import sys; import catspace.incubator.analyst.discovery.future_play as P; print('LOADED' if any(m.endswith('ladder_oracle') for m in sys.modules) else 'CLEAN')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.stdout.strip().endswith("CLEAN"), (out.stdout, out.stderr[-500:])


# ---- the complete reply test

REC = TC.Record({"p": 1, "q": 3}, {}, "test")


def row(place, kind, net, replies=(), first=None):
    return {"goal": False, "verdict": "CONFIRMED", "two_moves": False, "place_ix": place, "kind": kind, "net": net, "first": first or (9, place), "response": [{"first": r} for r in replies]}


def test_a_pair_the_kept_list_passes_falls_when_one_legal_reply_meets_both():
    rows = [row(1, "p", 5, [(0, 1)]), row(2, "q", 7, [(0, 2)])]
    c = {"mover": [], "other": rows}
    assert len(FG.goals_of_list(c, "mover", REC)["goals"]) == 1                          # the kept lists share no reply: a goal
    sets = {0: frozenset({"m1", "m9"}), 1: frozenset({"m2", "m9"})}                       # but the move m9 (not kept) meets both
    d = FG.complete_of_list(c, "mover", REC, sets, 20, rows)
    assert d["goals"] == [] and d["kept_pairs"] == 1 and d["fell"] == 1 and d["new"] == 0 and d["replies"] == 20
    assert d["fallen"][0]["common"] == ["m9"] and d["fallen"][0]["n_common"] == 1
    sets = {0: frozenset({"m1"}), 1: frozenset({"m2"})}
    d = FG.complete_of_list(c, "mover", REC, sets, 20, rows)
    assert len(d["goals"]) == 1 and d["goals"][0]["kind"] == "fork" and d["fell"] == 0 and d["goals"][0]["stake"] == 5
    sets = {0: frozenset(), 1: frozenset({"m2"})}
    d = FG.complete_of_list(c, "mover", REC, sets, 20, rows)
    assert d["goals"][0]["kind"] == "open"                                                # a threat that no reply meets, beside another


def test_a_goal_the_kept_test_missed_is_counted_new_and_the_same_unit_is_never_a_pair():
    rows = [row(1, "p", 5, [(0, 1), (3, 3)]), row(2, "q", 7, [(3, 3)])]                   # the kept lists share (3, 3): no goal by them
    c = {"mover": [], "other": rows}
    assert FG.goals_of_list(c, "mover", REC)["goals"] == []
    d = FG.complete_of_list(c, "mover", REC, {0: frozenset({"a"}), 1: frozenset({"b"})}, 5, rows)
    assert len(d["goals"]) == 1 and d["new"] == 1 and d["kept_pairs"] == 0
    same = [row(1, "p", 5), row(1, "q", 7)]
    assert FG.complete_of_list({"mover": [], "other": same}, "mover", REC, {0: frozenset(), 1: frozenset()}, 5, same)["goals"] == []


def test_move_cells_reads_the_two_place_names_of_a_move_word():
    places = [f"x{i}" for i in range(10)]
    assert FG.move_cells("x3x7", places) == (3, 7)
    assert FG.move_cells("x3x7z", places) == (3, 7)                                       # a suffix is ignored
    assert FG.move_cells("zzx7", places) is None and FG.move_cells("x3", places) is None


def test_every_legal_reply_is_read_and_a_place_no_one_threatens_is_met_by_all():
    from catspace.incubator.analyst.discovery import human_heads_data as HD
    from catspace.incubator.analyst.discovery import threat_list as TL
    import cs_engine as CE
    lab = FL.Labeller()
    moves = ["d2d4", "g8f6", "c2c4", "e7e6"]
    it = lab._items([moves], [len(moves)])[0]
    ctx = FG.Complete(HD.VARIANT, lab.reader, lab.lines, lab.record)
    legal = list(CE.state(HD.VARIANT, None, moves)[2])
    sets, n = ctx.reply_sets(it, "other", [{"place_ix": 0}])                              # the owner (the side to move) tries every legal move; nothing is taken at that place
    assert n == len(legal) and sets[0] == frozenset(legal)
    p = CE.passed(HD.VARIANT, CE.state(HD.VARIANT, None, moves)[0], [])
    sets2, n2 = ctx.reply_sets(it, "mover", [{"place_ix": 0}])
    assert n2 == len(CE.state(HD.VARIANT, p[0], [])[2]) and len(sets2[0]) == n2


# ---- the stable points

def rec_with(classes: list, plies: "list | None" = None) -> dict:
    plies = plies or [10 + 2 * i for i in range(len(classes))]
    return {"ord": 1, "part": "find", "y": 1, "n_moves": 100, "positions": [{"ply": p, "cls": c} for p, c in zip(plies, classes)]}


def test_a_winning_point_is_stable_only_when_the_class_holds_two_plies_on():
    fl = FL.stable_points(rec_with([1, 1, 0, 1, -1, -1, 0]))
    assert [f["stable_win"] for f in fl] == [True, False, False, False, False, False, False]
    assert [f["stable_lose"] for f in fl] == [False, False, False, False, True, False, False]
    assert [f["win"] for f in fl][:2] == [True, True]
    gap = FL.stable_points(rec_with([1, 1], plies=[10, 14]))                                # the next sampled position is not two plies on: never stable
    assert not gap[0]["stable_win"]


# ---- the kinds of frame

def step(goal=False, forced=False, threats=0, n_v=10):
    return {"goal": goal, "forced": forced, "threats": threats, "n_v": n_v}


def test_the_kind_of_a_frame_follows_the_order_goal_end_threat_taking_nothing():
    assert FF.frame_kind({"1": step(goal=True, forced=True, threats=2)}, 10) == "two"
    assert FF.frame_kind({"1": step(forced=True, threats=1)}, 10) == "end"
    assert FF.frame_kind({"1": step(threats=1, n_v=9)}, 10) == "one"
    assert FF.frame_kind({"1": step(n_v=10), "3": step(n_v=9)}, 10) == "take"
    assert FF.frame_kind({"1": step(), "3": step()}, 10) == "none"
    assert FF.frame_kind({}, 10) == "none"


def test_the_stake_is_the_victims_loss_less_the_owners_in_the_records_worths():
    worth = {0: 1, 1: 3, 2: 9}.get
    root = [[0, 0, 5], [0, 1, 6], [1, 0, 7], [1, 2, 8]]                                      # owner 0: kinds 0 and 1; victim 1: kinds 0 and 2
    last = [[0, 0, 5], [1, 0, 7]]                                                              # owner 0 lost kind 1 (3); victim lost kind 2 (9)
    assert FF.material_net(root, last, 0, 1, worth) == 9 - 3
    assert FF.material_net(root, root, 0, 1, worth) == 0
    gained = root + [[0, 2, 1]]                                                                # a unit that appears is no loss to anyone
    assert FF.material_net(root, gained, 0, 1, worth) == 0


# ---- the kind targets and the heads

def labelled(ords_plies: list) -> list:
    out = []
    for o, plies in ords_plies:
        out.append({"ord": o, "part": "find", "y": 0, "n_moves": 40, "positions": [{"ply": p, "units": 20, "n_us": 10, "mover": p % 2, "cls": 0, "mover_g": {"goal": False}, "other_g": {"goal": False}} for p in plies]})
    return out


def test_a_kind_target_looks_forward_for_a_stable_winning_point_of_that_kind_of_the_same_side():
    recs = labelled([(7, [10, 12, 14, 16, 36])])
    frames = {(7, 14): {"kind": "one", "owner": "x", "stable_win": True, "stable_lose": False},
              (7, 16): {"kind": "take", "owner": "y", "stable_win": False, "stable_lose": True},
              (7, 12): {"kind": "two", "owner": "x", "stable_win": False, "stable_lose": False}}                  # not stable: no target
    rows = FJ.rows_from_labels(recs, (2, 4), frames, ("take", "one", "two"))
    side = lambda s: [i for i in range(len(rows)) if rows["mover_side"][i] == s]
    mover = side(1)
    other = side(0)
    ply = rows["ply"]
    at = lambda idx, p: [i for i in idx if ply[i] == p][0]
    assert rows["Fone4"][at(mover, 10)] == 1 and rows["Fone2"][at(mover, 10)] == 0          # the point at ply 14 is within 4 plies of ply 10, not within 2
    assert rows["Fone4"][at(other, 10)] == 0                                                 # a point of the mover's is not the other side's
    assert rows["Ftake4"][at(other, 12)] == 1 and rows["Ftake4"][at(mover, 12)] == 0         # the other side's stable point at ply 16
    assert rows["Ftwo4"][at(mover, 10)] == 0                                                 # a frame that is not at a stable point is no target
    assert rows["now_Fone"][at(mover, 14)] == 1 and rows["now_Fone"][at(mover, 10)] == 0
    assert rows["Fone4"][at(mover, 36)] == -1                                                # the window leaves the game


def test_a_kind_head_is_fitted_like_any_head_on_find_and_asked_by_its_kind():
    rng = np.random.default_rng(1)
    n = 3000
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = (X[:, 0] > 0.8).astype(int)
    cols = {"game": np.repeat(np.arange(60), n // 60), "part": np.zeros(n, int), "Ftake4": y, "now_Ftake": np.zeros(n, int)}
    j = FJ.FutureJudge()
    j.fit(FJ.Rows(cols), X, targets=["Ftake4"])
    assert j.info["Ftake4"]["fitted"]
    p = j.chance(X, 4, "frame:take")
    assert FJ.auc(p, y) > 0.9
    with pytest.raises(ValueError):
        bad = dict(cols, part=np.ones(n, int))
        FJ.FutureJudge().fit(FJ.Rows(bad), X, targets=["Ftake4"])


# ---- the weighted chooser

class StubJudge:
    """A judge whose chance for a kind is a planted column of the side matrix."""

    def __init__(self, cols: dict):
        self.cols = cols

    def head(self, name: str):
        return True if name[1:-1] in self.cols or name[1:-2] in self.cols else None

    def chance(self, X, k, what):
        kd = what[6:]
        return X[:, self.cols[kd]]


def test_the_weighted_form_sums_the_kind_heads_by_their_weights_and_ours_less_theirs_orders_the_moves():
    judge = StubJudge({"take": 0, "one": 1})
    ch = FC.Chooser(judge, ("frames", 4), None, 1, {"take": 10.0, "one": 1.0})
    X = np.array([[0.5, 0.2], [0.0, 1.0]])
    assert np.allclose(ch.chance_of(X), [10 * 0.5 + 0.2, 1.0])
    assert np.allclose(FC.Chooser(judge, ("frames", 4), None, 1, {"take": 0.0, "one": 1.0}).chance_of(X), [0.2, 1.0])         # a zero weight leaves the head out
    ours = {"a": (1.0, 0.2), "b": (1.0, 0.2), "c": (0.9, 0.0)}
    assert ch.pick_of(["a", "b", "c"], ours) == "c" and ch.pick_of(["a", "b"], ours) is None                                  # equal scores: silent; the higher ours less theirs wins


def test_picks_from_scores_prefers_the_first_move_when_it_is_among_the_best_and_is_silent_on_a_tie():
    rows = [{"tied": ["a", "b", "c"]}, {"tied": ["a", "b"]}, {"tied": ["a", "b", "c"]}]
    got = FM.picks_from_scores(rows, np.array([0.1, 0.5, 0.2, 0.3, 0.3, 0.0, 0.7, 0.7]))
    assert got == [("b", False), ("a", True), ("b", False)]                                  # (the third: the first move is not among the best, so the first of the best)
    again = FM.picks_from_scores([{"tied": ["a", "b"]}], np.array([0.2, 0.2 + 1e-9]))
    assert again == [("a", True)]                                                            # a tie within six places is no preference


def test_the_chances_table_and_the_weights_are_made_from_find_games_only():
    text = (DIR / "future_measure.py").read_text()
    body = text.split("def cmd_chances")[1].split("def ")[0]
    assert 'r["part"] != "find"' in body
