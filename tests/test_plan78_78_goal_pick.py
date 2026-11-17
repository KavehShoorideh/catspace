"""Plan 78.78: a goal as a hint among the moves the search holds equal (catspace/incubator/analyst/discovery/subgoal_pick.py; the measure subgoal_pick_measure.py). SYSTEM_SPEC 7 (the hint orders, never adds to the value) and 9 (scored
by the surprises resolved less created). The pick stays inside the tie set; ties go to the search's first move; a built position where one tied move lowers a goal's count is picked by it; the picks read no answers; the wall (the
answers come in only at the score and the ceiling); the files name no chess object and the game's renderer is in the printing script only."""
from __future__ import annotations

import inspect
import random
import re
import sys
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import deep_heads as DH
from catspace.incubator.analyst.discovery import subgoal_goals as SG
from catspace.incubator.analyst.discovery import subgoal_pick as SP
from catspace.incubator.analyst.discovery import subgoal_pick_measure as SM

ROOT = Path(__file__).resolve().parents[1]
DISC = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
ROLES = ["white", "black"]
T = ("black", "kind_t", "t1")


# ------------------------------------------------------------------------------------------------------------------------------ a small world made of data (no engine)

def pf(units: dict, bear: dict, shut=None):
    at = {p: u for u, (_s, _k, p) in units.items()}
    return SG.PositionFacts("t", "white", dict(units), at, {u: set(v) for u, v in bear.items()}, {}, dict(shut or {}))


class World:
    """A `Position` stand-in: the graph now and the graph after each move, given."""

    def __init__(self, now, after: dict, moves=(), mover: int = 0):
        self.now, self._after, self.roles, self.mover, self.moves = now, after, ROLES, mover, list(moves)

    def after(self, move):
        return self._after.get(move, self.now)


def bring_goal(unit_place="s1", route=("s1", "s2"), moves=1, line=()):
    """A goal whose only unmet fact is `bears`, mended by bringing a unit of ours (one move): layer one's count is 1."""
    f = {"name": "bears", "vars": {"T": T}, "status": "unmet", "cause": {"type": "no_bearer", "stoppers": [], "victim": None, "words": ""}, "fix": None}
    return {"target": T, "taker": "white", "owner": "black", "facts": [f], "line": list(line), "stake": 0.1, "source": "no_bearer_no_move", "edits": [], "key": "k", "unmet": ["bears"], "verdict": None,
            "steps": [{"step": "bring", "unit": ("white", "kind_a", unit_place), "route": list(route), "moves": moves}], "own_moves": 1}


UNITS = {"ua": ("white", "kind_a", "s1"), "ub": ("black", "kind_t", "t1")}
NOW = pf(UNITS, {})
AFTER = {"m2": pf({"ua": ("white", "kind_a", "s2"), "ub": ("black", "kind_t", "t1")}, {"ua": ["t1"]})}


def table_of(counts_before, counts_after):
    return {"before": [{"own": a, "w": a if a is not None else 3} for a in counts_before],
            "after": {m: [{"own": a, "w": a if a is not None else 3} for a in cs] for m, cs in counts_after.items()}}


# ------------------------------------------------------------------------------------------------------------------------------ the pick

def test_a_built_position_where_one_tied_move_lowers_a_goal_s_count_is_picked_by_it():
    g = bring_goal()
    w = World(NOW, AFTER)
    tied = ["m1", "m2", "m3"]
    t = SP.count_table(w, tied, [g])
    assert t["before"][0]["own"] == 1 and t["after"]["m2"][0]["own"] == 0                     # layer one, asked again on the position after the move
    assert t["after"]["m1"][0]["own"] == 1 and t["after"]["m3"][0]["own"] == 1
    for mode in ("top1", "top2", "top3", "all"):
        assert SP.goal_pick(w, tied, [g], t, mode) == "m2"
        assert SP.goal_pick(w, tied, [g], None, mode) == "m2"                                  # the table is made here when none is handed in
    assert SP.goal_pick(w, tied, [g], t, "all", scale="w") == "m2"


def test_a_step_of_two_moves_is_one_move_shorter_after_a_move_of_the_unit_and_a_far_unit_changes_nothing():
    g = bring_goal(moves=2, route=["s1", "s3", "s2"])
    g["own_moves"] = 2
    after = {"m2": pf({"ua": ("white", "kind_a", "s3"), "ub": ("black", "kind_t", "t1")}, {})}      # the unit left the first place of its route and stands on the next: not yet bearing
    w = World(NOW, after)
    t = SP.count_table(w, ["m1", "m2"], [g])
    assert t["before"][0]["own"] == 2 and t["after"]["m2"][0]["own"] == 1 and t["after"]["m1"][0]["own"] == 2
    assert SP.goal_pick(w, ["m1", "m2"], [g], t, "top1") == "m2"


def test_a_goal_with_no_count_is_one_more_than_the_limit_so_a_move_that_makes_it_reachable_lowers_it():
    t = table_of([None], {"m1": [None], "m2": [2]})
    assert SP.drop(t, "m2", "own", None) == SP.NONE_COUNT - 2 > 0 and SP.drop(t, "m1", "own", None) == 0


def test_ties_go_to_the_search_s_first_move_and_a_silent_hint_returns_it():
    tied = ["m1", "m2", "m3"]
    silent = table_of([1, 2], {m: [1, 2] for m in tied})
    for mode in ("top1", "top2", "top3", "all"):
        assert SP.goal_pick(None, tied, [], silent, mode) == "m1"
    same = table_of([2], {"m1": [2], "m2": [1], "m3": [1]})                                      # two moves lower it by the same: the earlier of the two
    assert SP.goal_pick(None, tied, [], same, "top1") == "m2"
    worse = table_of([2], {"m1": [3], "m2": [3], "m3": [3]})                                     # every move raises the count: nothing lowers, the search's own move
    assert SP.goal_pick(None, tied, [], worse, "top1") == "m1"


def test_the_ranks_of_the_modes_read_the_first_k_goals_only():
    tied = ["m1", "m2"]
    t = table_of([2, 2, 2], {"m1": [2, 2, 2], "m2": [2, 2, 1]})                                  # only the third goal falls
    assert SP.goal_pick(None, tied, [], t, "top1") == "m1" and SP.goal_pick(None, tied, [], t, "top2") == "m1"
    assert SP.goal_pick(None, tied, [], t, "top3") == "m2" and SP.goal_pick(None, tied, [], t, "all") == "m2"


def test_the_pick_never_leaves_the_tie_set_for_any_table():
    rng = random.Random(3)
    for _ in range(200):
        tied = [f"m{k}" for k in range(rng.randint(1, 6))] or ["m0"]
        n = rng.randint(0, 5)
        before = [rng.choice([None, 0, 1, 2]) for _ in range(n)]
        t = table_of(before, {m: [rng.choice([None, 0, 1, 2]) for _ in range(n)] for m in tied})
        for mode in ("top1", "top2", "top3", "all"):
            for scale in ("own", "w"):
                assert SP.goal_pick(None, tied, [], t, mode, scale=scale) in tied


def test_the_rule_modes_hand_the_rule_the_tied_moves_and_return_one_of_them():
    g = bring_goal(line=["m2", "x1x2"])
    w = World(NOW, AFTER)
    lists = {"mover": [], "other": [], "exchange": [], "goal": [], "dropped_rows": []}
    for mode in ("rule", "goals"):
        assert SP.goal_pick(w, ["m1", "m2"], [g], None, mode, lists) == "m2"
    with pytest.raises(AssertionError):
        SP.goal_pick(w, ["m1", "m2"], [g], None, "rule", None)                                  # the rule reads the threat list


# ------------------------------------------------------------------------------------------------------------------------------ the preparation, the score and the ceiling on built data

class FakeTools:
    def __init__(self, eq, move):
        self.eq, self.mv = eq, move

    def search(self, position, seconds):
        return {"move": self.mv, "equals": list(self.eq)}

    def forced(self, position):
        return {"first": None, "kind": None}


class FakeReader:
    def __init__(self, goals):
        self.goals = goals

    def read(self, moves, steps):
        lists = {"mover": [], "other": [], "exchange": [], "goal": [], "dropped_rows": []}
        return {"lists": lists, "goals": self.goals, "position": World(NOW, AFTER, moves)}


def test_the_preparation_reads_no_answer_and_its_picks_are_tied_moves(monkeypatch):
    real = Path.read_text

    def no_answers(self, *a, **k):
        assert "answers" not in str(self), f"the preparation read {self}"
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", no_answers)
    prep = SM.prepare_position({"id": "p", "moves": ["a", "b"]}, FakeTools(["m1", "m2", "m3"], "m1"), FakeReader([bring_goal(line=["m2", "x1x2"])]))
    assert prep["read"] == "ok" and prep["tied"] == ["m1", "m2", "m3"] and prep["own_mismatch"] == 0
    assert prep["pick_rule"] in prep["tied"] and prep["pick_goals"] in prep["tied"]
    assert not any(re.search(r"oracle|accepted|answer", k, re.I) for k in prep)
    cands = SM.candidates()
    names = [c["name"] for c in cands]
    assert len(names) == len(set(names)) and sum(1 for c in cands if c.get("random")) == 1
    for c in cands:
        if not c.get("random"):
            assert c["pick"]("p", prep) in prep["tied"]
    no_tie = SM.prepare_position({"id": "q", "moves": ["a"]}, FakeTools(["m1"], "m1"), FakeReader([]))
    assert no_tie["read"] is None and all(c["pick"]("q", no_tie) == "m1" for c in cands if not c.get("random"))


def test_the_ceiling_counts_accepted_and_not_accepted_tied_moves_and_the_silence_counts_the_unchanged_positions():
    tied = ["m1", "m2", "m3"]
    t = table_of([1], {"m1": [1], "m2": [0], "m3": [1]})
    preps = {"own:g:1": {"eq": tied, "tied": tied, "read": "ok", "table": t, "goals": [{"source": "x"}], "own_mismatch": 0, "branch_rule": "4"}}
    pid = "own:g:1"
    ans = {pid: ["m2"]}
    c = SM.ceiling({pid: preps[pid]}, [pid], ans, DH.split_part(pid))
    r = c["rows"][("own", "any goal")]
    assert r["pos_acc"] == 1 and r["pairs_acc"] == [1, 1] and r["pairs_non"] == [0, 2]
    s = SM.silence(preps, [pid])
    assert s["silent_own"] == 0 and s["changing_own"] == [1] and s["goals"] == [1]
    flat = dict(preps[pid], table=table_of([1], {m: [1] for m in tied}))
    assert SM.silence({pid: flat}, [pid])["silent_any"] == 1


# ------------------------------------------------------------------------------------------------------------------------------ the wall and the scan

def test_the_wall_no_pick_reads_the_answers_and_they_come_in_only_where_the_score_and_the_ceiling_take_them():
    for fn in (SP.goal_pick, SP.count_table, SP.own_count, SP.drop, SP.goal_drop, SP.changes, SM.prepare_position, SM.candidates, SM._pick_of, SM._cost_of, SM.goal_info, SM.silence):
        src = inspect.getsource(fn)
        assert not re.search(r"answers|accepted|oracle|ladder_oracle", src, re.I), fn.__name__
    assert "answers" in inspect.signature(SM.ceiling).parameters and "answers" in inspect.signature(DH.score).parameters
    assert "ladder_oracle" not in sys.modules
    assert not re.search(r"answers|accepted|oracle", (DISC / "subgoal_pick.py").read_text(), re.I)


@pytest.mark.parametrize("name", ["subgoal_pick.py", "subgoal_pick_measure.py"])
def test_the_layer_and_the_measure_name_no_chess_object(name):
    hit = re.search(CHESS_WORDS, (DISC / name).read_text(), flags=re.I)
    assert hit is None, f"{name}: {hit.group(0)!r}"


def test_the_printing_script_is_the_only_file_that_spells_moves_in_the_game_s_words():
    assert "import chess" in (DISC / "ladder_print.py").read_text() and "pick_cases" in (DISC / "ladder_print.py").read_text()
    for name in ("subgoal_pick.py", "subgoal_pick_measure.py"):
        assert "import chess" not in (DISC / name).read_text()
