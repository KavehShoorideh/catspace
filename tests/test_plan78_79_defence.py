"""Plan 78.79: the other side's goals as ours to stop, the long count of own moves, the stoppers with a reply read (catspace/incubator/analyst/discovery/subgoal_pick.py; the measure subgoal_pick_measure.py; the reader
move_rule_play.RuleReader.read(side=...)). SYSTEM_SPEC 7 (reachable has two layers; the hint) and 9 (the score). The proposer for the other side equals the proposer for the mover on the position with the sides exchanged; a built
position where one tied move adds a stopper to their goal; a built walk of four own moves counted as four and one nearer after the first; the replied filter drops a move that loses its unit; picks stay in the tie set; no pick
source reads answers; the files name no chess object."""
from __future__ import annotations

import inspect
import random
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import move_rule as MR
from catspace.incubator.analyst.discovery import subgoal_goals as SG
from catspace.incubator.analyst.discovery import subgoal_pick as SP
from catspace.incubator.analyst.discovery import subgoal_pick_measure as SM

ROOT = Path(__file__).resolve().parents[1]
DISC = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
ROLES = ["white", "black"]
T = ("black", "kind_t", "t1")


# ------------------------------------------------------------------------------------------------------------------------------ the proposer for the other side (the engine; skipped where its data is absent)

def test_the_proposer_for_the_other_side_is_the_proposer_for_the_mover_on_the_position_with_the_sides_exchanged():
    import cs_engine as CE
    from catspace.incubator.analyst.discovery import human_heads_data as HD
    from catspace.incubator.analyst.discovery import move_rule_play as MP
    try:
        reader = MP.reader75()
    except FileNotFoundError as e:
        pytest.skip(f"data absent from this checkout: {e}")
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
    other = reader.read(moves, (2,), side="other")
    exch = CE.passed(HD.VARIANT, None, moves)
    assert other is not None and exch is not None
    mine = reader.read([], (2,), start=exch[0], ply=len(moves))                       # the same proposer, the mover of the exchanged position
    assert [MR.goal_words(g) for g in other["goals"]] == [MR.goal_words(g) for g in mine["goals"]]
    assert [g["taker"] for g in other["goals"]] == [g["taker"] for g in mine["goals"]] and other["goals"]
    plain = reader.read(moves, (2,))                                                  # the default call is unchanged: the goals of the side to move
    assert plain["goals"] and plain["goals"][0]["taker"] != other["goals"][0]["taker"]
    assert other["position"].mover != plain["position"].mover
    after = reader.read(moves + [CE.state(HD.VARIANT, None, moves)[2][0]], (2,))      # after one move of ours the plain read is theirs
    assert after is None or after["position"].mover == other["position"].mover


def test_the_reader_s_new_arguments_default_to_the_old_call():
    from catspace.incubator.analyst.discovery import move_rule_play as MP
    sig = inspect.signature(MP.RuleReader.read).parameters
    assert sig["side"].default is None and sig["start"].default is None and sig["ply"].default is None


def test_the_unit_stepped_test_has_a_public_name_and_the_old_one_is_kept():
    assert MR.unit_stepped is MR._unit_stepped


# ------------------------------------------------------------------------------------------------------------------------------ signal 1: the defence, on built goals

def their_goal(target, stoppers=1, source="no_bearer_no_move", steps=()):
    f = {"name": "bears", "vars": {"T": target}, "status": "unmet", "cause": {"type": "x", "stoppers": [("white", "kind_s", f"s{i}") for i in range(stoppers)], "victim": None, "words": ""}, "fix": None}
    return {"target": target, "taker": "white", "owner": "black", "facts": [f], "unmet": ["bears"], "steps": list(steps), "source": source, "stake": 0.1}


class FakeReplay:
    """`facts.replay`: the unit that made a move, from where to where (a dictionary of the built moves)."""

    def __init__(self, moved):
        self.moved = moved

    def replay(self, text, names):
        a, b = self.moved.get(names[0], (None, None))
        return [{"from": a, "to": b, "uid": "u", "took": None}]


class Pos:
    def __init__(self):
        class N:
            text = "t0"
        self.now, self.roles, self.mover = N, ROLES, 0


def test_a_tied_move_that_adds_a_stopper_to_their_goal_is_a_defence_and_one_that_removes_one_is_to_avoid():
    g0 = their_goal(T, stoppers=1)
    other = their_goal(("black", "kind_t", "t2"), stoppers=2)
    tied = ["m1", "m2", "m3", "m4", "m5"]
    after = {"m1": {"goals": [their_goal(T, stoppers=2), other], "works": []},            # a stopper added to the goal on T
             "m2": {"goals": [g0, other], "works": []},                                   # nothing changes
             "m3": {"goals": [other], "works": []},                                       # the goal is gone and does not work: it cannot be built now
             "m4": {"goals": [other], "works": [T]},                                      # their threat works: a move to avoid
             "m5": None}                                                                  # not read
    t = SP.cost_to_theirs(Pos(), tied, [g0, other], FakeReplay({}), after)
    assert SP.defence_gain(t, "m1") == 1 and SP.defends(t, "m1") and not SP.eases(t, "m1")
    assert SP.defence_gain(t, "m2") == 0 and not SP.defends(t, "m2")
    assert t["after"]["m3"][0]["state"] == "gone" and SP.defends(t, "m3")
    assert t["after"]["m4"][0]["state"] == "works" and SP.eases(t, "m4") and SP.defence_gain(t, "m4") == -1
    assert t["after"]["m5"] is None and SP.defence_gain(t, "m5") == 0
    tab = {"before": [], "after": {}, "their": t}
    assert SP.goal_pick(None, tied, [], tab, "defence") == "m1"                          # the move that costs them most; the avoided and the unread ones lose
    tab["their"] = SP.cost_to_theirs(Pos(), tied, [g0, other], FakeReplay({}), {m: after[m] for m in ("m2", "m4")} | {"m1": None, "m3": None, "m5": None})
    assert SP.goal_pick(None, tied, [], tab, "defence") == "m1"                          # nothing is a defence: the search's own move


def test_a_target_that_moved_is_followed_to_its_new_place():
    g0 = their_goal(T)
    moved_T = ("black", "kind_t", "t9")
    after = {"m1": {"goals": [their_goal(moved_T, stoppers=1)], "works": []}}
    t = SP.cost_to_theirs(Pos(), ["m1"], [g0], FakeReplay({"m1": ("t1", "t9")}), after)
    assert t["after"]["m1"][0]["state"] == "kept" and t["after"]["m1"][0]["cost"] == 0
    t2 = SP.cost_to_theirs(Pos(), ["m1"], [g0], FakeReplay({}), after)                    # unfollowed it would look gone
    assert t2["after"]["m1"][0]["state"] == "gone"


# ------------------------------------------------------------------------------------------------------------------------------ signal 2: the long count, on a built walk

class WalkFacts:
    """A line of places p0, p1, ...; one unit of ours walks it one place a move and bears on the target from p4 on. The text of a position says where the unit stands."""

    def of(self, text):
        k = int(text.split("@p")[1])
        units = {"ua": ("white", "kind_a", f"p{k}"), "ub": T}
        at = {p: u for u, (_s, _k, p) in units.items()}
        return SG.PositionFacts(text, "white", units, at, {"ua": {"t1"}} if k >= 4 else {}, {"ua": {f"p{k + 1}"}}, {})


class WalkEditor:
    def relocate(self, text, a, b):
        return f"u@{b}"


class WalkWorld:
    roles, mover = ROLES, 0

    def __init__(self):
        self.f = WalkFacts()
        self.now = self.f.of("u@p0")
        self._after = {"go": self.f.of("u@p1"), "idle": self.f.of("u@p0")}

    def after(self, move):
        return self._after[move]


def walk_goal():
    f = {"name": "bears", "vars": {"T": T}, "status": "unmet", "cause": {"type": "no_bearer", "stoppers": [], "victim": None, "words": ""}, "fix": None}
    return {"target": T, "taker": "white", "owner": "black", "facts": [f], "unmet": ["bears"], "steps": [], "source": "no_bearer_no_move", "stake": 0.1}


def test_a_walk_of_four_own_moves_is_counted_as_four_and_one_nearer_after_the_first():
    w, g = WalkWorld(), walk_goal()
    assert SP.long_count(w, g, w.f, 8, None, WalkEditor()) == 4
    assert SP.long_count(w, g, w.f, 8, "go", WalkEditor()) == 3
    assert SP.long_count(w, g, w.f, 8, "idle", WalkEditor()) == 4
    t = SP.long_table(w, ["idle", "go"], [g], w.f, 8, WalkEditor())
    assert t["before"] == [4] and t["after"] == {"idle": [4], "go": [3]}
    tab = {"before": [], "after": {}, "long": t}
    assert SP.long_drop(tab, "go") == 1 and SP.long_drop(tab, "idle") == 0
    assert SP.goal_pick(None, ["idle", "go"], [g], tab, "long") == "go"
    assert SP.LIMIT == 2 and SP.LONG_LIMIT > SP.LIMIT                                       # layer one still stops at two; the long count does not


def test_the_limit_of_the_search_is_a_cost_guard_that_says_when_it_cut():
    w, g = WalkWorld(), walk_goal()
    assert SP.long_count(w, g, w.f, 3, None, WalkEditor()) is None                         # four moves are needed, three are read
    tab = SP.reach_table(w.now, w.f, WalkEditor(), 3)
    assert tab["cut_depth"] and not tab["cut_fuse"]
    t = SP.long_table(w, ["idle", "go"], [g], w.f, 3, WalkEditor())
    assert t["cut"]["none_cut"] >= 1 and t["cut"]["tables"] == 3 and t["after"]["go"] == [3]
    small = SP.reach_table(WalkFacts().of("u@p0"), w.f, WalkEditor(), 8, 2)                  # the guard of graph reads
    assert small["cut_fuse"]


def test_a_goal_with_another_unmet_fact_has_no_long_count():
    w, g = WalkWorld(), walk_goal()
    g["facts"].append({"name": "kept", "vars": {}, "status": "unmet", "cause": {"type": "x", "stoppers": [], "victim": None, "words": ""}, "fix": None})
    assert SP.long_count(w, g, w.f, 8, None, WalkEditor()) is None
    done = walk_goal()
    done["facts"][0]["status"] = "met"
    assert SP.long_count(w, done, w.f, 8, None, WalkEditor()) == 0


# ------------------------------------------------------------------------------------------------------------------------------ signal 3: the stoppers with a reply read

class LineFacts:
    def __init__(self, plies):
        self.plies = plies

    def replay(self, text, names):
        return self.plies


def line(names=("a", "b", "c"), cut=False, ok=True):
    return {"ok": ok, "names": list(names), "settled": len(names), "cut": cut}


def test_the_replied_filter_drops_a_move_that_loses_its_unit():
    w = WalkWorld()
    kept = [{"uid": "u1", "took": None}, {"uid": "x", "took": ("u9", "white", "kind_o", "z")}]
    lost = [{"uid": "u1", "took": None}, {"uid": "x", "took": ("u1", "white", "kind_a", "z")}]
    their = [{"uid": "u1", "took": None}, {"uid": "x", "took": ("u7", "black", "kind_o", "z")}]       # the line takes a unit of theirs: the unit that moved is kept
    assert SP.line_keeps_unit(w, LineFacts(kept), "a", line()) is True
    assert SP.line_keeps_unit(w, LineFacts(lost), "a", line()) is False
    assert SP.line_keeps_unit(w, LineFacts(their), "a", line()) is True
    assert SP.line_keeps_unit(w, LineFacts(kept), "a", line(cut=True)) is None and SP.line_keeps_unit(w, LineFacts(kept), "a", None) is None
    assert SP.line_keeps_unit(w, LineFacts(kept), "a", line(ok=False)) is None
    table = {"before": [{"own": None, "w": 2}], "after": {"m1": [{"own": None, "w": 1}], "m2": [{"own": None, "w": 0}]}, "replied": {"m1": True, "m2": False}}
    assert SP.replied_drop(table, "m1") == 1 and SP.replied_drop(table, "m2") == 0                  # m2 takes more stoppers away but its line loses the unit: not counted
    assert SP.goal_pick(None, ["m0", "m1", "m2"], [], table, "replied") == "m1"
    assert SP.goal_pick(None, ["m0", "m1", "m2"], [], dict(table, replied={"m1": False, "m2": False}), "replied") == "m0"


def test_the_combinations_order_the_signals_as_the_plan_says():
    tied = ["m0", "m1", "m2"]
    their = {"before": [1], "after": {"m0": [{"cost": 0}], "m1": [{"cost": 1}], "m2": [{"cost": 1}]}}
    longt = {"before": [3], "after": {"m0": [3], "m1": [3], "m2": [2]}, "limit": 4}
    tab = {"before": [{"own": None, "w": 1}], "after": {m: [{"own": None, "w": 1}] for m in tied}, "their": their, "long": longt, "replied": {m: True for m in tied}}
    assert SP.goal_pick(None, tied, [], tab, "defence") == "m1"                            # the first of the moves with the larger defence
    assert SP.goal_pick(None, tied, [], tab, "defence_long") == "m2"                       # the defence first, then the long count between the two
    assert SP.goal_pick(None, tied, [], tab, "sum") == "m2"
    assert SP.goal_pick(None, tied, [], tab, "long") == "m2"


def test_the_pick_never_leaves_the_tie_set_for_any_table():
    rng = random.Random(5)
    for _ in range(300):
        n = rng.randint(1, 6)
        tied = [f"m{k}" for k in range(n)]
        g = rng.randint(0, 4)
        before = [{"own": rng.choice([None, 0, 1, 2]), "w": rng.randint(0, 3)} for _ in range(g)]
        table = {"before": before, "after": {m: [{"own": rng.choice([None, 0, 1, 2]), "w": rng.randint(0, 3)} for _ in range(g)] for m in tied}}
        table["their"] = {"before": [None] * g, "after": {m: (None if rng.random() < 0.2 else [{"cost": rng.choice([-1, 0, 1])} for _ in range(g)]) for m in tied}}
        table["long"] = {"before": [rng.choice([None, 0, 1, 4]) for _ in range(g)], "after": {m: [rng.choice([None, 0, 1, 4]) for _ in range(g)] for m in tied}, "limit": 4}
        table["replied"] = {m: rng.random() < 0.5 for m in tied}
        for mode in SP.NEW_MODES:
            assert SP.goal_pick(None, tied, [], table, mode) in tied


# ------------------------------------------------------------------------------------------------------------------------------ the measure on built data, the wall and the files

def prep_with(accepted_carry: bool):
    tied = ["m1", "m2", "m3"]
    their = {"before": [{}], "after": {"m1": [{"cost": 1 if accepted_carry else 0}], "m2": [{"cost": 0}], "m3": [{"cost": 1}]}}
    table = {"before": [{"own": None, "w": 1}], "after": {m: [{"own": None, "w": 1}] for m in tied}, "their": their, "long": {"before": [None], "after": {m: [None] for m in tied}, "limit": 4}, "replied": {m: True for m in tied}}
    return {"eq": tied, "tied": tied, "read": "ok", "table": table, "goals": [{"source": "x"}], "pick_rule": "m1", "pick_goals": "m1", "move": "m1"}


def test_the_base_rate_compares_accepted_with_not_accepted_tied_moves_for_each_signal():
    pid = "own:g:1"
    preps = {pid: prep_with(True)}
    br = SM.base_rate(preps, [pid], {pid: ["m1"]}, SM.DH.split_part(pid))
    r = br["rows"]["1 the defence: the move costs some goal of the other side"]
    assert br["n"] == 1 and r["pairs_acc"] == [1, 1] and r["pairs_non"] == [1, 2] and r["pos_acc"] == 1 and r["pos_non"] == 1
    sl = SM.signal_silence(preps, [pid])
    assert sl["rows"]["1 the defence: the move costs some goal of the other side"]["silent"] == 0
    assert sl["rows"]["2 the long count: the move lowers the count of some goal"]["silent"] == 1


def test_no_pick_source_reads_answers_and_the_signal_reads_are_in_the_preparation_only():
    for fn in (SP.cost_to_theirs, SP.defence_gain, SP.defends, SP.eases, SP.reach_table, SP.long_count, SP.long_table, SP.long_drop, SP.line_keeps_unit, SP.replied_drop, SP.signal_score, SP.goal_pick,
               SM.prepare_position, SM.prepare_signals, SM.works_targets, SM.candidates, SM._pick_of, SM._cost_of):
        assert not re.search(r"answers|accepted|oracle|ladder_oracle", inspect.getsource(fn), re.I), fn.__name__
    assert "answers" in inspect.signature(SM.base_rate).parameters
    assert not re.search(r"answers|accepted|oracle", (DISC / "subgoal_pick.py").read_text(), re.I)


def test_the_candidates_of_the_plan_are_there_and_every_pick_is_a_tied_move():
    cands = SM.candidates()
    names = [c["name"] for c in cands]
    assert len(names) == len(set(names))
    new = [c for c in cands if c.get("new")]
    assert {c["mode"] for c in new} == set(SP.NEW_MODES) and len(new) == 5
    prep = prep_with(True)
    for c in cands:
        if not c.get("random"):
            assert c["pick"]("p", prep) in prep["tied"]


@pytest.mark.parametrize("name", ["subgoal_pick.py", "subgoal_pick_measure.py"])
def test_the_layer_and_the_measure_name_no_chess_object(name):
    hit = re.search(CHESS_WORDS, (DISC / name).read_text(), flags=re.I)
    assert hit is None, f"{name}: {hit.group(0)!r}"


def test_the_reader_names_no_chess_object_in_its_new_code_and_the_renderer_stays_in_the_printing_script():
    for name in ("subgoal_pick.py", "subgoal_pick_measure.py"):
        assert "import chess" not in (DISC / name).read_text()
    assert "import chess" in (DISC / "ladder_print.py").read_text()
