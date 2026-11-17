"""Plan 78.75: the rule for choosing a move, in a sandbox (catspace/incubator/analyst/discovery/move_rule.py; the harness move_rule_play.py; the trace move_rule_print.py). SYSTEM_SPEC section 8 "The rule for choosing a move"
and section 7 "Holding a subgoal". One test for each branch of the rule, for the held subgoal's keeping and dropping, for the ranking by stoppers, for the graph's reading of a move after it, and the two guarantees the
plan states: the rule never returns a move outside the tied set, and the rule's own files name no chess object."""
from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import move_rule as MR
from catspace.incubator.analyst.discovery import subgoal_goals as SG

ROOT = Path(__file__).resolve().parents[1]
DISC = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
ROLES = ["white", "black"]


# ------------------------------------------------------------------------------------------------------------------------------ a small world made of data (no engine)

def pf(units: dict, bear: dict, shut: "dict | None" = None, mover: str = "white") -> SG.PositionFacts:
    """units: id -> (role, kind, place); bear: id -> places it bears on; shut: (id, place) -> the ids between."""
    at = {p: u for u, (_s, _k, p) in units.items()}
    return SG.PositionFacts("t", mover, dict(units), at, {u: set(v) for u, v in bear.items()}, {}, dict(shut or {}))


class World:
    """A `Position` stand-in: the graph now and the graph after each move, given."""

    def __init__(self, now, after: dict, moves=(), mover: int = 0):
        self.now, self._after, self.roles, self.mover, self.moves = now, after, ROLES, mover, list(moves)

    def after(self, move):
        return self._after.get(move, self.now)


T = ("black", "kind_t", "t1")


def fact(name, status="unmet", vars_=None, stoppers=()):
    return {"name": name, "vars": vars_ or {"T": T}, "status": status, "cause": {"type": "x", "stoppers": list(stoppers), "victim": None, "words": ""} if stoppers or status == "unmet" else None, "fix": None}


def goal(target=T, facts=None, line=(), stake=0.1, steps=()):
    return {"target": target, "taker": "white", "owner": "black", "facts": facts or [fact("bears", stoppers=[("white", "kind_a", "a1")])], "line": list(line), "steps": list(steps), "stake": stake, "source": "no_bearer_no_move",
            "edits": [], "key": "k", "unmet": [], "verdict": None}


def resp(move, how="a guard is added", state="KEPT"):
    return {"names": [move, "x1x2"], "how": how, "state": state, "ended": "quiet", "loses_nothing": True, "net": 0, "first": (0, 1)}


def row(kind="kind_u", place="u1", stake=0.4, responses=(), verdict="CONFIRMED", big=True, **kw):
    return dict({"side": "mover", "owner": 0, "kind": kind, "place": place, "place_ix": 5, "stake": stake, "verdict": verdict, "line": ["z1z2", "z2z3"], "ended": "quiet", "by": [("kind_b", "b1")], "goal": False,
                 "two_moves": False, "response": list(responses), "big": big}, **kw)


def lists(mover_rows=(), other_rows=()):
    return {"mover": list(mover_rows), "other": list(other_rows), "exchange": [], "goal": [], "dropped_rows": []}


EMPTY = pf({"u1": ("white", "kind_u", "u1"), "t": T and ("black", "kind_t", "t1")}, {})


def world(after=None, moves=()):
    return World(EMPTY, after or {}, moves)


# ------------------------------------------------------------------------------------------------------------------------------ branch 1: threats first

def test_step_1_answers_the_threat_with_a_kept_response_among_the_tied_moves():
    r = row(responses=[resp("m2"), resp("m3")])
    pick, reason, held = MR.choose(world(), ["m1", "m2", "m3"], lists([r]), [], None)
    assert pick == "m2" and reason.branch == "1" and "threat" in reason and "m2" in reason and held is None      # the order of the row's responses: the first kept one


def test_step_1_prefers_the_answer_that_is_also_a_step_of_the_held_goal():
    g = goal(line=["m3", "q1q2"])
    held = MR.goal_record(g, 0, 10)
    r = row(responses=[resp("m2"), resp("m3")])
    pick, reason, held2 = MR.choose(world(moves=["a"] * 11), ["m1", "m2", "m3"], lists([r]), [g], held)
    assert pick == "m3" and reason.branch == "1+held" and held2["target"] == T and held2["age"] == 1


def test_a_counter_threat_counts_only_when_the_read_kept_it():
    kept = row(responses=[resp("m2", how="a counter threat")])
    pick, reason, _h = MR.choose(world(), ["m1", "m2"], lists([kept]), [], None)
    assert pick == "m2" and "counter-threat" in reason
    not_kept = row(responses=[resp("m2", how="a counter threat", state="STILL WON")])      # the read found the taker still ahead: not an answer
    pick, reason, _h = MR.choose(world(), ["m1", "m2"], lists([not_kept]), [], None)
    assert reason.branch != "1" and pick == "m1"


def test_the_threat_of_the_highest_stake_is_answered_first():
    low = row(kind="kind_l", place="l1", stake=0.1, responses=[resp("m2")])
    high = row(kind="kind_h", place="h1", stake=0.5, responses=[resp("m3")])
    pick, reason, _h = MR.choose(world(), ["m1", "m2", "m3"], lists([low, high]), [], None)
    assert pick == "m3" and "h1" in reason


def test_a_threat_in_two_moves_or_not_confirmed_is_not_one_to_answer_now():
    r1 = row(responses=[resp("m2")], two_moves=True)
    r2 = row(responses=[resp("m2")], verdict="UNREAD")
    pick, reason, _h = MR.choose(world(), ["m1", "m2"], lists([r1, r2]), [], None)
    assert pick == "m1" and reason.branch == "4"


# ------------------------------------------------------------------------------------------------------------------------------ branch 2: advance the subgoal

def test_step_2_with_no_threat_takes_up_the_first_ranked_goal_and_plays_its_first_move():
    g = goal(line=["m2", "q1q2"])
    pick, reason, held = MR.choose(world(), ["m1", "m2", "m3"], lists(), [g], None)
    assert pick == "m2" and reason.branch == "2 first" and held["target"] == T and held["origin"] == "goal"
    assert "first move of the line" in reason


def test_step_2_when_every_tied_move_answers_the_threat_the_goal_decides_among_them():
    r = row(responses=[resp("m1"), resp("m2"), resp("m3")])
    g = goal(line=["m3", "q1q2"])
    pick, reason, _h = MR.choose(world(), ["m1", "m2", "m3"], lists([r]), [g], None)
    assert pick == "m3" and reason.branch == "2 first" and "answered by every tied move" in reason


def test_step_2_keeps_the_held_goal_while_its_stoppers_do_not_rise():
    other = goal(target=("black", "kind_o", "o1"), facts=[fact("bears", stoppers=[("white", "k", "a1")])], line=["m1", "x1x2"])      # ranked first: one stopper
    mine = goal(facts=[fact("bears", stoppers=[("white", "k", "a1"), ("black", "k", "a2")])], line=["m2", "x1x2"])
    held = MR.goal_record(mine, 3, 10)
    assert held["weight"] == 2
    pick, reason, held2 = MR.choose(world(moves=["a"] * 12), ["m1", "m2"], lists(), [other, mine], held)
    assert pick == "m2" and reason.branch == "2 held" and held2["target"] == T and held2["age"] == 4


def test_step_2_drops_a_held_goal_whose_stoppers_rose_and_takes_up_the_first_ranked():
    other = goal(target=("black", "kind_o", "o1"), facts=[fact("bears", stoppers=[("white", "k", "a1")])], line=["m1", "x1x2"])
    mine = goal(facts=[fact("bears", stoppers=[("white", "k", "a1"), ("black", "k", "a2"), ("black", "k", "a3")])], line=["m2", "x1x2"])
    held = dict(MR.goal_record(mine, 3, 10), weight=2)
    pick, reason, held2 = MR.choose(world(), ["m1", "m2"], lists(), [other, mine], held)
    assert pick == "m1" and "rose from 2 to 3" in reason and held2["target"] == ("black", "kind_o", "o1")


def test_step_2_takes_another_goal_when_the_held_one_has_no_step_among_the_tied_moves():
    other = goal(target=("black", "kind_o", "o1"), facts=[fact("bears", stoppers=[("white", "k", "a1"), ("black", "k", "a2")])], line=["m2", "x1x2"])
    mine = goal(line=["zz", "x1x2"])
    held = MR.goal_record(mine, 1, 10)
    pick, reason, held2 = MR.choose(world(), ["m1", "m2"], lists(), [mine, other], held)
    assert pick == "m2" and reason.branch == "2 other" and held2["target"] == ("black", "kind_o", "o1")


def test_a_move_advances_a_goal_when_the_graph_after_it_has_a_fact_true():
    """The graph says it: a unit of ours shuts the line to the target and a move of it opens the line (the fact `bears` becomes true); nothing in the move's name is read."""
    units0 = {"a": ("white", "kind_a", "a1"), "s": ("white", "kind_s", "a3"), "t": T}
    now = pf(units0, {"a": set()}, {("a", "t1"): ["s"]})
    open_ = pf({"a": ("white", "kind_a", "a1"), "s": ("white", "kind_s", "b5"), "t": T}, {"a": {"t1"}})
    g = goal(facts=[fact("bears", vars_={"A": ("white", "kind_a", "a1"), "T": T}, stoppers=[("white", "kind_s", "a3")])])
    w = World(now, {"open": open_})
    adv = MR.advance_of(g, w, "open")
    assert adv is not None and adv[2] > 0 and "bears" in adv[3]
    assert MR.advance_of(g, w, "idle") is None and MR.facts_after(g, open_) == ["bears"]
    pick, reason, _h = MR.choose(w, ["idle", "open"], lists(), [g], None)
    assert pick == "open" and reason.branch == "2 first"


def test_a_move_that_brings_a_unit_to_bear_is_a_step_of_the_goal():
    units0 = {"a": ("white", "kind_a", "c1"), "t": T}
    now = pf(units0, {"a": set()})
    moved = pf({"a": ("white", "kind_a", "d2"), "t": T}, {"a": set()})
    g = goal(facts=[fact("bears", stoppers=[])], steps=[{"step": "bring", "unit": ("white", "kind_a", "c1"), "route": ["c1", "d2", "t1"], "moves": 2, "by": "the graph"}])
    w = World(now, {"go": moved})
    assert MR.advance_of(g, w, "go")[1] is True and MR.advance_of(g, w, "stay") is None


# ------------------------------------------------------------------------------------------------------------------------------ branch 3: a big threat nothing answers

def test_step_3_a_big_threat_with_no_kept_response_becomes_the_held_subgoal():
    units = {"u": ("white", "kind_u", "u1"), "e": ("black", "kind_e", "e1"), "f": ("black", "kind_f", "f1")}
    now = pf(units, {"e": {"u1"}, "f": {"u1"}})
    after = pf(units, {"e": {"u1"}, "f": set()})                                      # after "m2" one of theirs no longer bears on the unit
    r = row(responses=[resp("zz")], big=True)
    pick, reason, held = MR.choose(World(now, {"m2": after}), ["m1", "m2"], lists([r]), [goal()], None)
    assert pick == "m2" and reason.branch == "3" and "refuting it is now the subgoal" in reason
    assert held["origin"] == "refute" and held["target"] == ("white", "kind_u", "u1")


def test_step_3_a_small_threat_nothing_answers_is_left_standing_and_the_goal_goes_on():
    r = row(responses=[resp("zz")], big=False)
    g = goal(line=["m2", "x1x2"])
    pick, reason, held = MR.choose(world(), ["m1", "m2"], lists([r]), [g], None)
    assert pick == "m2" and reason.branch == "2 first" and "small threat" in reason and "left standing" in reason


def test_the_refutation_held_is_released_when_its_threat_is_gone():
    held = {"origin": "refute", "target": ("white", "kind_u", "u1"), "source": "threat", "weight": 0, "unmet": [], "age": 2, "since": 5}
    g = goal(line=["m2", "x1x2"])
    pick, reason, held2 = MR.choose(world(), ["m1", "m2"], lists(), [g], held)
    assert held2["origin"] == "goal" and "gone" in reason


# ------------------------------------------------------------------------------------------------------------------------------ branch 4

def test_step_4_nothing_applies_the_search_s_own_choice():
    pick, reason, held = MR.choose(world(), ["m1", "m2", "m3"], lists(), [], None)
    assert pick == "m1" and reason.branch == "4" and held is None
    g = goal(line=["zz", "x1x2"])
    pick, reason, held = MR.choose(world(), ["m1", "m2"], lists(), [g], None)
    assert pick == "m1" and reason.branch == "4"


def test_the_steps_can_be_priced_apart():
    r = row(responses=[resp("m2")])
    g = goal(line=["m3", "x1x2"])
    ls, gs, tied = lists([r]), [g], ["m1", "m2", "m3"]
    assert MR.choose(world(), tied, ls, gs, None, (1, 3))[0] == "m2"          # threats only: step 2 off
    assert MR.choose(world(), tied, ls, gs, None, (2,))[0] == "m3"            # goals only: steps 1 and 3 off
    assert MR.choose(world(), tied, ls, [], None, (2,))[1].branch == "4"


# ------------------------------------------------------------------------------------------------------------------------------ the ranking counts stoppers, not facts

def test_goals_are_ranked_by_stoppers_not_by_facts():
    four = goal(target=("black", "k", "p1"), facts=[fact("bears", stoppers=[("w", "k", f"s{i}") for i in range(4)])], stake=0.9)      # one fact, four units in the way
    two = goal(target=("black", "k", "p2"), facts=[fact("bears", stoppers=[("w", "k", "s1")]), fact("unguarded", stoppers=[("b", "k", "s2")])], stake=0.1)    # two facts, one unit each
    assert MR.weight_of(four) == 4 and MR.weight_of(two) == 2
    assert MR.rank_goals([four, two])[0] is two


# ------------------------------------------------------------------------------------------------------------------------------ the guarantee: never a move outside the tied set

def test_the_rule_never_returns_a_move_outside_the_tied_set():
    rng = random.Random(7)
    names = [f"m{i}" for i in range(12)]
    for _ in range(200):
        tied = rng.sample(names, rng.randrange(1, 8))
        resp_moves = rng.sample(names, rng.randrange(0, 6))              # responses and lines may name moves that are not tied
        rows = [row(place=f"r{k}", stake=rng.choice([0.1, 0.4, None]), responses=[resp(m) for m in rng.sample(resp_moves, len(resp_moves))], big=rng.random() < 0.5) for k in range(rng.randrange(0, 3))]
        gl = [goal(target=("black", "kind_t", f"g{k}"), line=[rng.choice(names), "x1x2"], facts=[fact("bears", stoppers=[("w", "k", "s")] * rng.randrange(0, 4))]) for k in range(rng.randrange(0, 3))]
        held = MR.goal_record(gl[0], 1, 3) if gl and rng.random() < 0.5 else None
        for steps in ((1, 2, 3), (1, 3), (2,)):
            pick, reason, _h = MR.choose(world(), tied, lists(rows), gl, held, steps)
            assert pick in tied and reason.branch in ("1", "1+held", "2 held", "2 other", "2 first", "2 next", "3", "4")


def test_an_empty_tied_set_is_refused():
    with pytest.raises(AssertionError):
        MR.choose(world(), [], lists(), [], None)


# ------------------------------------------------------------------------------------------------------------------------------ on the engine's own graph

def test_the_engine_graph_after_a_move_tells_a_line_opened():
    from catspace.incubator.analyst.discovery import threat_concrete as TC
    facts = SG.Facts("chess")
    facts.goal_name = "king"
    text = "4k3/p7/8/8/8/N7/8/R3K3 w - - 0 1"            # our unit on a3 stands between the rook and the target on a7
    pos = MR.Position(facts, "chess", text, [], ["white", "black"], 0)
    item = {"start": text, "moves": [], "raw": {"mover": 0, "board": []}, "names": {"places": [f + r for f in "abcdefgh" for r in "12345678"], "roles": ["white", "black"], "kinds": []}, "list": {"mover": [], "other": []}}
    c = {"mover": [], "other": [], "exchange": [], "goal": [], "dropped": {}, "dropped_rows": [], "resp_read": 0}
    from catspace.incubator.analyst.discovery import subgoal_units as SU
    SU.every_unit([item], [[0]], [c], facts, TC.Lines(TC.Record({"q": 4343, "b": 1878, "n": 1661, "r": 1605, "p": 1045}, {"q": 120.0, "b": 606.0, "n": 628.0, "r": 534.0, "p": 508.0}, "synthetic"), threads=1), [None])
    gl = [g for g in SG.goals(item, 0, c, facts) if g["target"][2] == "a7"]
    assert gl, "a goal for the unit on a7"
    g = gl[0]
    assert "bears" in g["unmet"] and MR.weight_of(g) >= 1
    adv = MR.advance_of(g, pos, "a3b5")                    # the unit leaves the file: the line is open
    assert adv is not None and "bears" in adv[3]
    assert MR.advance_of(g, pos, "e1d1") is None


# ------------------------------------------------------------------------------------------------------------------------------ the rule's own files name no chess object

@pytest.mark.parametrize("name", ["move_rule.py", "move_rule_play.py"])
def test_the_rule_and_its_harness_name_no_chess_object(name):
    src = (DISC / name).read_text()
    hit = re.search(CHESS_WORDS, src, flags=re.I)
    assert hit is None, f"{name}: {hit.group(0)!r}"


def test_the_printing_script_is_the_only_file_that_spells_moves_in_the_game_s_words():
    assert "import chess" in (DISC / "move_rule_print.py").read_text()
    for name in ("move_rule.py", "move_rule_play.py"):
        assert "import chess" not in (DISC / name).read_text()
