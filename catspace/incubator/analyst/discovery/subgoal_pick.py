"""PLAN 78.78 A: A GOAL AS A HINT AMONG THE MOVES THE SEARCH HOLDS EQUAL. SYSTEM_SPEC section 7 "Holding a subgoal and navigating toward it" (the hint: a subgoal orders moves, never adds to the value; AGENTS.md rule 7, spec 15)
and section 9 (a rule is scored by the surprises it resolves less those it creates; the score is deep_heads.score). The piece is the OPPORTUNITY POLICY'S choice among tied moves. A sandbox: nothing calls it in play, nothing is written to
rules/learned or data/derived. Game-free: a kind, a role, a place and a move are words that arrive as data; no object of any game is named here. No reference answer, no engine evaluation and no assumed worth is read:
the picks read only the goals (subgoal_goals.goals), the graph after a move (move_rule.Position) and, for two modes, the rule of plan 78.75 as built.

INTERFACE (what this module takes, what it returns, who calls it).
  count_table(position, tied, goals) -> table
      position   a move_rule.Position (or anything with `now` and `after(move)`: the graph of the position and of the position after a move, as subgoal_goals.PositionFacts).
      tied       the moves the search holds equal, the search's own move FIRST.
      goals      the goals of the side to move, in the order subgoal_goals.goals returns them (the order of plan 78.74); a goal is read, never changed.
      returns    {"before": [count per goal], "after": {move: [count per goal]}}, a count = {"own": layer one's count of OUR OWN moves still needed (subgoal_goals.layer_one; None where it is not defined),
                 "w": the stoppers still in the way (move_rule.weight_graph)}. The count after a move is recomputed on the position after it: a fact the graph tells true there is met; a step the move made (the unit left
                 the first place of its route) is one move shorter, and gone when none is left; then layer_one is asked again. `before` is the goal's own `own_moves` (the same function with no move made).
  goal_drop(table, move, scale, i) -> int     the fall of one goal's count (goal number i of the order) when `move` is played.
  drop(table, move, scale, ranks) -> int
      How far `move` lowers the counts of the goals at `ranks` (indices into the goal order), summed; scale "own" or "w". A count that is None counts as NONE_COUNT (one more than layer one's limit), so a move
      that makes an unreachable goal reachable lowers it. The same function serves the cases and the ceiling.
  goal_pick(position, tied, goals, facts, mode, lists=None, scale="own") -> move
      facts      the count table (above), or None (then it is computed here).
      mode       "rule"   move_rule.choose as built (steps 1, 2, 3; no held goal at a single position: the first goal of the order); needs `lists`, the concrete threat list.
                 "goals"  move_rule.choose with step 2 only.
                 "top1" | "top2" | "top3"   the tied move whose summed drop over the first k goals of the order is the largest and above nothing.
                 "all"    the same over every goal.
                 Where no move lowers any count in the ranks asked (the hint is silent) the search's own move, the first of `tied`; a tie in the drop goes to the move earlier in `tied`.
      returns    one of `tied`, always (asserted): the hint orders what the search holds equal and never leaves the set.
Called by: subgoal_pick_measure.py (the run: preparation, the score, the ceiling) and ladder_print.py (the cases, which read the measure's pickle). Nothing in play calls it.
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import move_rule as MR
from catspace.incubator.analyst.discovery import subgoal_goals as SG

LIMIT = 2                                                        # layer one's own limit of moves (subgoal_goals.layer_one's default)
NONE_COUNT = LIMIT + 1                                           # a goal with no count (not reachable by our moves alone within the limit) counts as one more than the limit
MODES = ("rule", "goals", "top1", "top2", "top3", "all")
RANKS = {"top1": 1, "top2": 2, "top3": 3, "all": None}


def own_count(goal: dict, pf, now, move: "str | None") -> "int | None":
    """Layer one's count for `goal` read on the graph `pf` of the position after `move` (`move` None: the position as it is, the goal's own `own_moves`). A fact the graph tells true in `pf` is met; a step the move made is one move
    shorter (gone when none is left); the rest as subgoal_goals.layer_one reads them."""
    facts_c = []
    for f in goal["facts"]:
        st = f["status"]
        if move is not None and pf is not None and st in ("unmet", "unread") and MR.fact_true(f, goal, pf):
            st = "met"
        facts_c.append((f["name"], st, f["cause"]["type"] if f.get("cause") else None))
    steps_c = []
    for s in goal.get("steps") or ():
        m = s.get("moves")
        if move is not None and pf is not None and m is not None and (s.get("move") == move or MR._unit_stepped(s, now, pf)):
            m -= 1
        if m is not None and m <= 0:
            continue
        steps_c.append((s["step"], m))
    return SG.layer_one(facts_c, steps_c, LIMIT)


def count_table(position, tied: list, goals: list) -> dict:
    now = position.now
    before = [{"own": own_count(g, now, now, None), "w": MR.weight_graph(g, now)} for g in goals]
    after = {}
    for m in tied:
        pf = position.after(m)
        after[m] = [{"own": own_count(g, pf, now, m), "w": (MR.weight_graph(g, pf) if pf is not None else b["w"])} for g, b in zip(goals, before)]
    return {"before": before, "after": after}


def _num(c: "int | None") -> int:
    return NONE_COUNT if c is None else c


def goal_drop(table: dict, move: str, scale: str, i: int) -> int:
    """The fall of the count of goal number `i` of the order when `move` is played; positive is nearer."""
    key = "own" if scale == "own" else "w"
    return _num(table["before"][i][key]) - _num(table["after"][move][i][key])


def drop(table: dict, move: str, scale: str, ranks: "int | None") -> int:
    """The summed fall of the counts of the first `ranks` goals (None: every goal) when `move` is played; positive is nearer."""
    n = len(table["before"]) if ranks is None else min(ranks, len(table["before"]))
    return sum(goal_drop(table, move, scale, i) for i in range(n))


def changes(table: dict, move: str) -> bool:
    """`move` changes the count of some goal (either scale): the hint has something to say about it."""
    return any(x != y for x, y in zip(table["before"], table["after"][move]))


def goal_pick(position, tied: list, goals: list, facts: "dict | None", mode: str, lists: "dict | None" = None, scale: str = "own") -> str:
    assert tied, "the hint is handed the moves the search holds equal"
    assert mode in MODES, mode
    if mode in ("rule", "goals"):
        assert lists is not None, "the rule reads the threat list"
        pick = MR.choose(position, tied, lists, goals, None, (1, 2, 3) if mode == "rule" else (2,))[0]
    else:
        table = facts if facts is not None else count_table(position, tied, goals)
        ranks = RANKS[mode]
        best, pick = 0, tied[0]
        for m in tied:                                           # a strict gain over the best so far: a tie stays with the earlier move, the search's own first
            d = drop(table, m, scale, ranks)
            if d > best:
                best, pick = d, m
    assert pick in tied, "the hint orders what the search holds equal; it never leaves the set"
    return pick
