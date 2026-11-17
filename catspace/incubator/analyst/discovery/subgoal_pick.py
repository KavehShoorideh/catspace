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
                 PLAN 78.79: "defence" | "long" | "replied" | "defence_long" | "sum" (NEW_MODES, below); `facts` is then the count table with the keys "their", "long" and "replied" added by the caller.
                 Where no move lowers any count in the ranks asked (the hint is silent) the search's own move, the first of `tied`; a tie in the drop goes to the move earlier in `tied`.
      returns    one of `tied`, always (asserted): the hint orders what the search holds equal and never leaves the set.
Called by: subgoal_pick_measure.py (the run: preparation, the score, the ceiling) and ladder_print.py (the cases, which read the measure's pickle). Nothing in play calls it.

PLAN 78.79 (SYSTEM_SPEC 7: reachable has two layers, the hint; 8: the threat list as the near case; 9: the score). THREE SIGNALS the hint may read, each a table made once per position and read by the picks; all read the graph and the proposer only.
  shape_of(goal) -> {"target", "w", "steps", "unmet", "source"}
      a goal as the three numbers a defence is read by: the stoppers still in the way (move_rule.weight_of), the own moves its steps still need (a step with no count: STEP_NONE), the number of unmet facts.
  cost_to_theirs(position, tied, their_goals, facts, after=None) -> table                                              SIGNAL 1, THE DEFENCE
      position     a move_rule.Position (anything with `now.text`).   tied   the moves the search holds equal, its own move first.
      their_goals  the goals of the OTHER side at the position, as the proposer lists them for it (move_rule_play.RuleReader.read(..., side="other"): the same proposer run on the position with the sides exchanged).
      facts        a subgoal_goals.Facts: only `replay`, to tell which place the moved unit left and where it came (a target that moved is followed).
      after        per tied move: None (the position after it could not be read) or {"goals": the proposer's goals of that side to move at the position after the move (the same call, plain, on the position after it),
                   "works": the targets of theirs that already work there (subgoal_goals.works)}; `works` may hold the word "goal_unit" for the end of the game.
      returns      {"before": [shape_of per goal], "after": {move: None | [per goal {"state": "gone" | "works" | "kept", "cost": +1 | 0 | -1, "shape": shape or None}]}}. State "kept": the goal is still listed for the target (followed
                   to its place after the move); cost +1 when a count of its (stoppers, steps, unmet facts) rose and none fell; -1 when one fell and none rose. State "gone": the target is no longer a goal and does not work (it cannot be built now,
                   or its unit is no longer there): cost +1. State "works": their threat works after the move: cost -1 (a move that makes their goal easier is a move to avoid).
  defence_gain(table, move, ranks=None) -> int      the summed cost over the first `ranks` goals (None: all); positive is a defence, negative a move to avoid.   defends(table, move) / eases(table, move) -> bool: some goal costs / some goal eases.
  reach_table(pf, facts, editor, limit=LONG_LIMIT, fuse=LONG_FUSE) -> {"states": [(depth, PositionFacts)], "cut_depth", "cut_fuse", "reads"}                SIGNAL 2, THE LONG COUNT
      Every position that comes of ONE unit of ours walking k own moves (k up to `limit`), the other side not moving, over the graph's own move relation (`may`, the places the rules let the unit move to) with the unit put on the place by
      `editor.relocate` (the printing script's text editor, the same call subgoal_goals.route_of makes); a walk goes to an empty place only (a taking is not walked). `fuse` is the cost guard: the graph reads one table may make.
      cut_depth: some walk could go on past `limit`; cut_fuse: the guard was hit. Cached on `pf`.
  long_count(position, goal, facts, limit, move=None, editor=None, fuse=LONG_FUSE) -> int | None
      For a goal whose unmet facts are all of bearing (bears, borne_on, covered): the fewest own moves of some unit of ours that make each unmet fact true, summed (as layer one sums), at the position or, with `move`, at the position after it;
      0 when none is unmet; None for a goal with another unmet fact, when our unit is not to move there, or when no walk within `limit` (and the guard) does it. No move generator of its own: it asks the graph.
  long_table(position, tied, goals, facts, limit, editor, fuse=LONG_FUSE) -> {"before": [count], "after": {move: [count]}, "cut": {guard hits}, "limit"}      a count that is None counts as limit+1 in `long_drop`.
      "cut" counts the walk tables read, those cut by the limit of moves and by the guard of reads, and the (goal, position) pairs of a bearing-only goal with no count where the search was cut (`none_cut`: more walk might have found one)
      against where it ended by itself (`none_exhausted`).
  line_keeps_unit(position, facts, move, read) -> bool | None                                                                                            SIGNAL 3, THE STOPPERS WITH A REPLY READ
      `read` is the line read after the move (threat_concrete mode 9: the move, then the best replies of both sides): {"names", "settled", "ok", "cut"}. True when the line, replayed from the position, does not lose the unit that made
      the move; False when it does; None when the line was not read or was cut (then the signal does not count it).
  long_drop(table, move, ranks=None) / replied_drop(table, move, ranks=None)      the fall of the long count / of the stoppers (scale "w") when the line keeps the unit, else 0.
  signal_score(table, move, mode) -> tuple         the score of a move for the new modes: "defence", "long", "replied", "defence_long" (the defence first, then the long count), "sum".
Called by: subgoal_pick_measure.py only (and ladder_print.py reads its pickle for the cases).
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import move_rule as MR
from catspace.incubator.analyst.discovery import subgoal_goals as SG

LIMIT = 2                                                        # layer one's own limit of moves (subgoal_goals.layer_one's default)
NONE_COUNT = LIMIT + 1                                           # a goal with no count (not reachable by our moves alone within the limit) counts as one more than the limit
MODES = ("rule", "goals", "top1", "top2", "top3", "all")
NEW_MODES = ("defence", "long", "replied", "defence_long", "sum")
RANKS = {"top1": 1, "top2": 2, "top3": 3, "all": None}
LONG_LIMIT = 4                                                   # own moves a walk is read to (a cost guard of the search, not of the design; the table says how often it was hit)
LONG_FUSE = 400                                                  # graph reads one position's walks may make (a cost guard; the table says how often it was hit)
STEP_NONE = 3                                                    # a step with no count weighs this many own moves in a goal's shape
GAIN_UNIT = 1
BEARING = ("bears", "borne_on", "covered")


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
        if move is not None and pf is not None and m is not None and (s.get("move") == move or MR.unit_stepped(s, now, pf)):
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


# ------------------------------------------------------------------------------------------------------------------------------ signal 1: the defence

def shape_of(g: dict) -> dict:
    steps = sum(STEP_NONE if s.get("moves") is None else s["moves"] for s in g.get("steps") or ())
    return {"target": tuple(g["target"]), "w": MR.weight_of(g), "steps": steps, "unmet": len(g["unmet"]), "source": g.get("source")}


def moved_place(position, facts, move: str) -> "tuple | None":
    """(the place the unit that made `move` left, the place it came to), or None."""
    try:
        pl = facts.replay(position.now.text, [move])
    except Exception:
        return None
    return (pl[0]["from"], pl[0]["to"]) if pl and pl[0]["from"] and pl[0]["to"] else None


def _cost_of_pair(b: dict, a: dict) -> int:
    d = [a["w"] - b["w"], a["steps"] - b["steps"], a["unmet"] - b["unmet"]]
    up, down = any(x > 0 for x in d), any(x < 0 for x in d)
    return GAIN_UNIT if up and not down else (-GAIN_UNIT if down and not up else 0)


def cost_to_theirs(position, tied: list, their_goals: list, facts, after: "dict | None" = None) -> dict:
    before = [shape_of(g) for g in their_goals]
    out = {"before": before, "after": {}}
    for m in tied:
        rd = (after or {}).get(m)
        if rd is None:
            out["after"][m] = None
            continue
        mv = moved_place(position, facts, m)
        now = {}
        for g in rd["goals"]:
            sh = shape_of(g)
            now[sh["target"]] = sh
        works = {tuple(w) for w in rd.get("works", ()) if not isinstance(w, str)}
        works_unit = "goal_unit" in rd.get("works", ())
        row = []
        for b in before:
            side, kind, place = b["target"]
            if mv is not None and place == mv[0]:
                place = mv[1]
            tgt = (side, kind, place)
            if tgt in now:
                row.append({"state": "kept", "cost": _cost_of_pair(b, now[tgt]), "shape": now[tgt]})
            elif tgt in works or (works_unit and b.get("source") == "goal_unit"):
                row.append({"state": "works", "cost": -GAIN_UNIT, "shape": None})
            else:
                row.append({"state": "gone", "cost": GAIN_UNIT, "shape": None})
        out["after"][m] = row
    return out


def defence_gain(table: dict, move: str, ranks: "int | None" = None) -> int:
    row = table["after"].get(move)
    if row is None:
        return 0
    return sum(r["cost"] for r in (row if ranks is None else row[:ranks]))


def defends(table: dict, move: str) -> bool:
    row = table["after"].get(move)
    return bool(row) and any(r["cost"] > 0 for r in row)


def eases(table: dict, move: str) -> bool:
    row = table["after"].get(move)
    return bool(row) and any(r["cost"] < 0 for r in row)


# ------------------------------------------------------------------------------------------------------------------------------ signal 2: the long count

def reach_table(pf, facts, editor, limit: int = LONG_LIMIT, fuse: int = LONG_FUSE) -> dict:
    key = (limit, fuse)
    cache = getattr(pf, "_reach", None)
    if cache is None:
        cache = pf._reach = {}
    if key in cache:
        return cache[key]
    out = {"states": [], "cut_depth": False, "cut_fuse": False, "reads": 0}
    cache[key] = out
    if editor is None:
        return out
    side = pf.mover
    for uid in sorted(u for u, (s, _k, _p) in pf.units.items() if s == side):
        seen = {pf.units[uid][2]}
        frontier = [(uid, pf)]
        stop = False
        for depth in range(1, limit + 1):
            nxt = []
            for u, cur in frontier:
                a = cur.units[u][2]
                for q in sorted(cur.may.get(u, ())):
                    if q in cur.at or q in seen:
                        continue
                    if out["reads"] >= fuse:
                        out["cut_fuse"] = stop = True
                        break
                    t2 = editor.relocate(cur.text, a, q)
                    if t2 is None:
                        continue
                    g = facts.of(t2)
                    out["reads"] += 1
                    u2 = g.at.get(q)
                    if u2 is None:
                        continue
                    seen.add(q)
                    nxt.append((u2, g))
                    out["states"].append((depth, g))
                if stop:
                    break
            frontier = nxt
            if stop or not frontier:
                break
            if depth == limit and any(cur.may.get(u) for u, cur in frontier):
                out["cut_depth"] = True
    return out


def long_count(position, goal: dict, facts, limit: int, move: "str | None" = None, editor=None, fuse: int = LONG_FUSE) -> "int | None":
    pf = position.now if move is None else position.after(move)
    if pf is None:
        return None
    unmet = [f for f in goal["facts"] if f["status"] in ("unmet", "unread")]
    if not unmet:
        return 0
    if any(f["name"] not in BEARING for f in unmet):
        return None
    if pf.mover != position.roles[position.mover]:
        return None
    tab = reach_table(pf, facts, editor, limit, fuse)
    total = 0
    for f in unmet:
        if MR.fact_true(f, goal, pf):
            continue
        ds = [d for d, g in tab["states"] if MR.fact_true(f, goal, g)]
        if not ds:
            return None
        total += min(ds)
    return total


def _bearing_only(goal: dict) -> bool:
    return all(f["name"] in BEARING for f in goal["facts"] if f["status"] in ("unmet", "unread"))


def long_table(position, tied: list, goals: list, facts, limit: int, editor, fuse: int = LONG_FUSE) -> dict:
    cut = {"tables": 0, "cut_depth": 0, "cut_fuse": 0, "unwalkable": 0, "reads": 0, "none_cut": 0, "none_exhausted": 0}
    before = [long_count(position, g, facts, limit, None, editor, fuse) for g in goals]
    after = {m: [long_count(position, g, facts, limit, m, editor, fuse) for g in goals] for m in tied}
    mine = position.roles[position.mover]
    for pf, counts in [(position.now, before)] + [(position.after(m), after[m]) for m in tied]:
        if pf is None or pf.mover != mine:
            cut["unwalkable"] += 1
            continue
        t = getattr(pf, "_reach", {}).get((limit, fuse))
        if t is None:
            continue
        cut["tables"] += 1
        cut["cut_depth"] += t["cut_depth"]
        cut["cut_fuse"] += t["cut_fuse"]
        cut["reads"] += t["reads"]
        for g, c in zip(goals, counts):                          # a goal that could have a count and has none: was the search cut (more walk might have found one) or exhausted?
            if c is None and _bearing_only(g):
                if t["cut_depth"] or t["cut_fuse"]:
                    cut["none_cut"] += 1
                else:
                    cut["none_exhausted"] += 1
    return {"before": before, "after": after, "cut": cut, "limit": limit, "eligible": sum(_bearing_only(g) for g in goals)}


def long_drop(table: dict, move: str, ranks: "int | None" = None) -> int:
    lt = table["long"]
    none = lt["limit"] + 1
    n = len(lt["before"]) if ranks is None else min(ranks, len(lt["before"]))
    return sum((none if lt["before"][i] is None else lt["before"][i]) - (none if lt["after"][move][i] is None else lt["after"][move][i]) for i in range(n))


# ------------------------------------------------------------------------------------------------------------------------------ signal 3: the stoppers, with a reply read

def line_keeps_unit(position, facts, move: str, read: "dict | None") -> "bool | None":
    if not read or not read.get("ok") or read.get("cut") or not read.get("names"):
        return None
    try:
        pl = facts.replay(position.now.text, list(read["names"][:max(1, read.get("settled") or len(read["names"]))]))
    except Exception:
        return None
    if not pl or pl[0]["uid"] is None:
        return None
    mine = position.roles[position.mover]
    uid = pl[0]["uid"]
    return not any(p["took"] and p["took"][0] == uid and p["took"][1] == mine for p in pl)


def replied_drop(table: dict, move: str, ranks: "int | None" = None) -> int:
    if not table["replied"].get(move):
        return 0
    return drop(table, move, "w", ranks)


def signal_score(table: dict, move: str, mode: str) -> tuple:
    d = defence_gain(table["their"], move) if table.get("their") else 0
    l = long_drop(table, move) if table.get("long") else 0
    r = replied_drop(table, move) if table.get("replied") else 0
    if mode == "defence":
        return (d,)
    if mode == "long":
        return (l,)
    if mode == "replied":
        return (r,)
    if mode == "defence_long":
        return (d, l)
    if mode == "sum":
        return (d + l + r,)
    raise AssertionError(mode)


def goal_pick(position, tied: list, goals: list, facts: "dict | None", mode: str, lists: "dict | None" = None, scale: str = "own") -> str:
    assert tied, "the hint is handed the moves the search holds equal"
    assert mode in MODES + NEW_MODES, mode
    if mode in NEW_MODES:
        assert facts is not None, "the new modes read the tables of the signals"
        best, pick = (0,) * len(signal_score(facts, tied[0], mode)), tied[0]
        for m in tied:                                           # strictly above nothing and above the best so far: a tie stays with the earlier move, the search's own first
            sc = signal_score(facts, m, mode)
            if sc > best:
                best, pick = sc, m
    elif mode in ("rule", "goals"):
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
