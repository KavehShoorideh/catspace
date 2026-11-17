"""PLAN 78.75 A: THE RULE FOR CHOOSING A MOVE, IN A SANDBOX. SYSTEM_SPEC section 8 "The rule for choosing a move" (the wake's steps 3 to 6) and section 7 "Holding a subgoal and navigating toward it". The pieces it
joins are the THREAT LIST (threat_concrete.py: the rows against both sides, each with its kept responses, read as lines) and the PROPOSER (subgoal_goals.py, subgoal_units.py: the goals, as facts of the graph with
what stops them). A sandbox: the shipped player is not changed, nothing is written to rules/learned or data/derived. Game-free: a kind, a role, a place and a move are words read from the engine at run time; no unit, place
or kind is named here. No engine evaluation, no external model, no assumed worth: the rule reads nothing but the two lists and the graph. A SUBGOAL ORDERS AND NEVER ADDS TO THE VALUE (AGENTS.md rule 7, spec 15): the
rule is handed the moves the search holds EQUAL (the 78.50 form, `solve_equals_batch`) and returns one of them; it asserts that the move is one of them.

INTERFACE (what this module takes, what it returns, who calls it).
  choose(position, tied, lists, goals, held, steps=(1, 2, 3)) -> (move, reason, held')
      position   a `Position` (below): the moves played, the side to move, the roles, the graph of the position (`now`) and, asked move by move, the graph after a move (`after(move)`).
      tied       the moves the search holds equal, as names, the search's own move FIRST (the 78.50 form; the match harness of plan 78.67 builds it so). The rule returns one of them and never another.
      lists      the concrete threat list of the position (threat_concrete.concrete_lists, one item): {"mover": the rows against the side to move, "other": the rows against the other side, ...}. A row is
                 CONFIRMED (the taker ends ahead by the record) and carries `response` (its KEPT responses: lines read, each with `names`, the first of which is the move, and `how`), `stake`, `goal` (a forced end of the
                 game) and `two_moves` (a threat in two moves: not a threat to answer now).
      goals      the goals of the side to move (subgoal_goals.goals on the list augmented by subgoal_units.every_unit): facts met or unmet, with what stops each (`cause["stoppers"]`), the `steps`, the `line`.
                 The rule ranks them by STOPPERS (below), not by facts: it does not use the order it is given.
      held       the subgoal carried from the last move, or None: {"origin": "goal" | "refute", "target": (role, kind, place), "weight": the stoppers it had, "unmet": fact names, "age": moves held, "since": the ply}.
      steps      which of the rule's steps are on: (1, 2, 3) is the rule; (1, 3) is the threats only (step 2 off); (2,) is the goals only (steps 1 and 3 off). Used by the measure to price each half.
      -> move    one of `tied`, always. reason: a `Reason` (a str in words of the graph: which threat, which answer, which goal, which fact; with `.branch` the step that fired and `.detail` the data). held': the
                 subgoal now held (the same form as `held`), or None.
  weight_of(goal)                 the goal's count of STOPPERS (the 78.74 read: a fact with four units in the way is four problems, so it counts four, not one). weight_graph(goal, pf) the same asked of a graph (the units
                                  standing between, the units that guard), the scale `advance_of` compares before and after a move.
  rank_goals(goals)               the goals' order by that count, then the larger stake.
  advance_of(goal, position, move)  how one move serves one goal, asked of the graph after the move: (is the first move of the goal's line, brings a unit to bear (a step), the fall of the goal's weight, the
                                    words of what it did) or None if it does not serve it.
  facts_after(goal, pf)           the unmet facts of a goal that are TRUE in a graph `pf` (only the facts the graph can tell: bears, borne_on, unguarded, covered; the others are not told: never counted as made true).
  Position(facts, variant, start, moves, roles, mover)   the position's graph and, cached per move, the graph after it (subgoal_goals.Facts reads both).
  Called by: move_rule_play.py (the match harness of 78.67 as one more tie-break) and move_rule_print.py (the trace) now; the wake's choice of move later. Nothing in play calls it.

THE RULE (the plan 78.75 text, spec 8's guide made exact):
 1. THREATS FIRST. The CONFIRMED rows against us, by stake (a forced end of the game first; no stake last). For the first row that some, but not every, tied move answers, in that order: the answers are the tied
    moves that are KEPT responses of the concrete read (a counter-threat is one only when the read kept it); among several, one that is also a step of the held goal; then the order of the row's responses.
 2. NO THREAT, OR EVERY TIED MOVE ALREADY ANSWERS IT: advance the subgoal. A move advances a goal when it is the first move of the goal's line, a step that brings a unit to bear, or makes unmet facts true or removes
    a stopper (asked of the graph after the move). The held goal is kept while its stoppers do not rise, else the first-ranked goal is taken up. If no tied move advances the held goal, the first goal in rank order that
    one does is taken up and the reason says the held one had no step.
 3. A BIG THREAT WITH NO KEPT RESPONSE AMONG THE TIED MOVES: refuting it becomes the held subgoal; the reason says so. A move refutes in part when, in the graph after it, fewer of theirs bear on the unit or more of ours
    do. BIG is the stake ranked within the position (threat_list.rank_high_of: top half of the rows with a stake, stake above nothing; a forced end of the game is always big). A small threat with no answer is left standing.
 4. NOTHING APPLIES: the search's own choice (the first of `tied`).
"""
from __future__ import annotations

import cs_engine as CE

from catspace.incubator.analyst.discovery import threat_list as TL

STAKE_NONE = 9.0                                                # a goal with no stake (the goal unit) sorts as a large stake in the rank (as subgoal_goals.goals does)
CHECKED = ("bears", "borne_on", "unguarded", "covered")        # the facts the graph alone can tell after a move


class Reason(str):
    """The reason in words, with the step that fired (`branch`: "1", "1+held", "2 held", "2 other", "2 first", "2 next", "3", "4") and the data."""
    branch: str
    detail: dict

    def __new__(cls, text: str, branch: str, detail: "dict | None" = None):
        o = super().__new__(cls, text)
        o.branch, o.detail = branch, dict(detail or {})
        return o


class Position:
    """The graph of the position and of the position after a move. `facts` is a subgoal_goals.Facts; `roles` the role words; `mover` the absolute side to move."""

    def __init__(self, facts, variant: str, start: "str | None", moves: list, roles: list, mover: int):
        self.facts, self.variant, self.start, self.moves, self.roles, self.mover = facts, variant, start, list(moves), list(roles), mover
        self._now = None
        self._after: dict = {}

    @property
    def now(self):
        if self._now is None:
            self._now = self.facts.of(CE.state(self.variant, self.start, self.moves)[0])
        return self._now

    def after(self, move: str):
        if move not in self._after:
            try:
                text = CE.state(self.variant, self.start, self.moves + [move])[0]
                self._after[move] = self.facts.facts_for(text, self.roles[self.mover]) or self.facts.of(text)      # our view of it (the side that moved, to move again after the pass of the game: `shut` is read for the side to move only)
            except Exception:                                   # a move the engine will not play: no graph
                self._after[move] = None
        return self._after[move]


# ------------------------------------------------------------------------------------------------------------------------------ goals: the stoppers, the rank, how a move serves one

def stoppers_of(fact: dict) -> list:
    return list(fact["cause"]["stoppers"]) if fact.get("cause") else []


def weight_of(goal: dict) -> int:
    """The count of STOPPERS of a goal: for each unmet fact the units that stop it (one if none is named), summed."""
    return sum(max(1, len(stoppers_of(f))) for f in goal["facts"] if f["status"] in ("unmet", "unread"))


def rank_goals(goals: list) -> list:
    return sorted(goals, key=lambda g: (weight_of(g), -(g["stake"] if g["stake"] is not None else STAKE_NONE), g["target"][2]))


def _target_at(pf, goal: dict) -> bool:
    side, kind, place = goal["target"]
    uid = pf.at.get(place)
    return uid is not None and pf.units[uid] == (side, kind, place)


def fact_true(f: dict, goal: dict, pf) -> "bool | None":
    """True / False for a fact the graph can tell in `pf`, None for one it cannot (then it is never counted as made true)."""
    name = f["name"]
    if name not in CHECKED:
        return None
    taker, owner = goal["taker"], goal["owner"]
    if name in ("bears", "borne_on"):
        return _target_at(pf, goal) and bool(pf.bearers(goal["target"][2], taker))
    if name == "unguarded":
        return _target_at(pf, goal) and not pf.bearers(goal["target"][2], owner)
    p = f["vars"].get("P")
    return bool(p is not None and pf.bearers(p[2], taker))     # covered(P)


def facts_after(goal: dict, pf) -> list:
    """The names of the goal's unmet facts that are true in the graph `pf`."""
    return [f["name"] for f in goal["facts"] if f["status"] in ("unmet", "unread") and fact_true(f, goal, pf)]


def weight_graph(goal: dict, pf) -> int:
    """The goal's weight asked of a graph `pf`, on the same scale before and after a move: a fact the graph tells true weighs nothing; a line shut weighs the units standing between (the fewest on any line of ours to the
    place); a place of theirs that is guarded weighs the units that bear on it; any other unmet fact weighs the stoppers the line named (one at least)."""
    w = 0
    taker, owner = goal["taker"], goal["owner"]
    for f in goal["facts"]:
        if f["status"] not in ("unmet", "unread"):
            continue
        name = f["name"]
        if name in CHECKED and fact_true(f, goal, pf):
            continue
        base = max(1, len(stoppers_of(f)))
        if name in ("bears", "borne_on", "covered"):
            place = goal["target"][2] if name != "covered" else (f["vars"].get("P") or (None, None, None))[2]
            if place is None or (name != "covered" and not _target_at(pf, goal)):
                w += base
                continue
            shuts = [len(b) for (u, p), b in pf.shut.items() if p == place and pf.units[u][0] == taker and b]
            w += min(shuts) if shuts else base
        elif name == "unguarded":
            n = len(pf.bearers(goal["target"][2], owner)) if _target_at(pf, goal) else 0
            w += max(1, n)
        else:
            w += base
    return w


def _unit_stepped(step: dict, now, pf) -> bool:
    """A step's unit left the first place of its route and a unit of its kind and side stands on a later place of the route that was empty."""
    u, route = step.get("unit"), step.get("route") or []
    if not u or len(route) < 2:
        return False
    side, kind, here = u[0], u[1], route[0]
    if now.at.get(here) is None or now.units[now.at[here]] != (side, kind, here) or (here in pf.at and pf.units[pf.at[here]] == (side, kind, here)):
        return False
    return any(p in pf.at and pf.units[pf.at[p]] == (side, kind, p) and p not in now.at for p in route[1:])


def advance_of(goal: dict, position: Position, move: str) -> "tuple | None":
    """How `move` serves `goal`: (first move of the line, brings a unit to bear, the fall of the weight, words), or None when it does not."""
    pf = position.after(move)
    first = bool(goal.get("line")) and goal["line"][0] == move
    step = False
    stepped = ""
    for st in goal.get("steps") or ():
        if st.get("move") == move or (pf is not None and _unit_stepped(st, position.now, pf)):
            step = True
            stepped = f"the {st['unit'][1]} on {st['unit'][2]}" if st.get("unit") else "a unit"
            break
    fall, made, why = 0, [], ""
    if pf is not None:
        made = facts_after(goal, pf)
        fall = weight_graph(goal, position.now) - weight_graph(goal, pf)
    if not first and not step and fall <= 0:
        return None
    parts = []
    if first:
        parts.append("it is the first move of the line read for the goal")
    if step:
        parts.append(f"it is a step that brings {stepped} to bear")
    if made:
        parts.append("it makes true: " + ", ".join(made))
    if fall > 0 and not made:
        parts.append(f"it takes {fall} stopper(s) out of the way")
    return (first, step, max(0, fall), "; ".join(parts))


def goal_words(g: dict) -> str:
    s, k, p = g["target"]
    unmet = [f["name"] + ("(" + ", ".join(sorted(f["vars"])) + ")" if f["vars"] else "") for f in g["facts"] if f["status"] in ("unmet", "unread")]
    return f"to win the {s} {k} on {p} (stoppers {weight_of(g)}; unmet: {', '.join(unmet) or 'none'})"


def compact(g: dict) -> dict:
    """A goal as plain data (for a log): what the graph tests above need to be asked again at a later position."""
    return {"target": list(g["target"]), "taker": g["taker"], "owner": g["owner"], "stake": g["stake"], "source": g["source"], "weight": weight_of(g),
            "facts": [{"name": f["name"], "status": f["status"], "vars": {k: (list(v) if v else None) for k, v in f["vars"].items()}, "stoppers": [list(u) for u in stoppers_of(f)]} for f in g["facts"]]}


def goal_record(g: dict, age: int, ply: int, since: "int | None" = None) -> dict:
    return {"origin": "goal", "target": tuple(g["target"]), "source": g["source"], "weight": weight_of(g), "unmet": [f["name"] for f in g["facts"] if f["status"] in ("unmet", "unread")],
            "age": age, "since": ply if since is None else since}


def find_held(held: "dict | None", goals: list) -> "dict | None":
    """The goal of this position that the held subgoal is: the same target, else the one unit of that side and kind (the target may have moved)."""
    if not held or held.get("origin") != "goal":
        return None
    t = tuple(held["target"])
    for g in goals:
        if tuple(g["target"]) == t:
            return g
    same = [g for g in goals if tuple(g["target"][:2]) == t[:2]]
    return same[0] if len(same) == 1 else None


# ------------------------------------------------------------------------------------------------------------------------------ threats

def threat_rows(lists: dict) -> list:
    """The rows against us that are a threat now, the highest stake first: CONFIRMED, not a threat in two moves; a forced end of the game before every stake, a row with no stake last."""
    rows = [r for r in lists["mover"] if r.get("verdict") == "CONFIRMED" and not r.get("two_moves")]
    return sorted(rows, key=lambda r: (0 if r.get("goal") else 1, -(r["stake"] if r.get("stake") is not None and r["stake"] != float("inf") else -STAKE_NONE), r["place_ix"] if r.get("place_ix") is not None else -1))


def big_rows(lists: dict) -> dict:
    """{id(row): big}: by the stake ranked within the position, both sides pooled (threat_list.rank_high_of); a forced end of the game is big."""
    both = [r for r in lists["mover"] + lists["other"] if not r.get("goal")]
    rk = TL.rank_high_of(both)
    out = {id(r): bool(r["big"]) if "big" in r else bool(rk.get(id(r), False)) for r in both}   # a caller that read only one side's rows ranks them pooled before it drops the other side's: the flag `big`
    for r in lists["mover"]:
        if r.get("goal"):
            out[id(r)] = True
    return out


def row_words(r: dict) -> str:
    if r.get("goal"):
        return "the end of the game is forced against us"
    st = "no stake" if r.get("stake") is None else f"stake {r['stake']:+.3f}"
    by = ", ".join(f"{k} {p}" for k, p in r.get("by", ())) or "nothing named"
    line = " ".join(r["line"]) if r.get("line") else "(no line)"
    return f"our {r['kind']} on {r['place']} is threatened ({st}; borne on by {by}; the line {line}, {r.get('ended', '')})"


def answers_of(row: dict) -> dict:
    """{move: the kept response (read as a line)} for a row, the first response of each move."""
    out: dict = {}
    for x in row.get("response") or ():
        if x["state"] == "KEPT" and x.get("names"):
            out.setdefault(x["names"][0], x)
    return out


def _order_index(row: dict) -> dict:
    ix: dict = {}
    for k, x in enumerate(row.get("response") or ()):
        if x.get("names"):
            ix.setdefault(x["names"][0], k)
    return ix


def partial_refutation(row: dict, position: Position, move: str) -> int:
    """How far `move` goes toward refuting a threat that no tied move answers, asked of the graph after it: the fall in the number of units of theirs that bear on the threatened unit plus the rise in the number of ours."""
    pf, now = position.after(move), position.now
    if pf is None or row.get("goal") or row.get("place") is None:
        return 0
    us, them = position.roles[position.mover], position.roles[1 - position.mover]
    place = row["place"]
    if place not in pf.at or pf.units[pf.at[place]] != (us, row["kind"], place):
        return 0                                               # the unit moved or was taken: the read did not keep that as an answer, it is not counted as a refutation
    return (len(now.bearers(place, them)) - len(pf.bearers(place, them))) + (len(pf.bearers(place, us)) - len(now.bearers(place, us)))


# ------------------------------------------------------------------------------------------------------------------------------ the rule

def _best_advance(goal: dict, position: Position, tied: list) -> "tuple[str, tuple] | None":
    """The tied move that serves `goal` best: the first move of the line, then a step, then the larger fall of the weight; the search's own order (the order of `tied`) breaks ties."""
    best = None
    for k, m in enumerate(tied):
        a = advance_of(goal, position, m)
        if a is None:
            continue
        key = (a[0], a[1], a[2], -k)
        if best is None or key > best[0]:
            best = (key, m, a)
    return None if best is None else (best[1], best[2])


def choose(position: Position, tied: list, lists: dict, goals: list, held: "dict | None", steps: tuple = (1, 2, 3)) -> tuple:
    assert tied, "the rule is handed the moves the search holds equal"
    ply = len(position.moves)
    ranked = rank_goals(goals)
    hg = find_held(held, ranked)
    age = (held or {}).get("age", 0) + 1 if held else 0
    # ---- the held subgoal of the last move: a goal is kept while its stoppers do not rise; a refutation held is released when its threat is gone
    rows = threat_rows(lists)
    big = big_rows(lists)
    keep_held = None
    drop_note = ""
    if held and held.get("origin") == "goal":
        if hg is None:
            drop_note = "the held goal is gone from the position (its target left or was taken)"
        elif weight_of(hg) > held["weight"]:
            drop_note = f"the held goal's stoppers rose from {held['weight']} to {weight_of(hg)}: dropped"
        else:
            keep_held = hg
    elif held and held.get("origin") == "refute" and not any(r.get("place") == held["target"][2] and r.get("kind") == held["target"][1] for r in rows):
        drop_note = "the threat held as the subgoal is gone"
    elif held and held.get("origin") == "refute":
        pass
    held_now = held if (held and held.get("origin") == "refute" and not drop_note) else None
    pick, reason, new_held = None, None, None
    notes: list = []

    # ---- step 1 and step 3: the threats
    if (1 in steps or 3 in steps) and rows:
        for r in rows:
            ans = answers_of(r)
            cand = [m for m in tied if m in ans]
            if cand and len(cand) == len(tied):
                notes.append(f"the threat on our {r.get('kind')} {r.get('place')} is answered by every tied move")
                continue                                       # every tied move answers it: the response is carried, the goal decides among them
            if cand and 1 in steps:
                order = _order_index(r)
                kept_goal = keep_held
                if kept_goal is not None:
                    stepping = [m for m in cand if advance_of(kept_goal, position, m) is not None]
                else:
                    stepping = []
                pool = stepping or cand
                pick = sorted(pool, key=lambda m: (order.get(m, 99), tied.index(m)))[0]
                x = ans[pick]
                how = x.get("how", "")
                words = f"threat: {row_words(r)}. Answer: {pick} ({'a counter-threat kept by the read' if 'counter' in how else how.split(';')[0]}; after it the taker ends {x.get('ended', 'not ahead')}; the line {' '.join(x['names'])})"
                if stepping:
                    words += f". It is also a step of the held goal {goal_words(kept_goal)}"
                    reason = Reason(words + f". So {pick}.", "1+held", {"row": r, "answer": pick, "goal": kept_goal})
                else:
                    reason = Reason(words + f". So {pick}.", "1", {"row": r, "answer": pick, "answers_tied": cand})
                new_held = goal_record(kept_goal, age, ply, held["since"]) if kept_goal is not None else None
                break
            if not cand and 3 in steps and big.get(id(r), False):
                parts = [(partial_refutation(r, position, m), -k, m) for k, m in enumerate(tied)]
                top = max(parts)
                pick = top[2] if top[0] > 0 else tied[0]
                words = f"threat: {row_words(r)}. No tied move is a kept response, and the threat is big: refuting it is now the subgoal"
                words += (f". {pick} refutes it in part (fewer of theirs bear on it or more of ours do). So {pick}." if top[0] > 0 else f". No tied move refutes it even in part, so the search's own {pick}.")
                reason = Reason(words, "3", {"row": r, "refuted_in_part": top[0] > 0})
                new_held = {"origin": "refute", "target": (position.roles[position.mover], r.get("kind"), r.get("place")), "source": "threat", "weight": 0, "unmet": [], "age": age if held_now else 0, "since": held_now["since"] if held_now else ply}
                break
            notes.append(f"the {'big' if big.get(id(r), False) else 'small'} threat on our {r.get('kind')} {r.get('place')} has no answer among the tied moves and is left standing")

    # ---- step 2: advance the subgoal
    if pick is None and 2 in steps and ranked:
        order = []
        if keep_held is not None:
            order.append(("held", keep_held))
        for g in ranked:
            if keep_held is None or g is not keep_held:
                order.append(("other" if keep_held is not None else "first", g))
        for kind, g in order:
            b = _best_advance(g, position, tied)
            if b is None:
                continue
            pick, a = b
            skipped = ranked.index(g) if kind != "held" else 0
            br = {"held": "2 held", "other": "2 other"}.get(kind, "2 first" if skipped == 0 else "2 next")
            lead = ("; ".join(notes) + ". ") if notes else (drop_note + ". " if drop_note else "")
            pre = "No move is forced. "
            if kind == "held":
                words = f"{lead}{pre}The held goal {goal_words(g)} is kept; {pick} advances it: {a[3]}. So {pick}."
            elif kind == "other":
                words = f"{lead}{pre}The held goal had no step among the tied moves; the goal {goal_words(g)} is taken up; {pick} advances it: {a[3]}. So {pick}."
            elif skipped == 0:
                words = f"{lead}{pre}No goal is held; the first-ranked goal {goal_words(g)} is taken up; {pick} advances it: {a[3]}. So {pick}."
            else:
                words = f"{lead}{pre}No goal is held; the {skipped} goal(s) ranked before {goal_words(g)} have no step among the tied moves, so this one is taken up; {pick} advances it: {a[3]}. So {pick}."
            reason = Reason(words, br, {"goal": g, "advance": a})
            new_held = goal_record(g, age if kind == "held" else 0, ply, held["since"] if kind == "held" else None)
            break

    # ---- step 4: nothing applies
    if pick is None:
        pick = tied[0]
        why = ("no threat to answer and no goal that a tied move advances" if rows == [] else "the threats are answered by every tied move or left standing, and no goal that a tied move advances")
        if not ranked and 2 in steps:
            why = "no threat to answer and no goal"
        reason = Reason(f"{drop_note + '. ' if drop_note else ''}Nothing applies ({why}): the search's own choice {pick}.", "4", {})
        new_held = held_now if held_now is not None else (goal_record(keep_held, age, ply, held["since"]) if keep_held is not None else None)
    assert pick in tied, "the rule orders what the search holds equal; it never leaves the set"
    return pick, reason, new_held
