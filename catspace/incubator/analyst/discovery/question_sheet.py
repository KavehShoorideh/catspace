"""PLAN 78.41 stage 1 THE QUESTION SHEET (a training-side measurement tool; the wake does not use it).

From a position, every unit of both sides is a TARGET with a STAKE: the unit the goal clause names has the top stake, the rest are ordered by the judge's worth of their kind. A
QUESTION is (target, asking side = the side that does not own the target): "can this target be taken or confined, and by what". The FIRST ORDER alternates the sides down the
stakes: mine at the top stake, theirs at the top stake, theirs at the next, mine at the next, mine, theirs, ... ("mine" is the side to move).

Each question is deepened one LEVEL at a time by `step(question, reading) -> Finding` (what it found, what it cost, and whether the question is now closed, idle or live):
  level 0   the board as it stands: what strikes the target, what guards it, how many of its exits are open. The goal unit is read by the scenario report (its holes and covered
            places); any other target by the fact view (the strike relation of the description and the units' places).
  level 1   the asking side's moves alone: which of its units could come to strike the target or cover an exit, in how many moves. The goal unit: the scenario report's supply and
            scenarios; any other target: the move relations (go, go_within2, go_within3) of the fact view, which hold for a unit standing on ANY place, so a hypothetical placement
            is read from the table. No supplier closes the question, with its reason.
  level 2+  with the reply: the existing exact tactics census of the asking side at depth = level (the ledger search over that side's lines), kept to the lines whose targets include
            this one. The API cannot restrict a search to named root moves, so the census is run for the whole side once per (position, side, depth) and shared by the questions.
  quiet     offered only when every other question is closed or idle: the unit of mine that does least (fewest strikes and places to go).

`run(position, worth, ...)` is the anytime schedule: all level-0 steps first, in the first order; then the next step goes to the open question with the highest
priority = stake * urgency / expected cost of its next step; urgency halves when a step finds nothing new and doubles when it finds a live plan (named parameters in `Params`).
The `Sheet` records the steps taken and the final state of every question; `explain(sheet)` prints it in plain words.

Nothing here names a game's unit: a position is an opaque string the kernel accepts, units are (side, kind, place) read from the fact view, the game's own words come from its
description and its position codec (`curriculum.codec_of`). The game name is the command's argument.

  python -m catspace.incubator.analyst.discovery.question_sheet <game> --games-dir DIR [--max-steps N] [--seconds S] [--limit N] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cs_engine as CE

from catspace.io import paths

SELF, OTHER = "us", "them"             # the sides as the fact view names them in the mover's perspective
LIVE, IDLE, CLOSED = "live", "idle", "closed"
BIG_VALUE = 10 ** 9                    # a ledger value at or above this is a line that reaches the goal
UNIT_RELATION = "cell"                 # the description's relation that holds a unit: its last two arguments are the unit's side and kind, the rest its place
SCENARIO_VOCABULARY = "scenario"       # the store's vocabulary that names the move relations and composes them onto the description


@dataclass
class Params:
    """The named numbers of the schedule and of the levels."""
    urgency_start: float = 1.0
    urgency_down: float = 0.5          # a step that finds nothing new
    urgency_up: float = 2.0            # a step that finds a live plan
    reach_k: int = 2                   # moves of its own a supplying unit may use (level 1); the description's move relations go to 3
    max_level: int = 4                 # the deepest level; a question that finds nothing there is closed
    census_fuse: int = 2_000_000
    goal_stake_multiple: float = 2.0   # the goal unit's stake: this many times the largest worth
    min_stake: float = 0.1             # no unit's stake is lower (a stake of nothing would never be asked)
    quiet_stake_share: float = 0.5     # the quiet question's stake: this share of the lowest stake
    cost_prior: tuple = (0.002, 0.01, 0.01)    # seconds expected for level 0, 1, 2 before any step of that level has been seen
    cost_growth: float = 3.0           # expected cost of a level when only the one below has been seen: that times the one below


# ---------------------------------------------------------------------------------------------------------------------------------------------- units and the reading

@dataclass(frozen=True)
class Unit:
    side: str                          # SELF or OTHER
    kind: str                          # the description's own word
    at: tuple                          # the place, as the description writes it


def place_text(at: tuple) -> str:
    return "".join(at)


def unit_words(u: Unit) -> str:
    return f"{'my' if u.side == SELF else 'their'} {u.kind} on {place_text(u.at)}"


def side_word(side: str) -> str:
    return "mine" if side == SELF else "theirs"


class Env:
    """What a game needs to be read: its description, its fact vocabulary with the move relations, the names of the relations by the roles of their arguments."""

    def __init__(self, game: str, variant: "str | None" = None, vocabulary: str = SCENARIO_VOCABULARY):
        from catspace.incubator.analyst.discovery import curriculum as CU
        from catspace.incubator.analyst.discovery import game_store as GS
        self.game, self.variant = game, variant or game
        raw = GS.vocabulary_file(game, vocabulary)
        self.relations = raw["scenario"]
        self.vocabulary = {**GS.vocabulary(game, vocabulary), "scenario": raw["scenario"]}
        self.vocabulary_json = json.dumps(self.vocabulary)
        self.spec = GS.spec_texts(game, vocabulary)[self.variant]
        self.codec = CU.codec_of(game)


def _rows(facts: list, spec: dict) -> dict:
    """{(kind, owner, from place): [to place, ...]} of one relation named by its argument roles."""
    out: dict = {}
    for f in facts:
        if f[0] != spec["relation"]:
            continue
        a = f[1:]
        out.setdefault((a[spec["kind"]], a[spec["owner"]], tuple(a[i] for i in spec["from"])), []).append(tuple(a[i] for i in spec["to"]))
    return out


def kind_keys(codec, position: str) -> dict:
    """{kind as the description names it: kind as the kernel's lines and the worth table name it}, from the codec's `describe` and `owned`. The two lists hold the same units
    but `owned` need not keep `describe`'s order, so each alignment (as given, reversed) is tried, and the one whose sides agree unit by unit and whose pairing is a function wins."""
    owned, described = list(codec.owned(position)), list(codec.describe(position))
    for seq in (owned, owned[::-1]):
        pairs = list(zip(seq, described))
        if all(o[0] == d[0] for o, d in pairs):
            mapping: dict = {}
            if all(mapping.setdefault(d[1], o[1]) == o[1] for o, d in pairs):
                return mapping
    return {}


class Reading:
    """One position read once: its units, the strike table, the move tables, and the shared answers of the engine calls that several questions use."""

    def __init__(self, env: Env, position: str, worth: dict):
        self.env, self.position, self.worth = env, position, dict(worth)
        t0 = time.perf_counter()
        state = CE.state(env.variant, position, [])
        if state[3] is not None or not state[2]:
            raise ValueError("the position is over or has no legal moves")
        self.mover = state[1]
        facts = CE.facts(env.variant, env.spec, env.vocabulary_json, position, [], True)
        rel = env.relations
        self.units = sorted((Unit(f[-2], f[-1], tuple(f[1:-2])) for f in facts if f[0] == UNIT_RELATION), key=lambda u: (u.side != SELF, u.kind, u.at))
        self.acts = _rows(facts, rel["acts_on"])
        self.go = [_rows(facts, spec) for spec in rel["can_go"]]
        self.kind_key = kind_keys(env.codec, position)
        self.read_seconds = time.perf_counter() - t0
        self._scenario: dict = {}
        self._census: dict = {}
        self._goal_kind: "tuple | None" = None

    # ---- stakes
    def worth_of(self, kind: str) -> float:
        return float(self.worth.get(kind, self.worth.get(self.kind_key.get(kind, kind), 0)))

    def goal_kind(self) -> "str | None":
        """The kind the goal clause names (read by the scenario report from either side), None when the description does not name one clear target."""
        if self._goal_kind is None:
            kind = None
            for role in ("mover", "other"):
                r = self.scenario(role)
                if r.get("target"):
                    kind = r["target"]["kind"]
                    break
            self._goal_kind = (kind,)
        return self._goal_kind[0]

    def goal_unit(self, side: str) -> "Unit | None":
        k = self.goal_kind()
        if k is None:
            return None
        found = [u for u in self.units if u.side == side and u.kind == k]
        return found[0] if len(found) == 1 else None

    # ---- shared engine answers
    def scenario(self, role: str) -> dict:
        """The scenario report with `role` ("mover" or "other") as the side that wants the goal (cached)."""
        if role not in self._scenario:
            self._scenario[role] = CE.scenario(self.env.variant, self.env.spec, self.env.vocabulary_json, self.position, [], role=role, k=self.params_k)
        return self._scenario[role]

    params_k = 2

    def census(self, side: str, depth: int, fuse: int) -> list:
        """The tactics of the side (SELF: the side to move; OTHER: the other side, as if it were its turn) at `depth` (cached)."""
        key = (side, depth)
        if key not in self._census:
            tactics, _stats = CE.census(self.env.variant, self.position, [], 1, depth, fuse, "mover" if side == SELF else "other", self.worth)
            self._census[key] = tactics
        return self._census[key]

    # ---- the tables
    def strikes_of(self, u: Unit) -> set:
        return set(self.acts.get((u.kind, u.side, u.at), ()))

    def strikers_of(self, target: Unit, side: str) -> list:
        return [u for u in self.units if u.side == side and u != target and target.at in self.strikes_of(u)]

    def exits_of(self, target: Unit) -> set:
        return set(self.go[0].get((target.kind, target.side, target.at), ())) if self.go else set()

    def struck_by(self, side: str) -> set:
        out: set = set()
        for u in self.units:
            if u.side == side:
                out |= self.strikes_of(u)
        return out


# ---------------------------------------------------------------------------------------------------------------------------------------------- questions

@dataclass
class Question:
    target: "Unit | None"
    asker: str
    stake: float
    rank: int                          # 1-based place in the first order
    goal: bool = False
    quiet: bool = False
    level: int = 0                     # the next level to ask
    state: str = "open"                # open or closed
    status: str = IDLE                 # what the last step found: live, idle or closed
    urgency: float = 1.0
    spent: float = 0.0
    live_by_level1: bool = False       # a step of level 0 or 1 found something live
    closed_at: "int | None" = None
    reason: str = ""
    best_value: "float | None" = None
    findings: list = field(default_factory=list)

    @property
    def owner(self) -> str:
        return OTHER if self.asker == SELF else SELF

    def label(self) -> str:
        return "the quiet question" if self.quiet else f"{unit_words(self.target)}{' (the goal unit)' if self.goal else ''}, asked by {side_word(self.asker)}"


@dataclass
class Finding:
    question: Question
    level: int
    status: str                        # live, idle or closed
    text: str                          # in plain words
    cost: float                        # seconds
    new: bool = False                  # something not found before
    value: "float | None" = None       # the best line's value in the judge's units (levels 2 and on)
    line: tuple = ()
    detail: dict = field(default_factory=dict)


def stakes(reading: Reading, params: Params) -> dict:
    """{unit: stake}: the goal unit of each side the top, the rest by worth."""
    top = max([reading.worth_of(u.kind) for u in reading.units] + [1.0]) * params.goal_stake_multiple
    out = {}
    goal = {reading.goal_unit(SELF), reading.goal_unit(OTHER)} - {None}
    for u in reading.units:
        out[u] = top if u in goal else max(reading.worth_of(u.kind), params.min_stake)
    return out


def first_order(reading: Reading, params: Params = Params()) -> list:
    """The questions in the first order: my unit at the top stake, theirs at the top stake, theirs at the next, mine at the next, then mine, theirs, theirs, mine, ..."""
    st = stakes(reading, params)
    mine = sorted([u for u in reading.units if u.side == SELF], key=lambda u: (-st[u], u.kind, u.at))
    theirs = sorted([u for u in reading.units if u.side == OTHER], key=lambda u: (-st[u], u.kind, u.at))
    order: list = []
    for i in range(max(len(mine), len(theirs))):
        pair = [mine[i:i + 1], theirs[i:i + 1]]
        if i % 2 == 1:
            pair.reverse()
        order += pair[0] + pair[1]
    goals = {reading.goal_unit(SELF), reading.goal_unit(OTHER)} - {None}
    return [Question(u, OTHER if u.side == SELF else SELF, st[u], n + 1, goal=u in goals, urgency=params.urgency_start) for n, u in enumerate(order)]


def quiet_question(reading: Reading, rank: int, params: Params, lowest: float) -> "Question | None":
    if not any(u.side == SELF for u in reading.units):
        return None
    return Question(None, SELF, max(lowest * params.quiet_stake_share, 1e-6), rank, quiet=True, urgency=params.urgency_start)


# ---------------------------------------------------------------------------------------------------------------------------------------------- the levels

def _units_at(reading: Reading, side: str, raw: list) -> list:
    """The units of a scenario report's `by` lists ({kind, at "a:b"}) as Units."""
    out = []
    for r in raw:
        at = tuple(r["at"].split(":"))
        out += [u for u in reading.units if u.side == side and u.kind == r["kind"] and u.at == at]
    return out


def _names(units: list) -> str:
    return ", ".join(unit_words(u) for u in units) if units else "none"


def _role_of(asker: str) -> str:
    return "mover" if asker == SELF else "other"


def level0(q: Question, r: Reading, params: Params) -> tuple:
    t = q.target
    guards = [u for u in r.units if u.side == t.side and u != t and t.at in r.strikes_of(u)]
    if q.goal:
        rep = r.scenario(_role_of(q.asker))
        if rep.get("target") and not rep.get("unreadable"):
            own = next((c for c in rep["covered"] if c["own"]), None)
            strikers = _units_at(r, q.asker, own["by"]) if own else []
            open_exits = sum(1 for h in rep["holes"] if h["kind"] == "exit")
            total = open_exits + sum(1 for c in rep["covered"] if not c["own"])
            live = bool(strikers) or bool(rep.get("goal_holds"))
            text = f"struck by {_names(strikers)}; guarded by {_names(guards)}; {open_exits} of its {total} exits open" + ("; the goal already holds" if rep.get("goal_holds") else "")
            return (LIVE if live else IDLE), text, {"strikers": strikers, "guards": guards, "open_exits": open_exits, "exits": total}
    strikers = r.strikers_of(t, q.asker)
    exits = r.exits_of(t)
    covered = r.struck_by(q.asker)
    open_exits = len(exits - covered)
    cheapest = min((r.worth_of(u.kind) for u in strikers), default=None)
    live = bool(strikers) and (q.goal or not guards or cheapest < r.worth_of(t.kind))
    why = ""
    if strikers and not q.goal:
        why = "; unguarded" if not guards else ("; struck by a cheaper unit" if live else "; guarded")
    text = f"struck by {_names(strikers)}; guarded by {_names(guards)}; {open_exits} of its {len(exits)} exits open{why}"
    return (LIVE if live else IDLE), text, {"strikers": strikers, "guards": guards, "open_exits": open_exits, "exits": len(exits)}


def level1(q: Question, r: Reading, params: Params) -> tuple:
    t, k = q.target, params.reach_k
    side = side_word(q.asker)
    if q.goal:
        rep = r.scenario(_role_of(q.asker))
        if rep.get("target") and not rep.get("unreadable"):
            if rep.get("goal_holds"):
                return LIVE, "the goal already holds", {"scenarios": 0}
            if rep["safe"] is True:
                return CLOSED, f"no unit of {side} can reach it within {k} moves", {"scenarios": 0}
            if rep["scenarios"]:
                best = min(rep["scenarios"], key=lambda s: s["moves"])
                plan = "; ".join(f"{p['kind']} from {p['from'].replace(':', '')} to {p['to'].replace(':', '')} in {p['moves']}" for p in best["placements"])
                return LIVE, f"a plan of {best['moves']} moves of {side}: {plan} ({len(rep['scenarios'])} plans)", {"scenarios": len(rep["scenarios"]), "moves": best["moves"]}
            return IDLE, f"the supply was not fully read (caps hit); {side} not shown to reach it", {"scenarios": 0}
    now = r.strikers_of(t, q.asker)
    exits = r.exits_of(t)
    open_exits = exits - r.struck_by(q.asker)
    reach_tables = r.go[:min(k, len(r.go))]
    supply: list = []
    for u in r.units:
        if u.side != q.asker or u in now:
            continue
        seen: set = set()
        for j, table in enumerate(reach_tables, start=1):
            for to in table.get((u.kind, u.side, u.at), ()):
                if to in seen:
                    continue
                seen.add(to)
                hit = r.acts.get((u.kind, u.side, to), ())
                if t.at in hit:
                    supply.append((j, u, to, "strike"))
                elif open_exits & set(hit):
                    supply.append((j, u, to, "cover an exit"))
    supply.sort(key=lambda x: (x[0], x[1].kind, x[1].at, x[2]))
    if not supply and not now:
        return CLOSED, f"no unit of {side} can reach it within {k} moves", {"supply": []}
    if not supply:
        return IDLE, f"already struck by {_names(now)}; no other unit of {side} can join within {k} moves", {"supply": []}
    shown = "; ".join(f"{unit_words(u)} could {what} from {place_text(to)} in {j}" for j, u, to, what in supply[:4])
    return LIVE, f"{len(supply)} ways for {side} to join within {k} moves: {shown}", {"supply": supply}


def _matches(q: Question, r: Reading, tactic: dict) -> bool:
    if q.goal and "goal" in tactic["targets"]:
        return True
    return f"{r.kind_key.get(q.target.kind, q.target.kind)}@{place_text(q.target.at)}" in tactic["targets"]


def level_deep(q: Question, r: Reading, params: Params) -> tuple:
    depth = q.level
    tactics = [x for x in r.census(q.asker, depth, params.census_fuse) if _matches(q, r, x)]
    side = side_word(q.asker)
    if not tactics:
        if q.level >= params.max_level:
            return CLOSED, f"no line of {side} within {depth} plies takes it", {"depth": depth, "tactics": 0}
        return IDLE, f"no line of {side} within {depth} plies takes it", {"depth": depth, "tactics": 0}
    best = max(tactics, key=lambda x: (x["value"], -len(x["line"])))
    value = best["value"]
    shown = "reaches the goal" if value >= BIG_VALUE else f"value {value}"
    free = f", after a free move {best['free'][0]}" if best["free"] else ""
    return LIVE, f"best line of {side} within {depth} plies{free}: {' '.join(best['line'])}, {shown} ({len(tactics)} lines take it)", {"depth": depth, "tactics": len(tactics), "value": value, "line": tuple(best["line"])}


def level_quiet(q: Question, r: Reading, params: Params) -> tuple:
    rows = []
    for u in r.units:
        if u.side == SELF:
            rows.append((len(r.strikes_of(u)) + len(r.go[0].get((u.kind, u.side, u.at), ()) if r.go else ()), -r.worth_of(u.kind), u.kind, u.at, u))
    rows.sort(key=lambda x: x[:4])
    least = rows[0][4]
    strikes, places = len(r.strikes_of(least)), len(r.go[0].get((least.kind, least.side, least.at), ()) if r.go else ())
    return CLOSED, f"{unit_words(least)} does least: it strikes {strikes} places and can go to {places}", {"unit": least, "strikes": strikes, "places": places}


def step(q: Question, r: Reading, params: Params = Params()) -> Finding:
    """Deepen the question by one level: what it found, what it cost, and whether it is now closed (with the reason in plain words), idle or live."""
    r.params_k = params.reach_k
    level = q.level
    t0 = time.perf_counter()
    if q.quiet:
        status, text, detail = level_quiet(q, r, params)
    elif level == 0:
        status, text, detail = level0(q, r, params)
    elif level == 1:
        status, text, detail = level1(q, r, params)
    else:
        status, text, detail = level_deep(q, r, params)
    cost = time.perf_counter() - t0
    value = detail.get("value")
    new = status == LIVE and (value is None or q.best_value is None or value > q.best_value)
    f = Finding(q, level, status, text, cost, new, value, detail.get("line", ()), detail)
    q.findings.append(f)
    q.spent += cost
    q.level += 1
    q.status = status
    if value is not None and (q.best_value is None or value > q.best_value):
        q.best_value = value
    if level <= 1 and status == LIVE:
        q.live_by_level1 = True
    if status == CLOSED:
        q.state, q.closed_at, q.reason = "closed", level, text
    return f


# ---------------------------------------------------------------------------------------------------------------------------------------------- the schedule

@dataclass
class Step:
    question: Question
    level: int
    status: str
    finding: str
    cost: float


@dataclass
class Sheet:
    position: str
    questions: list
    steps: list = field(default_factory=list)
    seconds: float = 0.0
    read_seconds: float = 0.0
    reading: "Reading | None" = None


class _Costs:
    """The expected cost of a question's next step: the mean seen at that level, else the level below times a growth, else a prior."""

    def __init__(self, params: Params):
        self.p, self.sum, self.n = params, {}, {}

    def add(self, level: int, cost: float) -> None:
        self.sum[level] = self.sum.get(level, 0.0) + cost
        self.n[level] = self.n.get(level, 0) + 1

    def expect(self, level: int) -> float:
        if self.n.get(level):
            return max(self.sum[level] / self.n[level], 1e-6)
        if self.n.get(level - 1):
            return max(self.sum[level - 1] / self.n[level - 1] * self.p.cost_growth, 1e-6)
        pr = self.p.cost_prior
        return pr[min(level, len(pr) - 1)] * (self.p.cost_growth ** max(0, level - len(pr) + 1))


def priority(q: Question, costs: _Costs) -> float:
    return q.stake * q.urgency / costs.expect(q.level)


def _record(sheet: Sheet, f: Finding, costs: _Costs, params: Params) -> None:
    q = f.question
    sheet.steps.append(Step(q, f.level, f.status, f.text, f.cost))
    costs.add(f.level if not q.quiet else 0, f.cost)
    if f.status == LIVE and f.new:
        q.urgency *= params.urgency_up
    elif f.status != CLOSED:
        q.urgency *= params.urgency_down
    if q.state == "open" and not q.quiet and q.level > params.max_level:
        q.state, q.closed_at, q.reason = "closed", f.level, "asked to the deepest level"


def run(position: str, worth: dict, *, max_steps: "int | None" = None, budget_seconds: "float | None" = None, env: "Env | None" = None, game: "str | None" = None,
        params: Params = Params()) -> Sheet:
    """The anytime schedule over the position's questions (see the module). Stops at `max_steps` steps or `budget_seconds` (default 40 steps when neither is given)."""
    if env is None:
        if game is None:
            raise ValueError("run needs an Env or a game name")
        env = Env(game)
    t0 = time.perf_counter()
    r = Reading(env, position, worth)
    r.params_k = params.reach_k
    questions = first_order(r, params)
    sheet = Sheet(position, questions, read_seconds=r.read_seconds, reading=r)
    if max_steps is None and budget_seconds is None:
        max_steps = 40
    costs = _Costs(params)

    def over() -> bool:
        return (max_steps is not None and len(sheet.steps) >= max_steps) or (budget_seconds is not None and time.perf_counter() - t0 >= budget_seconds)

    for q in questions:                                      # every question's level 0 first, in the first order
        if over():
            break
        _record(sheet, step(q, r, params), costs, params)
    quiet: "Question | None" = None
    lowest = min((q.stake for q in questions), default=params.min_stake)
    while not over():
        if quiet is None and questions and all(q.state == "closed" or q.status == IDLE for q in questions):
            quiet = quiet_question(r, len(questions) + 1, params, lowest)
            if quiet is not None:
                sheet.questions.append(quiet)
        pool = [q for q in sheet.questions if q.state == "open"]
        if not pool:
            break
        q = max(pool, key=lambda x: (priority(x, costs), -x.rank))
        _record(sheet, step(q, r, params), costs, params)
    sheet.seconds = time.perf_counter() - t0
    return sheet


def finish_levels(sheet: Sheet, upto: int = 1, params: Params = Params()) -> None:
    """Step every open question that has not yet been asked up to level `upto` (the measures read levels 0 and 1 of all questions)."""
    costs = _Costs(params)
    for q in sheet.questions:
        while not q.quiet and q.state == "open" and q.level <= upto:
            _record(sheet, step(q, sheet.reading, params), costs, params)


def explain(sheet: Sheet) -> list:
    """The sheet in plain words: the steps in the order taken, then where every question stands."""
    out = [f"{len(sheet.questions)} questions in the first order; {len(sheet.steps)} steps taken in {sheet.seconds:.3f} s"]
    words = {0: "the board as it stands", 1: "its moves alone"}
    for n, s in enumerate(sheet.steps, start=1):
        what = "" if s.question.quiet else ", " + words.get(s.level, f"with the reply, {s.level} plies deep")
        out.append(f"step {n}: {s.question.label()}{what}: {s.finding} -> {s.status} ({s.cost * 1000:.1f} ms)")
    out.append("where the questions stand:")
    for q in sheet.questions:
        if q.state == "closed":
            how = f"closed: {q.reason}"
        else:
            how = f"{q.status}, asked to level {q.level - 1}" + (f", best value {q.best_value}" if q.best_value is not None else "")
        out.append(f"  {q.rank}. {q.label()}: {how}")
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------- the measures

@dataclass
class Row:
    """What the sheet found for one question of the first order, as the measures read it."""
    owner: str                         # the side that owns the target
    live: bool                         # a step of level 0 or 1 found it live (this is "open")
    closed1: bool                      # closed by level 1
    live0: bool = False                # live at level 0 (struck now and not safely guarded)
    near: bool = False                 # live at level 0, or a plan of one move at level 1


@dataclass
class Record:
    key: tuple                         # (game, ply)
    bad: bool                          # a bad move was played here
    ranks: list                        # one Row per question in the first order
    fell: "bool | None"                # the mover's unit balance fell within 4 plies (None: not read)
    costs: list                        # (level, seconds) per step
    text: list = field(default_factory=list)


def summarize_sheet(key: tuple, bad: bool, fell: "bool | None", sheet: Sheet) -> Record:
    ranks = []
    for q in sheet.questions:
        if q.quiet:
            continue
        f = {x.level: x for x in q.findings}
        live0 = 0 in f and f[0].status == LIVE
        one = 1 in f and f[1].status == LIVE and (f[1].detail.get("moves") == 1 or any(x[0] == 1 for x in f[1].detail.get("supply", ())))
        ranks.append(Row(q.owner, q.live_by_level1, q.closed_at == 1 and q.state == "closed", live0, live0 or one))
    return Record(key, bad, ranks, fell, [(s.level, s.cost) for s in sheet.steps if not s.question.quiet])


def load_positions(directory: str, variant: str, limit: "int | None" = None) -> list:
    """[(key, position, bad, fell)] from a surprise meter's directory (games.jsonl, surprises.jsonl). A position is a game's ply that has a reading of ours; a bad move is a
    MISSED surprise whose judge jump is below 0. `fell`: the mover's balance (by ply, from the games file) was lower at one of the next 4 plies, the other side's read negated."""
    d = Path(directory)
    games = {}
    for line in (d / "games.jsonl").read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            games[g["game"]] = g
    bad = set()
    for line in (d / "surprises.jsonl").read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            if s.get("kind") == "MISSED" and (s.get("judge") or {}).get("jump", 0) < 0:
                bad.add((s["game"], s["ply"]))
    out = []
    for gid in sorted(games):
        g = games[gid]
        ours = {int(k): v for k, v in g.get("ours", {}).items()}
        for ply in sorted(ours):
            if ply >= len(g["moves"]):
                continue
            here = ours[ply].get("balance")
            later = [(j, ours[ply + j].get("balance")) for j in range(1, 5) if ply + j in ours and ours[ply + j].get("balance") is not None]
            fell = None if here is None or not later else any((b if j % 2 == 0 else -b) < here for j, b in later)
            out.append(((gid, ply), (g.get("variant") or variant, g.get("start"), g["moves"][:ply]), (gid, ply) in bad, fell))
    if limit is not None and limit < len(out):
        stride = len(out) / limit
        out = [out[int(i * stride)] for i in range(limit)]
    return out


def _pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def _open_lines(records: list, label: str, pick, n: int, bad: int) -> list:
    opened = [r for r in records if any(pick(x) for x in r.ranks)]
    none = [r for r in records if not any(pick(x) for x in r.ranks)]
    ob, nb = sum(r.bad for r in opened), sum(r.bad for r in none)
    return [f"OBS (a) [{label}] no question open: {len(none)} of {n} positions ({_pct(len(none), n)}); bad move follows {_pct(ob, len(opened))} when at least one is open "
            f"({len(opened)} positions), {_pct(nb, len(none))} when none is ({len(none)}); the none-open positions hold {_pct(nb, bad)} of the bad moves"]


def format_obs(records: list, unreadable: int = 0) -> list:
    """The OBS lines of the measures (a) and (b) and the costs (c)."""
    n = len(records)
    bad = sum(r.bad for r in records)
    out = [f"OBS question sheet: positions read {n} (unreadable {unreadable}); bad moves {bad} ({_pct(bad, n)} of positions)"]
    out += _open_lines(records, "open = a step of level 0 or 1 found it live", lambda x: x.live, n, bad)
    out += _open_lines(records, "open = not closed by level 1", lambda x: not x.closed1, n, bad)
    out += _open_lines(records, "open = live at level 0", lambda x: x.live0, n, bad)
    out += _open_lines(records, "open = live at level 0 or a plan of one move at level 1", lambda x: x.near, n, bad)
    depth = max((len(r.ranks) for r in records), default=0)
    for k in range(min(depth, 8)):
        yes = [r for r in records if len(r.ranks) > k and r.ranks[k].live]
        no = [r for r in records if len(r.ranks) > k and not r.ranks[k].live]
        y0 = [r for r in records if len(r.ranks) > k and r.ranks[k].live0]
        n0 = [r for r in records if len(r.ranks) > k and not r.ranks[k].live0]
        side = next((r.ranks[k].owner for r in records if len(r.ranks) > k), SELF)
        out.append(f"OBS (a) rank {k + 1} (the unit asked about is {'mine' if side == SELF else 'theirs'} in the first position): open (level 0 or 1) in {_pct(len(yes), len(yes) + len(no))} of positions; "
                   f"bad move follows {_pct(sum(r.bad for r in yes), len(yes))} when open, {_pct(sum(r.bad for r in no), len(no))} when not; open ones hold {_pct(sum(r.bad for r in yes), bad)} of the bad moves; "
                   f"live at level 0 in {_pct(len(y0), len(y0) + len(n0))}, bad move follows {_pct(sum(r.bad for r in y0), len(y0))} when live, {_pct(sum(r.bad for r in n0), len(n0))} when not")
    mine_closed = [r for r in records if any(x.owner == SELF and x.closed1 for x in r.ranks)]
    read = [r for r in records if r.fell is not None]
    mc = [r for r in mine_closed if r.fell is not None]
    all_safe = [r for r in read if any(x.owner == SELF for x in r.ranks) and all(x.closed1 for x in r.ranks if x.owner == SELF)]
    out.append(f"OBS (b) level 1 closed on a unit of mine in {len(mine_closed)} positions ({_pct(len(mine_closed), n)}); of those with the balance read ({len(mc)}) the mover's balance fell within 4 plies in "
               f"{sum(r.fell for r in mc)} ({_pct(sum(r.fell for r in mc), len(mc))}); in all positions with the balance read ({len(read)}): {sum(r.fell for r in read)} ({_pct(sum(r.fell for r in read), len(read))})")
    out.append(f"OBS (b) all my units closed at level 1 in {len(all_safe)} positions; the balance fell within 4 plies in {sum(r.fell for r in all_safe)} ({_pct(sum(r.fell for r in all_safe), len(all_safe))})")
    by: dict = {}
    for r in records:
        for level, cost in r.costs:
            by.setdefault(level, []).append(cost)
    for level in sorted(by):
        c = by[level]
        out.append(f"OBS (c) level {level}: {len(c)} steps, mean {1000 * sum(c) / len(c):.2f} ms, max {1000 * max(c):.2f} ms")
    return out


def measure(items: list, runner, samples: int = 3) -> tuple:
    """items from `load_positions`; `runner(variant, start, moves) -> (Sheet or None)`. -> (records, sample explanation lines, number unreadable)."""
    records, unreadable, picked = [], 0, []
    for key, pos, bad, fell in items:
        try:
            sheet = runner(*pos)
        except ValueError:
            sheet = None
        if sheet is None:
            unreadable += 1
            continue
        rec = summarize_sheet(key, bad, fell, sheet)
        records.append(rec)
        if len(picked) < samples and (any(x.live for x in rec.ranks) or len(records) % 50 == 0):
            picked.append((key, bad, sheet))
    text = []
    for key, bad, sheet in picked:
        text.append(f"OBS sample game {key[0]} ply {key[1]}{' (a bad move followed)' if bad else ''}:")
        text += [f"OBS   {ln}" for ln in explain(sheet)]
    return records, text, unreadable


def store_worth(game: str) -> dict:
    """The worth table of the game's HEAD snapshot (its judge), empty when there is none."""
    from catspace.incubator.analyst.discovery import game_store as GS
    h = GS.head(game)
    if not h or not GS.path_of(game, h).is_file():
        return {}
    return (json.loads(GS.path_of(game, h).read_bytes()).get("judge") or {}).get("worth", {})


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="a name under the derived data directory to write the OBS lines to")
    a = ap.parse_args(argv)
    env = Env(a.game)
    worth = store_worth(a.game)
    items = load_positions(a.games_dir, a.game, a.limit)
    params = Params()
    t0 = time.perf_counter()

    def runner(variant, start, moves):
        position = CE.state(variant, start, list(moves))[0]
        sheet = run(position, worth, max_steps=a.max_steps, budget_seconds=a.seconds, env=env, params=params)
        finish_levels(sheet, 1, params)
        return sheet

    records, text, unreadable = measure(items, runner)
    lines = format_obs(records, unreadable) + text
    lines.append(f"OBS question sheet: {time.perf_counter() - t0:.1f} s for {len(items)} positions, worth {worth}")
    for ln in lines:
        print(ln)
    if a.out:
        out = Path(paths.derived(a.out))
        out.mkdir(parents=True, exist_ok=True)
        (out / "question_sheet_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
