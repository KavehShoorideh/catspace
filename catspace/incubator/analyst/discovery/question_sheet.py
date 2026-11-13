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
from typing import NamedTuple
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.frame_search import FrameSearch

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
    search: bool = True                # level 2 on: when the asker is to move, play the first move of each way and settle it by the reasoning search on the child (False: the census)
    max_ways: int = 4                  # ways settled per step
    reason_fuse: int = 50_000          # nodes the reasoning search may use on one child
    search_depth_cap: int = 3          # the search is not deepened past this many plies (a deeper level reads the cached answers)
    links: bool = True                 # level 1 as a chain of three links (reach, a place to stand, a strike that gains); False: reach alone, the first version
    max_open_exits: int = 1            # the goal unit, no full plan read: live only when at most this many of its exits are open
    cost_growth: float = 3.0           # expected cost of a level when only the one below has been seen: that times the one below
    frames: bool = True                # level 2 of a non-goal unit by the exchange frame (False: the play-and-reply search or the census, the earlier forms)
    frame_budget: int = 20_000         # nodes one frame search may expand
    passes: bool = True                # the asker not to move: hand it the move by the kernel's pass and run the frames by real moves (False: the static form, for comparison)
    goal_near: bool = True             # the goal unit is live at level 1 only when open exits are at most max_open_exits AND an asker unit is near (False: the earlier rule)


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
    """{kind as the description names it: kind as the kernel's lines and the worth table name it}: the codec's `describe` and `owned` list the same units in the same order."""
    return {d[1]: o[1] for o, d in zip(codec.owned(position), codec.describe(position))}


class Reading:
    """One position read once: its units, the strike table, the move tables, and the shared answers of the engine calls that several questions use."""

    def __init__(self, env: Env, position: str, worth: dict):
        self.env, self.position, self.worth = env, position, dict(worth)
        t0 = time.perf_counter()
        state = CE.state(env.variant, position, [])
        if state[3] is not None or not state[2]:
            raise ValueError("the position is over or has no legal moves")
        self.mover, self.legal = state[1], list(state[2])
        self._settled: dict = {}
        facts = CE.facts(env.variant, env.spec, env.vocabulary_json, position, [], True)
        rel = env.relations
        self.units = sorted((Unit(f[-2], f[-1], tuple(f[1:-2])) for f in facts if f[0] == UNIT_RELATION), key=lambda u: (u.side != SELF, u.kind, u.at))
        self.acts = _rows(facts, rel["acts_on"])
        self.go = [_rows(facts, spec) for spec in rel["can_go"]]
        self.kind_key = kind_keys(env.codec, position)
        self.read_seconds = time.perf_counter() - t0
        self._scenario: dict = {}
        self._census: dict = {}
        self.frame_log: list = []          # (seconds, source) of every frame solved: source real, given or static
        self._frames: dict = {}
        self._goal_kind: "tuple | None" = None
        self.passes = True                 # set by `step` from Params.passes
        self._pass: "tuple | None" = None  # (the passed position string or None when refused,) once asked
        self.pass_log: list = []           # True when the pass was given, False when refused, one entry per position asked

    # ---- stakes
    def worth_of(self, kind: str) -> float:
        return float(self.worth.get(kind, self.worth.get(self.kind_key.get(kind, kind), 0)))

    def lowest_worth(self) -> float:
        """The lowest worth above nothing in the table."""
        return min([w for w in self.worth.values() if w > 0] or [0.0])

    def weighty(self, u: Unit) -> bool:
        """A goal unit, or a unit whose kind is worth more than the lowest worth."""
        return u == self.goal_unit(u.side) or self.worth_of(u.kind) > self.lowest_worth()

    def weight(self, u: Unit) -> float:
        """What losing the unit costs: its worth, and for a goal unit the top stake (its worth in the table is nothing)."""
        if u == self.goal_unit(u.side):
            return max([self.worth_of(x.kind) for x in self.units] + [1.0]) * Params().goal_stake_multiple
        return self.worth_of(u.kind)

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

    def passed_position(self) -> "str | None":
        """This position with the other side to move (the kernel's pass), None when passes are off or the kernel refuses it (the mover under attack on its goal unit, the game over)."""
        if not self.passes:
            return None
        if self._pass is None:
            got = CE.passed(self.env.variant, self.position, [])
            self._pass = (got[0] if got else None,)
            self.pass_log.append(got is not None)
        return self._pass[0]

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
            return (LIVE if live else IDLE), text, {"strikers": strikers, "guards": guards, "open_exits": open_exits, "exits": total, "struck": bool(strikers), "cornered": bool(strikers) and open_exits == 0}
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


LINKS = {1: "reach", 2: "a place to stand", 3: "a strike that gains"}


def _link_check(r: Reading, t: Unit, u: Unit, to: tuple, moved: bool, what: str, struck_now: bool) -> tuple:
    """(link that fails or 0, why) for a unit `u` of the asking side that would stand on `to` and strike the target `t` (what = "strike") or cover one of its exits.
    The strikes of the other units are read on the board as it stands: the hypothetical arrival of `u` does not change what any other unit strikes (a line it blocks or opens
    is not followed). A unit that covers an exit gains only if the target is already struck by the asking side, so for it link 3 also asks that."""
    if moved:
        hit = [v for v in r.units if v.side == t.side and to in r.strikes_of(v)]
        cheaper = [v for v in hit if r.weight(v) < r.weight(u)]
        backed = [w for w in r.units if w.side == u.side and w != u and to in r.strikes_of(w)]
        if cheaper:
            return 2, f"{unit_words(u)} would stand on {place_text(to)}, struck by {unit_words(cheaper[0])}, which is worth less"
        if hit and not backed:
            return 2, f"{unit_words(u)} would stand on {place_text(to)}, struck by {unit_words(hit[0])} and not guarded"
    if what != "strike" and not struck_now:
        return 3, f"{unit_words(u)} would only cover an exit of {unit_words(t)} and nothing strikes it yet"
    guards = [v for v in r.units if v.side == t.side and v != t and t.at in r.strikes_of(v)]
    if guards and r.weight(t) <= r.weight(u):
        return 3, f"{unit_words(t)} is guarded by {unit_words(guards[0])} and is worth no more than {unit_words(u)}"
    return 0, ""


def level1(q: Question, r: Reading, params: Params) -> tuple:
    status, text, detail = _level1(q, r, params)
    if q.goal:
        rep = r.scenario(_role_of(q.asker))
        if rep.get("target") and not rep.get("unreadable"):
            detail["suppliers"] = len({(x["kind"], x["from"]) for per in rep["supply"] for x in per})
            detail["total_moves"] = min((sc["moves"] for sc in rep["scenarios"]), default=None)
    return status, text, detail


def _level1(q: Question, r: Reading, params: Params) -> tuple:
    t, k = q.target, params.reach_k
    side = side_word(q.asker)
    if q.goal:
        rep = r.scenario(_role_of(q.asker))
        if rep.get("target") and not rep.get("unreadable"):
            if rep.get("goal_holds"):
                return LIVE, "the goal already holds", {"scenarios": 0, "live_old": True}
            if rep["safe"] is True and not params.goal_near:
                return CLOSED, f"no unit of {side} can reach it within {k} moves", {"scenarios": 0, "failed_at": 1, "live_old": False}
            if rep["safe"] is True:
                open_exits = sum(1 for h in rep["holes"] if h["kind"] == "exit")
                if open_exits <= params.max_open_exits:
                    return IDLE, f"only {open_exits} of its exits are open, but no unit of {side} can reach it within {k} moves", {"scenarios": 0, "open_exits": open_exits, "near": False, "live_old": False}
                return CLOSED, f"no unit of {side} can reach it within {k} moves", {"scenarios": 0, "failed_at": 1, "open_exits": open_exits, "near": False, "live_old": False}
            open_exits = sum(1 for h in rep["holes"] if h["kind"] == "exit")
            near = asker_near(r, t, q.asker, k)
            held = {"open_exits": open_exits, "near": near, "scenarios": len(rep["scenarios"])}
            if rep["scenarios"]:
                best = min(rep["scenarios"], key=lambda s: s["moves"])
                plan = "; ".join(f"{p['kind']} from {p['from'].replace(':', '')} to {p['to'].replace(':', '')} in {p['moves']}" for p in best["placements"])
                ways = []
                for sc in sorted(rep["scenarios"], key=lambda s: s["moves"])[:3]:
                    for pl in sc["placements"]:
                        w = (pl["moves"], Unit(q.asker, pl["kind"], tuple(pl["from"].split(":"))), tuple(pl["to"].split(":")), "strike")
                        if w not in ways and pl["moves"] > 0:
                            ways.append(w)
                held.update(moves=best["moves"], supply=ways)
                fact = f"a plan of {best['moves']} moves of {side}: {plan} ({len(rep['scenarios'])} plans)"
                held["live_old"] = True
                if not params.goal_near:
                    return LIVE, fact, held
                if open_exits <= params.max_open_exits and near:
                    return LIVE, f"{open_exits} of its exits open and a unit of {side} near; {fact}", held
                if open_exits <= params.max_open_exits:
                    return IDLE, f"{open_exits} of its exits open but no unit of {side} is near; {fact}", held
                return IDLE, f"{open_exits} of its exits are open, more than {params.max_open_exits}; {fact}", held
            if params.links:
                held["live_old"] = open_exits <= params.max_open_exits
                if params.goal_near:
                    if open_exits <= params.max_open_exits and near:
                        return LIVE, f"no full plan read, but only {open_exits} of its exits are open and a unit of {side} is near", held
                    if open_exits <= params.max_open_exits:
                        return IDLE, f"only {open_exits} of its exits are open, but no unit of {side} is near", held
                elif open_exits <= params.max_open_exits:
                    return LIVE, f"no full plan read, but only {open_exits} of its exits are open", held
                return CLOSED, f"no way for {side} passes all three links: it fails at {LINKS[3]}, {open_exits} of its exits are open and no plan covers them", {**held, "failed_at": 3}
            return IDLE, f"the supply was not fully read (caps hit); {side} not shown to reach it", held
    now = r.strikers_of(t, q.asker)
    exits = r.exits_of(t)
    open_exits = exits - r.struck_by(q.asker)
    reach_tables = r.go[:min(k, len(r.go))]
    supply: list = [(0, u, u.at, "strike") for u in now]
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
    if not params.links:
        supply = [x for x in supply if x[0] > 0]
        if not supply and not now:
            return CLOSED, f"no unit of {side} can reach it within {k} moves", {"supply": []}
        if not supply:
            return IDLE, f"already struck by {_names(now)}; no other unit of {side} can join within {k} moves", {"supply": []}
        shown = "; ".join(f"{unit_words(u)} could {what} from {place_text(to)} in {j}" for j, u, to, what in supply[:4])
        return LIVE, f"{len(supply)} ways for {side} to join within {k} moves: {shown}", {"supply": supply}
    if not supply:
        return CLOSED, f"no unit of {side} can reach it within {k} moves", {"supply": [], "failed_at": 1}
    passed, failed = [], []
    for j, u, to, what in supply:
        link, why = _link_check(r, t, u, to, j > 0, what, bool(now))
        (failed.append((link, why, what)) if link else passed.append((j, u, to, what)))
    detail = {"supply": passed, "reach": len(supply), "failed": failed, "links": [len(supply), sum(1 for f in failed if f[0] > 2) + len(passed), len(passed)]}
    if not passed:
        pool = [f for f in failed if f[2] == "strike"] or failed          # the way that came furthest (a strike on the target before a mere cover of an exit); the last of them
        link, why, _ = max(pool, key=lambda f: f[0]) if len({f[0] for f in pool}) == 1 else [f for f in pool if f[0] == max(g[0] for g in pool)][-1]
        detail["failed_at"] = link
        return CLOSED, f"{len(supply)} ways for {side} reach it, none passes all three links; the last fails at {LINKS[link]}: {why}", detail
    shown = "; ".join(f"{unit_words(u)} could {what} from {place_text(to)} in {j}" for j, u, to, what in passed[:4])
    return LIVE, f"{len(passed)} of {len(supply)} ways for {side} pass reach, a place to stand and a strike that gains: {shown}", detail


def _matches(q: Question, r: Reading, tactic: dict) -> bool:
    if q.goal and "goal" in tactic["targets"]:
        return True
    return f"{r.kind_key.get(q.target.kind, q.target.kind)}@{place_text(q.target.at)}" in tactic["targets"]


def _first_moves(r: Reading, q: Question, way: tuple) -> list:
    """The (origin, destination) handles of the first move of a way: the strike itself for a unit already striking, the step to the arrival place, or to a place from which
    the arrival place is one move on."""
    j, u, to = way[0], way[1], way[2]
    if j == 0:
        return [(place_text(u.at), place_text(q.target.at))]
    if j == 1:
        return [(place_text(u.at), place_text(to))]
    one = r.go[0] if r.go else {}
    mids = sorted(m for m in one.get((u.kind, u.side, u.at), ()) if to in one.get((u.kind, u.side, m), ()))
    return [(place_text(u.at), place_text(m)) for m in mids]


def _gain(r: Reading, child: str) -> float:
    """What the move into `child` takes from the other side, in worth: the units it has fewer of than the position before."""
    codec = r.env.codec

    def count(position):
        out: dict = {}
        for side, kind in codec.owned(position):
            if side != r.mover:
                out[kind] = out.get(kind, 0) + 1
        return out
    before, after = count(r.position), count(child)
    return sum(int(r.worth.get(k, 0)) * (n - after.get(k, 0)) for k, n in before.items() if n > after.get(k, 0))


def _settle(r: Reading, mv: str, depth: int, params: Params) -> dict:
    """Play `mv` and read the child by the reasoning search at `depth`: the asker's value over the line = what the move takes + a threat nobody can parry - the best reply (a parry
    that wins something, or a counter of its own). confirmed: above 0; refuted: not, with the reply that does it; open: the move makes no threat and takes nothing."""
    key = (mv, depth)
    if key in r._settled:
        return r._settled[key]
    st = CE.state(r.env.variant, r.position, [mv])
    child = st[0]
    if st[3] is not None:
        out = {"move": mv, "status": "confirmed", "value": BIG_VALUE, "reply": None}
    else:
        rep = CE.reason(r.env.variant, child, [], depth, params.reason_fuse, {k: int(v) for k, v in r.worth.items()})
        gain = _gain(r, child)
        replies = [(a["net"], a["line"][0]) for a in rep["answers"] if a["line"]] + [(o["value"], o["line"][0]) for o in rep["ours"] if o["line"]]
        top = max(replies, key=lambda x: x[0]) if replies else (0, None)
        bonus = max((t["stake"] for t in rep["threats"]), default=0) if rep["threats"] and not rep["answers"] else 0
        value = gain + bonus - max(top[0], 0)
        reply = top[1] if top[0] > 0 or not rep["answers"] else rep["answers"][0]["line"][0]
        if value > 0:
            out = {"move": mv, "status": "confirmed", "value": value, "reply": None}
        elif rep["threats"] or gain > 0:
            out = {"move": mv, "status": "refuted", "value": value, "reply": reply}
        else:
            out = {"move": mv, "status": "open", "value": 0, "reply": None}
    r._settled[key] = out
    return out


def level_search(q: Question, r: Reading, params: Params) -> tuple:
    depth = min(q.level, params.search_depth_cap)
    ways = sorted(q.findings[1].detail.get("supply", []) if len(q.findings) > 1 else [], key=lambda w: (w[0], w[1].kind, w[1].at, w[2]))[:params.max_ways]
    results = []
    for way in ways:
        mv = None
        for o, d in _first_moves(r, q, way):
            mv = r.env.codec.find_move(r.position, o, d)
            if mv in r.legal:
                break
            mv = None
        results.append({"way": way, **(_settle(r, mv, depth, params) if mv else {"move": None, "status": "open", "value": 0, "reply": None})})
    conf = [x for x in results if x["status"] == "confirmed"]
    refu = [x for x in results if x["status"] == "refuted"]
    detail = {"mode": "search", "depth": depth, "results": results, "confirmed": len(conf), "refuted": len(refu), "open": len(results) - len(conf) - len(refu)}
    side = side_word(q.asker)
    if not results:
        return IDLE, f"no way of {side} to play at {depth} plies", detail
    if conf:
        best = max(conf, key=lambda x: x["value"])
        detail.update(value=best["value"], line=(best["move"],))
        shown = "reaches the goal" if best["value"] >= BIG_VALUE else f"value {best['value']}"
        return LIVE, f"{len(conf)} of {len(results)} ways of {side} stand after the reply within {depth} plies; best: {best['move']}, {shown}", detail
    if len(refu) == len(results):
        x = refu[0]
        return CLOSED, f"every way is answered: {x['move']} is met by {x['reply']}", detail
    return IDLE, f"{len(refu)} of {len(results)} ways are answered within {depth} plies, {detail['open']} not settled", detail


# ---------------------------------------------------------------------------------------------------------------------------------------------- frames: the exchange on a unit's place

def asker_near(r: Reading, t: Unit, asker: str, k: int) -> bool:
    """At least one unit of the asking side strikes the target's place or one of its exits now, or can come to a place that does within `k` moves."""
    goals = {t.at} | r.exits_of(t)
    for u in r.units:
        if u.side != asker:
            continue
        if r.strikes_of(u) & goals:
            return True
        for table in r.go[:min(k, len(r.go))]:
            if any(set(r.acts.get((u.kind, u.side, to), ())) & goals for to in table.get((u.kind, u.side, u.at), ())):
                return True
    return False


class _RealExchange:
    """The exchange frame over real positions (a position string, the side to move being the side that may take): its moves are the legal moves that take the unit on one place, the
    cheapest taker first; the gain of a taking is the worth of the unit taken; the exit is 0 (the side to move may stop). Lines opened by a taking and units that may not legally move
    are the kernel's business: the moves are its own."""

    def __init__(self, r: Reading, place: tuple):
        self.r, self.place, self.codec = r, place_text(place), r.env.codec
        self._st: dict = {}

    def state(self, pos: str) -> tuple:
        if pos not in self._st:
            self._st[pos] = CE.state(self.r.env.variant, pos, [])
        return self._st[pos]

    def moves(self, pos: str) -> list:
        s = self.state(pos)
        if s[3] is not None:
            return []
        found = self.codec.takings_on(pos, self.place, list(s[2]))
        found.sort(key=lambda x: (self.r.worth_of(x[1]), x[0]))
        return [m for m, _ in found]

    def gain(self, pos: str, move: str) -> float:
        return float(self.r.worth_of(self.codec.kind_at(pos, self.place) or ""))

    def next(self, pos: str, move: str) -> str:
        return CE.state(self.r.env.variant, pos, [move])[0]

    def exit(self, pos: str) -> float:
        return 0.0

    def key(self, pos: str) -> str:
        return pos


class _StaticExchange:
    """The same frame read from the fact view: a state is (strikers of the side to move, strikers of the other side, worth of the unit standing on the place); a striker is
    (worth, may take only when the other side has none left, label), sorted cheapest first. The standard swap with the option to stop. Lines behind the strikers are not read."""

    def moves(self, s: tuple) -> list:
        mine, theirs, _occ = s
        for i, (_w, only_last, _l) in enumerate(mine):
            if not only_last or not theirs:
                return [i]
        return []

    def gain(self, s: tuple, move: int) -> float:
        return float(s[2])

    def next(self, s: tuple, move: int) -> tuple:
        mine, theirs, _occ = s
        return (theirs, mine[:move] + mine[move + 1:], mine[move][0])

    def exit(self, s: tuple) -> float:
        return 0.0

    def key(self, s: tuple) -> tuple:
        return s


def _line_of(fs, frame, state, label) -> tuple:
    """The line the best taking leads to (the moves of both sides in turn, `label` naming a move), empty when no taking is worth more than stopping or none exists."""
    out = []
    root = {m: frame.gain(state, m) - fs.value(frame.next(state, m)) for m in frame.moves(state)}
    if not root:
        return ()
    best = max(root, key=lambda m: (root[m], -list(root).index(m)))
    out.append(label(state, best))
    state = frame.next(state, best)
    for _ in range(20):
        v = fs.value(state)
        nxt = [m for m in frame.moves(state) if frame.gain(state, m) - fs.value(frame.next(state, m)) == v]
        if not nxt or frame.exit(state) == v:
            break
        out.append(label(state, nxt[0]))
        state = frame.next(state, nxt[0])
    return tuple(out)


def _solve(r: Reading, frame, state, label, params: Params, source: str) -> dict:
    """The starter's value of a frame in the judge's worth units (it may stop: never below 0), its line, and the source; the time of the solve is logged on the reading."""
    t0 = time.perf_counter()
    fs = FrameSearch(frame, budget=params.frame_budget)
    value = fs.value(state)
    line = _line_of(fs, frame, state, label) if value >= 0 else ()
    r.frame_log.append((time.perf_counter() - t0, source))
    return {"value": value, "line": line, "source": source, "open": fs.stopped > 0}


def _static_state(r: Reading, strikers: list, guards: list, occupant: float) -> tuple:
    goal = {r.goal_kind()} - {None}

    def item(u):
        return (r.worth_of(u.kind), u.kind in goal, unit_words(u))
    return (tuple(sorted(item(u) for u in strikers)), tuple(sorted(item(u) for u in guards)), occupant)


def _static_label(state: tuple, move: int) -> str:
    return state[0][move][2]


def _guards_of(r: Reading, t: Unit, at: tuple) -> list:
    return [u for u in r.units if u.side == t.side and u != t and at in r.strikes_of(u)]


def exchange_static(r: Reading, t: Unit, asker: str, params: Params, strikers: "list | None" = None, at: "tuple | None" = None, source: str = "static") -> dict:
    """The exchange on the target's place started by the asker, read from the fact view (the strikers and guards as the board stands; `strikers` and `at` give a hypothetical
    arrival). Marked: read from the board as it stands, lines behind not read."""
    at = at or t.at
    strikers = r.strikers_of(t, asker) if strikers is None else strikers
    state = _static_state(r, strikers, _guards_of(r, t, at), r.worth_of(t.kind))
    out = _solve(r, _StaticExchange(), state, _static_label, params, source)
    out["note"] = "read from the board as it stands, lines behind not read"
    return out


def exchange_stands(r: Reading, t: Unit, asker: str, params: Params = Params()) -> dict:
    """The exchange frame started by the asker on the target as it stands (cached). The asker is to move: the frame's moves are the real legal takings (source real). The asker is not to
    move: the position with the asker to move by the kernel's pass, the same real frame; when the pass is refused (or switched off), the static form."""
    key = ("stands", t, asker)
    if key not in r._frames:
        base = r.position if asker == SELF else r.passed_position()
        if base is not None:
            r._frames[key] = _solve(r, _RealExchange(r, t.at), base, lambda pos, m: m, params, "real")
        else:
            r._frames[key] = exchange_static(r, t, asker, params)
    return r._frames[key]


def exchange_after(r: Reading, t: Unit, asker: str, way: tuple, params: Params = Params()) -> dict:
    """The exchange frame after the way's first move (a unit of the asker comes to strike). The asker to move: the first move is played for real, the owner answers with every legal
    move (the target followed where it goes), and the asker starts the real frame from there; the value is the least over the answers (source given). The asker not to move: the move
    cannot be played from the position as it stands: it is played from the passed position (the asker to move by the kernel's pass), the same way; the static form with the supplier
    placed only when the pass is refused."""
    j, u, to, what = way
    base = r.position if asker == SELF else r.passed_position()
    if base is not None:
        legal = list(CE.state(r.env.variant, base, [])[2])
        mv = None
        for o, d in _first_moves(r, Question(t, asker, 0.0, 0), way):
            mv = r.env.codec.find_move(base, o, d)
            if mv in legal:
                break
            mv = None
        if mv is not None:
            t0 = time.perf_counter()
            child = CE.state(r.env.variant, base, [mv])
            if child[3] is not None:
                r.frame_log.append((time.perf_counter() - t0, "given"))
                return {"value": BIG_VALUE if 100 in list(child[3]) else 0.0, "line": (mv,), "source": "given", "open": False}
            worst, line, where = None, (mv,), place_text(t.at)
            for m2 in child[2]:
                nxt = CE.state(r.env.variant, child[0], [m2])
                from_h, to_h = r.env.codec.move_ends(m2)
                moved_to = to_h if from_h == where else where
                if nxt[3] is not None or r.env.codec.kind_at(nxt[0], moved_to) is None:
                    v = {"value": 0.0, "line": (mv, m2)}
                else:
                    v = _solve(r, _RealExchange(r, tuple(moved_to)), nxt[0], lambda pos, m: m, params, "given")
                    r.frame_log.pop()                          # one step is the whole question below, logged once
                    v["line"] = (mv, m2) + tuple(v["line"])
                if worst is None or v["value"] < worst["value"]:
                    worst = v
                if worst["value"] <= 0:
                    break
            r.frame_log.append((time.perf_counter() - t0, "given"))
            return {"value": worst["value"] if worst else 0.0, "line": worst["line"] if worst else (mv,), "source": "given", "open": False}
    strikers = [v for v in r.strikers_of(t, asker) if v != u] + ([u] if what == "strike" else [])
    return exchange_static(r, t, asker, params, strikers=strikers, at=t.at, source="static")


def cornering(r: Reading, t: Unit, asker: str, params: Params = Params()) -> dict:
    """Is the unit cornered: lost where it stands (the exchange frame of the asker says so) and lost on every place it can go to. The owner to move: its real legal moves, each read with
    the asker to move (the real frame). The owner not to move: the same on the passed position (the owner to move by the kernel's pass); when the pass is refused, its `go` places from the fact view, each read by the static form with the unit placed there. A unit with no place to go
    is cornered when it is lost where it stands. -> {cornered, lost_now, places: [(place, lost)], open_exits: places where it is not lost}."""
    now = exchange_stands(r, t, asker, params)
    places = []
    where = place_text(t.at)
    if asker == OTHER:
        for m in r.env.codec.moves_of(where, r.legal):
            child = CE.state(r.env.variant, r.position, [m])
            dest = r.env.codec.move_ends(m)[1]
            if child[3] is not None:
                places.append((dest, False))
                continue
            v = _solve(r, _RealExchange(r, tuple(dest)), child[0], lambda pos, mm: mm, params, "real")
            places.append((dest, v["value"] > 0))
    elif r.passed_position() is not None:
        base = r.passed_position()
        for m in r.env.codec.moves_of(where, list(CE.state(r.env.variant, base, [])[2])):
            child = CE.state(r.env.variant, base, [m])
            dest = r.env.codec.move_ends(m)[1]
            if child[3] is not None:
                places.append((dest, False))
                continue
            v = _solve(r, _RealExchange(r, tuple(dest)), child[0], lambda pos, mm: mm, params, "real")
            places.append((dest, v["value"] > 0))
    else:
        own = {u.at for u in r.units if u.side == t.side and u != t}
        for to in sorted(r.exits_of(t) - own):
            hit = [u for u in r.units if u.side == asker and u.at != to and to in r.strikes_of(u)]
            gd = [u for u in r.units if u.side == t.side and u != t and to in r.strikes_of(u)]
            v = _solve(r, _StaticExchange(), _static_state(r, hit, gd, r.worth_of(t.kind)), _static_label, params, "static")
            places.append((place_text(to), v["value"] > 0))
    lost_now = now["value"] > 0
    return {"cornered": lost_now and all(lost for _p, lost in places), "lost_now": lost_now, "places": places, "open_exits": sum(1 for _p, lost in places if not lost)}


SOURCE_WORDS = {"real": "real moves", "given": "the way's first move played, every answer, real moves", "static": "read from the board as it stands, lines behind not read"}


def level_frames(q: Question, r: Reading, params: Params) -> tuple:
    """Level 2 of a non-goal unit by the exchange frame: the asker's frame on the target as it stands, then on the target after each way that passed the three links. CONFIRMED (live)
    when the asker's value is above 0, REFUTED (closed) otherwise, with the line."""
    t0 = len(r.frame_log)
    ways = sorted(q.findings[1].detail.get("supply", []) if len(q.findings) > 1 else [], key=lambda w: (w[0], w[1].kind, w[1].at, w[2]))[:params.max_ways]
    stands = exchange_stands(r, q.target, q.asker, params)
    results = [{"way": None, "move": None, "status": "confirmed" if stands["value"] > 0 else "refuted", **stands}]
    for way in ways:
        if way[0] == 0:
            continue                                                  # a unit that strikes it now is the frame as it stands
        key = ("after", q.target, q.asker, way[:4])
        if key not in r._frames:
            r._frames[key] = exchange_after(r, q.target, q.asker, way, params)
        v = r._frames[key]
        results.append({"way": way, "move": v["line"][0] if (v["line"] and v["source"] == "given") else None, "status": "confirmed" if v["value"] > 0 else "refuted", **v})
    conf = [x for x in results if x["status"] == "confirmed"]
    side = side_word(q.asker)
    detail = {"mode": "frame", "results": results, "confirmed": len(conf), "refuted": len(results) - len(conf), "open": 0, "lost_where_it_stands": stands["value"] > 0,
              "sources": sorted({x["source"] for x in results}), "frame_steps": len(r.frame_log) - t0}
    if stands["value"] > 0:
        detail.update(cornering(r, q.target, q.asker, params))
    if conf:
        best = max(conf, key=lambda x: x["value"])
        detail.update(value=best["value"], line=tuple(best["line"]))
        where = "where it stands" if best["way"] is None else f"after {unit_words(best['way'][1])} comes to {place_text(best['way'][2])}"
        corner = ", and cornered" if detail.get("cornered") else ""
        return LIVE, f"the exchange {where} is worth {best['value']:g} to {side}: {' '.join(best['line']) or 'no taking'} ({SOURCE_WORDS[best['source']]}){corner}", detail
    x = results[0]
    return CLOSED, f"the exchange is stopped, worth nothing to {side}: {' '.join(x['line']) or 'no taking pays'} ({SOURCE_WORDS[x['source']]}); {len(results)} frames read", detail


def level_deep(q: Question, r: Reading, params: Params) -> tuple:
    if params.frames and not q.goal:
        return level_frames(q, r, params)
    if params.search and q.asker == SELF:
        return level_search(q, r, params)
    status, text, detail = level_census(q, r, params)
    detail["mode"] = "census" + ("" if q.asker == SELF else ", as if it were their turn")
    return status, (text if q.asker == SELF else text + " (read as if it were their turn)"), detail


def level_census(q: Question, r: Reading, params: Params) -> tuple:
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
    r.passes = params.passes
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
    mode: str = ""


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
    sheet.steps.append(Step(q, f.level, f.status, f.text, f.cost, f.detail.get("mode", "")))
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
    """Step every open question that has not yet been asked up to level `upto` (the measures read levels 0 and 1 of all questions); then, when every question is closed or idle and
    the quiet question has not been offered, offer it and step it."""
    costs = _Costs(params)
    for q in sheet.questions:
        while not q.quiet and q.state == "open" and q.level <= upto:
            _record(sheet, step(q, sheet.reading, params), costs, params)
    plain = [q for q in sheet.questions if not q.quiet]
    if plain and not any(q.quiet for q in sheet.questions) and all(q.state == "closed" or q.status == IDLE for q in plain):
        quiet = quiet_question(sheet.reading, len(plain) + 1, params, min(q.stake for q in plain))
        if quiet is not None:
            sheet.questions.append(quiet)
            _record(sheet, step(quiet, sheet.reading, params), costs, params)


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
    kind: str = ""                     # the target's kind
    place: str = ""                    # where the target stands
    weighty: bool = False              # a goal unit, or a kind worth more than the lowest worth
    live1: bool = False                # live at level 1 (a way passes all three links)
    failed_at: int = 0                 # the link a question closed at level 1 failed at
    taken: "bool | None" = None        # the unit was taken within 4 plies in the game (my units only; None: not read)


@dataclass
class Record:
    key: tuple                         # (game, ply)
    bad: bool                          # a bad move was played here
    ranks: list                        # one Row per question in the first order
    fell: "bool | None"                # the mover's unit balance fell within 4 plies (None: not read)
    costs: list                        # (level label, seconds) per step
    text: list = field(default_factory=list)
    quiet_at: "str | None" = None      # the place of the unit the quiet question names
    moved: "str | None" = None         # the unit handle of the mover's side that the move played at this ply left
    weight: float = 0.0                # the bad move's weight (surprises file)
    obj0: "bool | None" = None         # the sheet objects to the move played, at level 0
    obj2: "bool | None" = None         # ... a level 2 search confirms a way for the new mover
    alt_share: "float | None" = None   # share of the other legal moves the sheet does not object to at level 0
    ways: tuple = (0, 0, 0)            # level 2 search: ways confirmed, refuted, not settled
    answered: int = 0                  # questions closed by "every way is answered"
    lost_units: list = field(default_factory=list)   # (owner, lost where it stands by the exchange frame, taken) of each weighty non-goal unit (measure G, independent of level 1)
    frames: list = field(default_factory=list)  # (seconds, source) of each frame solved on the sheet
    passes: list = field(default_factory=list)  # True per pass the kernel gave, False per pass it refused, one per position
    units: list = field(default_factory=list)   # (owner, group, level 2 result, taken) of each weighty non-goal unit; group: flagged (open after level 1), closed1, other


def summarize_sheet(key: tuple, bad: bool, fell: "bool | None", sheet: Sheet, moved: "str | None" = None, taken_of=None, weight: float = 0.0, follow_all: bool = False) -> Record:
    ranks, ways, answered, units, lost_units = [], [0, 0, 0], 0, [], []
    for q in sheet.questions:
        if q.quiet:
            continue
        f = {x.level: x for x in q.findings}
        live0 = 0 in f and f[0].status == LIVE
        one = 1 in f and f[1].status == LIVE and (f[1].detail.get("moves") == 1 or any(x[0] == 1 for x in f[1].detail.get("supply", ())))
        closed1 = q.closed_at == 1 and q.state == "closed"
        place = place_text(q.target.at)
        taken = taken_of(place) if (taken_of and q.owner == SELF) else None
        ranks.append(Row(q.owner, q.live_by_level1, closed1, live0, live0 or one, q.target.kind, place, bool(sheet.reading and sheet.reading.weighty(q.target)),
                         1 in f and f[1].status == LIVE, f[1].detail.get("failed_at", 0) if (closed1 and 1 in f) else 0, taken))
        if follow_all and not q.goal and ranks[-1].weighty:
            group = "flagged" if q.live_by_level1 else ("closed1" if closed1 else "other")
            l2 = f.get(2)
            result = "" if l2 is None else {LIVE: "confirmed", CLOSED: "refuted", IDLE: "not settled"}[l2.status]
            units.append((q.owner, group, result, taken_of(place) if taken_of else None))
            ex = exchange_stands(sheet.reading, q.target, q.asker)
            lost_units.append((q.owner, ex["value"] > 0, taken_of(place) if taken_of else None))
        for x in q.findings:
            if x.detail.get("mode") == "search":
                ways[0] += x.detail["confirmed"]
                ways[1] += x.detail["refuted"]
                ways[2] += x.detail["open"]
        answered += q.state == "closed" and q.reason.startswith("every way is answered")
    quiet = next((q for q in sheet.questions if q.quiet and q.findings), None)
    quiet_at = place_text(quiet.findings[0].detail["unit"].at) if quiet else None
    costs = [(f"{s.level}" + (f" ({s.mode})" if s.mode else ""), s.cost) for s in sheet.steps if not s.question.quiet]
    return Record(key, bad, ranks, fell, costs, quiet_at=quiet_at, moved=moved, weight=weight, ways=tuple(ways), answered=answered, units=units, lost_units=lost_units,
                  frames=list(sheet.reading.frame_log) if sheet.reading else [], passes=list(sheet.reading.pass_log) if sheet.reading else [])


class Item(NamedTuple):
    key: tuple                         # (game, ply)
    pos: tuple                         # (variant, start, moves before the ply)
    bad: bool
    fell: "bool | None"
    moved: "str | None"                # the handle the mover's unit left with the move played
    moved_to: "str | None"             # ... and the handle it arrived on
    weight: float
    played: "str | None"               # the move played at this ply
    full: list                         # the whole game's moves


def _units_of(codec, position: str, side: str) -> set:
    return {h for h, d in zip(codec.units(position), codec.describe(position)) if d[0] == side}


def moved_pair(codec, variant: str, position: str, nxt: str) -> tuple:
    """(handle the mover's unit left, handle it arrived on) between a position and the one after a move (None where nothing changed hands)."""
    mover = CE.state(variant, position, [])[1]
    before, after = _units_of(codec, position, mover), _units_of(codec, nxt, mover)
    gone, came = sorted(before - after), sorted(after - before)
    return (gone[0] if gone else None), (came[0] if came else None)


def load_positions(directory: str, variant: str, limit: "int | None" = None, game: "str | None" = None) -> list:
    """[Item] from a surprise meter's directory (games.jsonl, surprises.jsonl). A position is a game's ply that has a reading of ours; a bad move is a MISSED surprise whose judge
    jump is below 0 (its `weight` from the file). `fell`: the mover's balance (by ply, from the games file) was lower at one of the next 4 plies, the other side's read negated.
    With `game` (its codec): `moved` and `moved_to`, the handles the mover's unit left and arrived on with the move played (a handle is taken to read as the place text)."""
    from catspace.incubator.analyst.discovery import curriculum as CU
    codec = CU.codec_of(game) if game else None
    d = Path(directory)
    games = {}
    for line in (d / "games.jsonl").read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            games[g["game"]] = g
    bad = {}
    for line in (d / "surprises.jsonl").read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            if s.get("kind") == "MISSED" and (s.get("judge") or {}).get("jump", 0) < 0:
                bad[(s["game"], s["ply"])] = float(s.get("weight", 1.0))
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
            v = g.get("variant") or variant
            frm = to = None
            if codec is not None:
                frm, to = moved_pair(codec, v, CE.state(v, g.get("start"), g["moves"][:ply])[0], CE.state(v, g.get("start"), g["moves"][:ply + 1])[0])
            out.append(Item((gid, ply), (v, g.get("start"), g["moves"][:ply]), (gid, ply) in bad, fell, frm, to, bad.get((gid, ply), 0.0), g["moves"][ply], g["moves"]))
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
    """The OBS lines of the measures (a), (b), (c), (d), (e) and the level 2 search counts."""
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
    out.append(f"OBS (b) level 1 closed on a unit of mine in {len(mine_closed)} positions ({_pct(len(mine_closed), n)}); of those with the balance read ({len(mc)}) the mover's balance fell within 4 plies in "
               f"{sum(r.fell for r in mc)} ({_pct(sum(r.fell for r in mc), len(mc))}); in all positions with the balance read ({len(read)}): {sum(r.fell for r in read)} ({_pct(sum(r.fell for r in read), len(read))})")
    mine = [x for r in records for x in r.ranks if x.owner == SELF and x.taken is not None]
    groups = [("live at level 0", lambda x: x.live0), ("live at level 1 (a way passes all three links)", lambda x: x.live1), ("open after level 1 (live at level 0 or 1)", lambda x: x.live),
              ("closed at level 1 by reach", lambda x: x.closed1 and x.failed_at == 1), ("closed at level 1 by a place to stand", lambda x: x.closed1 and x.failed_at == 2),
              ("closed at level 1 by no gain", lambda x: x.closed1 and x.failed_at == 3), ("closed at level 1 (any link)", lambda x: x.closed1)]
    for label, pick in groups:
        hi = [x for x in mine if pick(x) and x.weighty]
        lo = [x for x in mine if pick(x) and not x.weighty]
        out.append(f"OBS (b) my units, {label}: taken within 4 plies, above the lowest worth {sum(x.taken for x in hi)} of {len(hi)} ({_pct(sum(x.taken for x in hi), len(hi))}); "
                   f"the lowest {sum(x.taken for x in lo)} of {len(lo)} ({_pct(sum(x.taken for x in lo), len(lo))})")
    out.append(f"OBS (b) my units, all questions: taken {sum(x.taken for x in mine)} of {len(mine)} ({_pct(sum(x.taken for x in mine), len(mine))})")
    c = sum(r.ways[0] for r in records), sum(r.ways[1] for r in records), sum(r.ways[2] for r in records)
    out.append(f"OBS level 2 search: ways confirmed {c[0]}, refuted {c[1]}, not settled {c[2]}; questions closed by every way answered {sum(r.answered for r in records)}")
    quiet_pos = [r for r in records if not any(x.live for x in r.ranks)]
    named = [r for r in quiet_pos if r.quiet_at is not None and r.moved is not None]
    yes = [r for r in named if r.moved == r.quiet_at]
    no = [r for r in named if r.moved != r.quiet_at]
    out.append(f"OBS (d) positions with nothing live at level 0 or 1: {len(quiet_pos)} ({_pct(len(quiet_pos), n)}); with the quiet question named and the move read: {len(named)}; "
               f"the move played was a move of the unit it names in {len(yes)} ({_pct(len(yes), len(named))}); bad move follows {_pct(sum(r.bad for r in yes), len(yes))} when it was ({len(yes)}), "
               f"{_pct(sum(r.bad for r in no), len(no))} when it was not ({len(no)}); bad-move rate in all positions {_pct(bad, n)}")
    by: dict = {}
    for r in records:
        for level, cost in r.costs:
            by.setdefault(level, []).append(cost)
    for level in sorted(by):
        c = by[level]
        out.append(f"OBS (c) level {level}: {len(c)} steps, mean {1000 * sum(c) / len(c):.2f} ms, max {1000 * max(c):.2f} ms")
    judged = [r for r in records if r.obj0 is not None]
    if judged:
        bads, goods = [r for r in judged if r.bad], [r for r in judged if not r.bad]
        tw = sum(r.weight for r in bads)
        for label, pick in (("level 0", lambda r: r.obj0), ("level 0 or 2", lambda r: r.obj0 or r.obj2)):
            ob, og = sum(1 for r in bads if pick(r)), sum(1 for r in goods if pick(r))
            yes, no = [r for r in judged if pick(r)], [r for r in judged if not pick(r)]
            held = sum(r.weight for r in bads if pick(r))
            out.append(f"OBS (e) the sheet objects at {label} to {ob} of {len(bads)} bad moves ({_pct(ob, len(bads))}) and to {og} of {len(goods)} other moves ({_pct(og, len(goods))}); bad-move rate when it objects "
                       f"{_pct(sum(r.bad for r in yes), len(yes))} ({len(yes)}), when it does not {_pct(sum(r.bad for r in no), len(no))} ({len(no)}); it holds {100.0 * held / tw if tw else 0.0:.1f}% of the bad moves' weight (total {tw:.1f})")
        alts = [r.alt_share for r in judged if r.alt_share is not None]
        if alts:
            out.append(f"OBS (e) alternatives: in {len(alts)} bad-move positions sampled the sheet does not object at level 0 to a mean {100.0 * sum(alts) / len(alts):.1f}% of the other legal moves "
                       f"(least {100.0 * min(alts):.1f}%, most {100.0 * max(alts):.1f}%)")
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------- the sheet's objection to a move

def live0_places(sheet: Sheet) -> set:
    """The places of my units that are live at level 0 in the sheet's position."""
    return {place_text(q.target.at) for q in sheet.questions if not q.quiet and q.target.side == SELF and q.findings and q.findings[0].status == LIVE}


def objection(env: Env, worth: dict, params: Params, position_after: str, live_before: set, moved: tuple, search_depths: int = 2) -> tuple:
    """(objects at level 0, a level 2 search confirms a way) for the position after a move. The sheet is read there to level 0 for the units of the side that moved (now "theirs") that
    are goal units or worth more than the lowest worth; it OBJECTS at level 0 when one is live and was not live before the move (a unit that moved is followed to its new place). Level 1
    is read for those units and, for the ones still open, the level 2 search is asked `search_depths` times (depth 2, 3): confirmed when a way of the new mover stands after the reply."""
    r = Reading(env, position_after, worth)
    cand = [q for q in first_order(r, params) if q.target.side == OTHER and r.weighty(q.target)]
    obj0 = False
    for q in cand:
        if step(q, r, params).status == LIVE:
            place = place_text(q.target.at)
            before = moved[0] if (moved[1] is not None and place == moved[1]) else place
            obj0 = obj0 or before not in live_before
    obj2 = False
    for q in cand if search_depths else ():
        if q.state != "open":
            continue
        step(q, r, params)
        for _ in range(search_depths):
            if q.state != "open":
                break
            f = step(q, r, params)
            obj2 = obj2 or f.detail.get("confirmed", 0) > 0
    return obj0, obj2


def alternatives_share(env: Env, worth: dict, params: Params, position: str, live_before: set, played: "str | None") -> "float | None":
    """Of the other legal moves, the share the sheet does not object to at level 0 (see `objection`)."""
    variant, codec = env.variant, env.codec
    legal = [m for m in CE.state(variant, position, [])[2] if m != played]
    if not legal:
        return None
    free = 0
    for m in legal:
        child = CE.state(variant, position, [m])[0]
        try:
            obj0, _ = objection(env, worth, params, child, live_before, moved_pair(codec, variant, position, child), search_depths=0)
        except ValueError:
            obj0 = False
        free += not obj0
    return free / len(legal)


def measure(items: list, runner, samples: int = 3, codec=None, objector=None, alternatives=None, alt_limit: int = 100, follow_all: bool = False) -> tuple:
    """items from `load_positions`; `runner(variant, start, moves) -> Sheet`. `codec.taken_within` follows my units in the game; `objector(item, live_before) -> (obj0, obj2)` and
    `alternatives(item, live_before) -> share` are the (e) readings. -> (records, sample explanation lines, number unreadable)."""
    records, unreadable, picked, n_alt = [], 0, [], 0
    for it in items:
        try:
            sheet = runner(*it.pos)
        except ValueError:
            sheet = None
        if sheet is None:
            unreadable += 1
            continue
        variant, start, moves = it.pos
        taken_of = (lambda place, it=it: codec.taken_within(start, it.full, it.key[1], place, 4)) if codec else None
        rec = summarize_sheet(it.key, it.bad, it.fell, sheet, it.moved, taken_of, it.weight, follow_all)
        live_before = live0_places(sheet)
        if objector is not None:
            try:
                rec.obj0, rec.obj2 = objector(it, live_before)
            except ValueError:
                pass
        if alternatives is not None and it.bad and n_alt < alt_limit:
            try:
                rec.alt_share = alternatives(it, live_before)
            except ValueError:
                pass
            n_alt += 1
        records.append(rec)
        if len(picked) < samples and (any(x.live for x in rec.ranks) or len(records) % 50 == 0):
            picked.append((it.key, it.bad, sheet))
    text = []
    for key, bad, sheet in picked:
        text.append(f"OBS sample game {key[0]} ply {key[1]}{' (a bad move followed)' if bad else ''}:")
        text += [f"OBS   {ln}" for ln in explain(sheet)]
    return records, text, unreadable


def settle_flagged(sheet: Sheet, params: Params = Params()) -> None:
    """Step to level 2 every non-goal unit above the lowest worth whose question is open after level 1 (the measure G reads the level 2 result of the flagged units)."""
    costs = _Costs(params)
    for q in sheet.questions:
        if not q.quiet and not q.goal and q.state == "open" and q.level == 2 and q.live_by_level1 and sheet.reading.weighty(q.target):
            _record(sheet, step(q, sheet.reading, params), costs, params)


def format_flagged(records: list, frames: bool = False) -> list:
    """OBS lines of measure G: for the units above the lowest worth (not the goal unit) of each side, the share taken within 4 plies by group and by the level 2 result. Mine are read as
    if it were their turn (the census), theirs with the asker to move (the search)."""
    out = []
    modes = (("level 2 by the exchange frame, the asker not to move: real moves after the kernel's pass, the static form where it is refused (or with --no-pass)", "level 2 by the exchange frame, the asker to move: real takings") if frames
             else ("level 2 read as if it were their turn", "level 2 read with the asker to move, by search"))
    for side, word, mode in ((SELF, "mine", modes[0]), (OTHER, "theirs", modes[1])):
        rows = [u for r in records for u in r.units if u[0] == side and u[3] is not None]

        def cell(pick):
            sel = [u for u in rows if pick(u)]
            k = sum(u[3] for u in sel)
            return f"{k} of {len(sel)} ({_pct(k, len(sel))})"
        flagged = [u for u in rows if u[1] == "flagged"]
        out.append(f"OBS (G) {word} ({mode}): units read {len(rows)}; flagged (open after level 1) {len(flagged)}: taken {cell(lambda u: u[1] == 'flagged')}")
        for res in ("confirmed", "refuted", "not settled", ""):
            label = res or "no level 2 step"
            out.append(f"OBS (G) {word}, flagged, level 2 {label}: taken {cell(lambda u, res=res: u[1] == 'flagged' and u[2] == res)}")
        out.append(f"OBS (G) {word}, closed at level 1: taken {cell(lambda u: u[1] == 'closed1')}")
        out.append(f"OBS (G) {word}, neither flagged nor closed at level 1: taken {cell(lambda u: u[1] == 'other')}")
    if frames:
        out += format_frame_lines(records)
    return out


def format_frame_lines(records: list) -> list:
    """The measure over the exchange frame as the unit stands, independent of level 1, and the cost and sources of the frames."""
    out = []
    for side, word in ((SELF, "mine"), (OTHER, "theirs")):
        rows = [u for r in records for u in r.lost_units if u[0] == side and u[2] is not None]
        lost, safe = [u for u in rows if u[1]], [u for u in rows if not u[1]]
        out.append(f"OBS (G) {word}, exchange frame as it stands, units above the lowest worth: lost where it stands: taken within 4 plies {sum(u[2] for u in lost)} of {len(lost)} ({_pct(sum(u[2] for u in lost), len(lost))}); "
                   f"not lost: {sum(u[2] for u in safe)} of {len(safe)} ({_pct(sum(u[2] for u in safe), len(safe))})")
    log = [x for r in records for x in r.frames]
    if log:
        ms = [1000 * c for c, _s in log]
        out.append(f"OBS (G) frames: {len(log)} frame steps, mean {sum(ms) / len(ms):.2f} ms, max {max(ms):.2f} ms; sources: real moves {sum(1 for _c, s in log if s == 'real')}, "
                   f"the way's first move played with every answer {sum(1 for _c, s in log if s == 'given')}, static fallback (read from the board as it stands, lines behind not read) {sum(1 for _c, s in log if s == 'static')}")
    asked = [x for r in records for x in r.passes]
    if asked:
        out.append(f"OBS (G) passes: asked on {len(asked)} positions, given {sum(asked)}, refused {len(asked) - sum(asked)}")
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------- F: the goal unit's verdict against how games ended

BANDS = ((1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 10 ** 9))
SIGNS = (("live (level 0 or 1)", "live"), ("live at level 1, new rule (open exits at most 1 and an asker unit near)", "live1"), ("live at level 1, earlier rule", "live1_old"),
         ("an asker unit near", "near"), ("open exits at most 1", "exits1"), ("a scenario exists", "scenario"))


def band_of(plies_to_end: int) -> int:
    return next(i for i, (lo, hi) in enumerate(BANDS) if lo <= plies_to_end <= hi)


def band_name(i: int) -> str:
    lo, hi = BANDS[i]
    return f"{lo}-{hi}" if hi < 10 ** 8 else f"{lo}+"


def roles_of(variant: str) -> list:
    """The roles in the order the outcome lists them: the side to move at the start, then the other."""
    first = CE.state(variant, None, [])
    return [first[1], CE.state(variant, None, [first[2][0]])[1]]


def game_ending(variant: str, game: dict, roles: list) -> tuple:
    """(kind, index of the role that won or None): kind is decisive (the outcome field names a winner, `outcome` [100, 0] or [0, 100], and the kernel's final position is over with that
    outcome: the goal is met), draw (the outcome is [50, 50]), ply cap (`unfinished`), or other."""
    out = game.get("outcome")
    if game.get("unfinished") or not out:
        return "ply cap", None
    final = CE.state(variant, game.get("start"), game["moves"])
    if final[3] is not None and list(final[3]) == list(out):
        if sorted(out) == [0, 100]:
            return "decisive", out.index(100)
        if list(out) == [50, 50]:
            return "draw", None
    return "other", None


def goal_reading(env: Env, position: str, worth: dict, params: Params) -> dict:
    """{side: reading of the goal-unit question of that side} at levels 0 and 1 only: live (level 0 or 1), open exits, whether a scenario exists and its total moves, suppliers."""
    r = Reading(env, position, worth)
    out = {"mover": r.mover}
    for q in first_order(r, params):
        if not q.goal:
            continue
        lv0 = step(q, r, params)
        lv1 = step(q, r, params)
        exits = lv0.detail.get("open_exits")
        out[q.target.side] = {"live": q.live_by_level1, "live0": lv0.status == LIVE, "live1": lv1.status == LIVE, "live1_old": bool(lv0.status == LIVE or lv1.detail.get("live_old", lv1.status == LIVE)),
                              "near": bool(lv1.detail.get("near")), "cornered": bool(lv0.detail.get("cornered")), "open_exits": exits, "exits1": exits is not None and exits <= 1,
                              "scenario": lv1.detail.get("scenarios", 0) > 0, "total_moves": lv1.detail.get("total_moves"), "suppliers": lv1.detail.get("suppliers")}
    return out


def goal_rows(variant: str, game: dict, roles: list, readings: list, ending: tuple) -> list:
    """The rows of one game: one per goal unit per position, with its group (loser, winner, undecided), whether it is read as mine (its owner to move) or as theirs, and the plies before the end."""
    kind, winner = ending
    rows = []
    n = len(game["moves"])
    for ply, rd in enumerate(readings):
        if rd is None:
            continue
        for side in (SELF, OTHER):
            g = rd.get(side)
            if g is None:
                continue
            owner = rd["mover"] if side == SELF else next(x for x in roles if x != rd["mover"])
            group = "undecided" if kind != "decisive" else ("winner" if roles.index(owner) == winner else "loser")
            rows.append({"game": game["game"], "ply": ply, "to_end": n - ply, "group": group, "read": "mine" if side == SELF else "theirs", **g})
    return rows


def _quartiles(xs: list) -> tuple:
    xs = sorted(xs)
    pick = lambda q: xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]
    return pick(0.25), pick(0.5), pick(0.75)


def format_goal_verdict(rows: list, endings: dict, positions: int) -> list:
    """The OBS lines of F. `endings` is {(directory, game): (kind, winner)}."""
    kinds: dict = {}
    for k, w in endings.values():
        kinds[k] = kinds.get(k, 0) + 1
    out = [f"OBS (F) games read {len(endings)}, positions {positions}; endings: " + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items()))
           + "; decisive means the games file's `outcome` is [100, 0] or [0, 100] and the kernel's final position is over with it (the goal met)"]
    for group, label in (("loser", "the goal unit of the side that LOST a decisive game"), ("winner", "the goal unit of the side that WON a decisive game"),
                         ("undecided", "both goal units in games not decided (false alarm rate)")):
        for read, rlabel in (("mine", "read as mine (its owner to move)"), ("theirs", "read as theirs (the attacker to move)")):
            sel = [x for x in rows if x["group"] == group and x["read"] == read]
            for sname, key in SIGNS:
                cells = []
                for i in range(len(BANDS)):
                    b = [x for x in sel if band_of(x["to_end"]) == i]
                    k = sum(1 for x in b if x[key])
                    cells.append(f"{band_name(i)}: {k}/{len(b)} ({_pct(k, len(b))})")
                out.append(f"OBS (F) {label}, {rlabel}, sign {sname}, by plies before the end: " + "; ".join(cells))
    for sname, key in SIGNS:
        runs, none = [], 0
        by_game: dict = {}
        for x in rows:
            if x["group"] == "loser":
                by_game.setdefault(x["game"], {})[x["to_end"]] = x[key]
        for g, series in by_game.items():
            n = 0
            while series.get(n + 1):
                n += 1
            (runs.append(n) if n else None)
            none += n == 0
        if runs:
            q1, med, q3 = _quartiles(runs)
            out.append(f"OBS (F) losing goal unit, sign {sname}: live from the end back to {med} plies before the end (median; quartiles {q1} and {q3}) in {len(runs)} games; not live at the last position in {none}")
        else:
            out.append(f"OBS (F) losing goal unit, sign {sname}: never live at the last position in {none} games")
    return out


def goal_verdict(dirs: list, env: Env, worth: dict, params: Params, limit: "int | None" = None) -> tuple:
    """(rows, endings, positions): the goal-unit readings of every position of every game in the directories."""
    roles = roles_of(env.variant)
    rows, endings, positions = [], {}, 0
    for d in dirs:
        for line in (Path(d) / "games.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            g = json.loads(line)
            g["game"] = f"{Path(d).name}:{g['game']}"
            v = g.get("variant") or env.variant
            ending = game_ending(v, g, roles)
            endings[(d, g["game"])] = ending
            readings = []
            for ply in range(len(g["moves"])):
                try:
                    readings.append(goal_reading(env, CE.state(v, g.get("start"), g["moves"][:ply])[0], worth, params))
                except ValueError:
                    readings.append(None)
                positions += 1
            rows += goal_rows(v, g, roles, readings, ending)
            if limit and len(endings) >= limit:
                return rows, endings, positions
    return rows, endings, positions


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
    ap.add_argument("--games-dir", required=True, action="append", help="a surprise meter directory; with --goal-verdict the flag may repeat")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-links", action="store_true", help="level 1 by reach alone (the first version)")
    ap.add_argument("--no-frames", action="store_true", help="level 2 of a non-goal unit by the earlier forms (the search, the census) instead of the exchange frame")
    ap.add_argument("--no-pass", action="store_true", help="the asker not to move: the static form instead of the kernel's pass and real moves")
    ap.add_argument("--no-goal-near", action="store_true", help="the goal unit's level 1 by the earlier rule (no asker unit needed near)")
    ap.add_argument("--no-search", action="store_true", help="level 2 by the census alone (the first version)")
    ap.add_argument("--alternatives", type=int, default=100, help="bad-move positions whose other legal moves are read for the sheet's objection (0: none)")
    ap.add_argument("--no-objection", action="store_true", help="skip measure (e)")
    ap.add_argument("--goal-verdict", action="store_true", help="measure F: the goal unit's question against how the games ended (levels 0 and 1 only)")
    ap.add_argument("--flagged", action="store_true", help="measure G: the level 2 result on the flagged units of both sides, followed for 4 plies")
    ap.add_argument("--search-depth", type=int, default=None, help="the deepest level and the deepest search (default: the Params'); 2 keeps G to depth 2")
    ap.add_argument("--games", type=int, default=None, help="with --goal-verdict: at most this many games")
    ap.add_argument("--out", default=None, help="a name under the derived data directory to write the OBS lines to")
    a = ap.parse_args(argv)
    env = Env(a.game)
    worth = store_worth(a.game)
    params = Params(links=not a.no_links, search=not a.no_search, frames=not a.no_frames, goal_near=not a.no_goal_near, passes=not a.no_pass)
    if a.search_depth:
        params.max_level = params.search_depth_cap = a.search_depth
    t0 = time.perf_counter()
    if a.goal_verdict:
        rows, endings, positions = goal_verdict(a.games_dir, env, worth, params, a.games)
        lines = format_goal_verdict(rows, endings, positions) + [f"OBS question sheet: goal verdict {time.perf_counter() - t0:.1f} s for {len(endings)} games, {positions} positions"]
        for ln in lines:
            print(ln)
        return 0
    if len(a.games_dir) != 1:
        ap.error("several --games-dir only with --goal-verdict")
    items = load_positions(a.games_dir[0], a.game, a.limit, game=a.game)

    def runner(variant, start, moves):
        position = CE.state(variant, start, list(moves))[0]
        sheet = run(position, worth, max_steps=a.max_steps, budget_seconds=a.seconds, env=env, params=params)
        finish_levels(sheet, 1, params)
        if a.flagged:
            settle_flagged(sheet, params)
        return sheet

    def objector(it, live_before):
        variant, start, moves = it.pos
        after = CE.state(variant, start, list(moves) + [it.played])[0]
        return objection(env, worth, params, after, live_before, (it.moved, it.moved_to))

    def alternatives(it, live_before):
        variant, start, moves = it.pos
        return alternatives_share(env, worth, params, CE.state(variant, start, list(moves))[0], live_before, it.played)

    records, text, unreadable = measure(items, runner, codec=env.codec, objector=None if a.no_objection else objector,
                                        alternatives=None if (a.no_objection or not a.alternatives) else alternatives, alt_limit=a.alternatives, follow_all=a.flagged)
    lines = format_obs(records, unreadable) + (format_flagged(records, params.frames) if a.flagged else []) + text
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
