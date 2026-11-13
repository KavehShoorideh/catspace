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
    geometry: bool = True              # level 1 by the medium of movement (units ignored) then the obstacles on each route, as follow-up questions (False: the move tables, the board frozen)
    obstacle_depth: int = 1            # follow-up questions asked on the obstacles of a route, this deep (0: none asked, the obstacles are reported as not read)
    max_obstacles: int = 3             # obstacles of one route asked about; the rest are not read
    max_blocked_steps: int = 12        # blocked steps one question may examine by the counterfactual (a kernel question per other unit); the rest are not read
    aside_places: int = 4              # places tried when a unit the kernel will not read the board without is moved aside instead of removed (0: off)
    leaving_audit_every: int = 0       # n > 0: every n-th cost of leaving is also read by the whole reread and compared (the audit of the narrow reading)
    leaving_wide_sign: bool = False    # the goal sign of the cost of leaving is also read when the step touches any line to the goal unit (the audit's misses fall, the time doubles)
    leaving_full: bool = False         # the cost of leaving by a whole new reading of the child position (the earlier form); False: only what the leaving can change
    pair_cap: int = 0                  # pairs of units tried when no single removal makes a blocked step legal (0: off)
    joint: bool = True                 # no single removal makes a blocked step legal and the first layer names two or more units on it: remove the whole set together (False: "none", the earlier form)
    max_joint: int = 3                 # units in such a set; more is recorded as "more than N units in the way" (a refutation by cost, said in words)
    duties: bool = True                # the cost of leaving: what the travelling unit holds where it stands, read after its step
    refute_counterfactual: bool = True # a refuted route: would it be confirmed without the unit whose answer stopped it?
    max_obstacle_moves: int = 8        # moves of an own unit in the way tried when asking whether it can move away
    compare_old: bool = True           # level 1 also reads the move tables' reach, to count where the two readings disagree
    weak_places: bool = True           # the goal unit's question names its weak places (guards not fewer than strikers) and asks what could be brought
    routes: bool = True                # a way of more than one move is judged by following its whole route (False: after its first move only, the earlier form)
    max_routes: int = 3                # routes of a way the reasoner follows, the best-ordered first
    route_nodes: int = 2500            # nodes (an owner's answer examined, a frame node) one way may use over all its routes; past it the way is NOT SETTLED, never refuted
    waiting_urgency: float = 0.5       # a question left idle by a waiting way: its urgency is multiplied by this as well as by urgency_down
    chances: "str | None" = None       # a table of chances fitted at sleep (a path, see `fit_chances`): every question and node then carries `chance`; None: left empty, as before
    chance_horizon: int = 4            # the table's horizon (plies within which the exact unit is taken): 4 or 8
    clock: str = "present"             # the order of steps in `run`: "present" (stake x urgency over cost), "chance" (chance x stake at risk over cost), "plain" (the first order, each question to full depth in turn)
    move_question: bool = False        # after the questions, the question of each legal move (`move_sheet`) is asked and kept on the sheet (False: as before)
    risk: float = 1.0                  # THE TOLERANCE FOR RISK: the stake at risk of my own unit is risk x its high bound + (1 - risk) x the midpoint of its bounds; 1 = the worst case, 0 = the expected value


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
        full = {**GS.vocabulary(game, vocabulary), "scenario": raw["scenario"]}
        self.medium_names = [raw["scenario"][role]["relation"] for role in MEDIUM_ROLES if role in raw["scenario"]]
        self.vocabulary = {**full, "derived": [x for x in full["derived"] if x not in self.medium_names]}          # the medium does not depend on the position: read once, not at every position
        self.vocabulary_json = json.dumps(self.vocabulary)
        self.medium_json = json.dumps({**full, "derived": list(self.medium_names)})
        self.spec = GS.spec_texts(game, vocabulary)[self.variant]
        self.codec = CU.codec_of(game)
        self._medium: "Medium | None | bool" = None

    def worth_fn(self, worth: dict):
        """kind (the description's word) -> its worth in the table, the same reading as `Reading.worth_of` but for any kind of the game, whatever the position holds (the keys are read once from the
        initial position)."""
        if not hasattr(self, "_kind_key"):
            self._kind_key = kind_keys(self.codec, CE.state(self.variant, None, [])[0])
        table, keys = dict(worth), self._kind_key
        return lambda kind: float(table.get(kind, table.get(keys.get(kind, kind), 0)))

    def medium(self) -> "Medium | None":
        """The medium of movement of the game (the one-move relations on the empty board), read once per game from the initial position and from the position after one move (so both
        sides are read as the mover); None when the vocabulary does not name the roles."""
        if self._medium is None:
            if len(self.medium_names) < len(MEDIUM_ROLES):
                self._medium = False
            else:
                first = CE.state(self.variant, None, [])
                second = CE.state(self.variant, None, [first[2][0]])
                tables = {}
                for st in (first, second):
                    facts = CE.facts(self.variant, self.spec, self.medium_json, st[0], [], True)
                    tables[st[1]] = [f for f in facts if f[0] in self.medium_names and f[2] == SELF]
                self._medium = Medium(tables, self.relations)
        return self._medium or None


MEDIUM_ROLES = ("hop", "hop_take", "hop_over")


class Medium:
    """LAYER ONE: the medium of movement with every unit ignored. `hops(kind, side, place)` are the places one move reaches, `distance` the least number of hops (breadth first), `routes` the
    shortest routes (the places landed on), `over` the places passed over by one hop along a line, `strike_places(kind, side, x)` the places from which a unit of that kind bears on `x`
    (by the take-only hops where the kind has them, else by its hops). Sides are the kernel's own names; places are the texts the codec writes."""

    def __init__(self, tables: dict, relations: dict):
        self.hop: dict = {}                    # (kind, side, from) -> {to: take_only}
        self.over_of: dict = {}                # (kind, side, from, to) -> places passed over
        self.takes: set = set()                # (kind, side): kinds with take-only hops
        self._dist: dict = {}
        self._strike: dict = {}
        for side, facts in tables.items():
            for f in facts:
                spec = relations[next(role for role in MEDIUM_ROLES if relations[role]["relation"] == f[0])]
                a = f[1:]
                kind, frm, to = a[spec["kind"]], "".join(a[i] for i in spec["from"]), "".join(a[i] for i in spec["to"])
                if f[0] == relations["hop_over"]["relation"]:
                    self.over_of.setdefault((kind, side, frm, to), []).append("".join(a[i] for i in spec["over"]))
                else:
                    take = f[0] == relations["hop_take"]["relation"]
                    self.hop.setdefault((kind, side, frm), {})[to] = take
                    if take:
                        self.takes.add((kind, side))

    def lines_into(self, to: str) -> set:
        """The places passed over by some hop of some kind and side that lands on `to` (every line to it, units ignored)."""
        if not hasattr(self, "_into"):
            self._into: dict = {}
            for (_k, _s, _f, t), over in self.over_of.items():
                self._into.setdefault(t, set()).update(over)
        return self._into.get(to, set())

    def hops(self, kind: str, side: str, frm: str) -> dict:
        return self.hop.get((kind, side, frm), {})

    def over(self, kind: str, side: str, frm: str, to: str) -> tuple:
        return tuple(sorted(self.over_of.get((kind, side, frm, to), ())))

    def distances(self, kind: str, side: str, frm: str) -> dict:
        key = (kind, side, frm)
        if key not in self._dist:
            dist, layer = {frm: 0}, [frm]
            while layer:
                nxt = []
                for p in layer:
                    for to in self.hops(kind, side, p):
                        if to not in dist:
                            dist[to] = dist[p] + 1
                            nxt.append(to)
                layer = nxt
            self._dist[key] = dist
        return self._dist[key]

    def distance(self, kind: str, side: str, frm: str, to: str) -> "int | None":
        return self.distances(kind, side, frm).get(to)

    def routes(self, kind: str, side: str, frm: str, to: str, cap: int = 200) -> list:
        """Every shortest route (the places landed on, in order) from `frm` to `to`, at most `cap`."""
        n = self.distance(kind, side, frm, to)
        out: list = []

        def walk(p, left, path):
            if len(out) >= cap:
                return
            if left == 0:
                if p == to:
                    out.append(tuple(path))
                return
            for y in sorted(self.hops(kind, side, p)):
                if self.distance(kind, side, y, to) == left - 1:
                    walk(y, left - 1, path + [y])
        if n:
            walk(frm, n, [])
        return out

    def strike_places(self, kind: str, side: str, x: str) -> set:
        key = (kind, side, x)
        if key not in self._strike:
            take = (kind, side) in self.takes
            self._strike[key] = {p for (k, sd, p), row in self.hop.items() if k == kind and sd == side and x in row and (row[x] if take else True)}
        return self._strike[key]




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
        self.params = Params()             # the schedule the reading is used under (set by the cost of leaving)
        self.leaving_audit: list = []      # one dict per audited cost of leaving: the narrow reading against the whole reread
        self._audit_n = 0
        self._refused: set = set()         # (position, move) the kernel refused as not a position
        self._states: dict = {}            # (position, move or None) -> the kernel's state: the route reasoner meets the same positions again and again
        self._other: "str | None" = None
        self.cf_cut = 0                    # blocked steps not examined because the question's cap was reached
        self.joint_log: list = []          # one dict per set of units that must all go: members, asked, dead, moves more
        self.cf_log: list = []             # one dict per blocked step examined by the counterfactual: ms, kernel calls, units found, describing words
        self.questions: list = []          # the sheet's questions (set by run): the time obstacle reads what they already hold
        self.graph = Graph()               # the claims of every question of this position and how they follow from one another
        self.pool: "Pool | None" = None    # set by `run`: the waiting pool that carries ways from one position of a game to the next

    def state_of(self, position: str, move: "str | None" = None) -> tuple:
        """The kernel's state of `position` (after `move` when given), cached."""
        key = (position, move)
        if key not in self._states:
            self._states[key] = CE.state(self.env.variant, position, [move] if move else [])
        return self._states[key]

    def try_state(self, position: str, move: "str | None" = None) -> "tuple | None":
        """`state_of`, or None when the kernel refuses the board (not a position of the game); the refusal is remembered too."""
        key = (position, move)
        if key in self._refused:
            return None
        try:
            return self.state_of(position, move)
        except Exception:
            self._refused.add(key)
            return None

    def absolute(self, side: str) -> str:
        """The side as the kernel names it (SELF is the mover; the other side is the mover after one move): the same name in every position of a game."""
        if side == SELF:
            return self.mover
        if self._other is None:
            self._other = self.state_of(self.position, self.legal[0])[1]
        return self._other

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


# ---------------------------------------------------------------------------------------------------------------------------------------------- the graph of claims

EDGE_KINDS = ("needs", "blocked_by", "if_removed", "stopped_by", "waits_on", "breaks", "holds")


@dataclass
class Node:
    """A question or a claim: what is asked, the answer, the verdict. kind: question, way, obstacle, follow_up, weak_place, candidate, answer, condition."""
    id: int
    kind: str
    asks: str
    answer: str = ""
    verdict: str = ""
    target: "Unit | None" = None
    asker: "str | None" = None
    reads: frozenset = frozenset()
    stake_low: float = 0.0             # what the node is worth to the asker, in the judge's units: a bound, tightened as answers arrive (nothing read: 0 to the target's stake)
    stake_high: "float | None" = None  # None until a parent or the maker gives it
    chance: "float | None" = None      # left empty: no probability is invented
    tried: int = 0                     # route nodes: the owner's answers examined
    cut: int = 0                       # route nodes: 1 when the node cap stopped the search
    data: dict = field(default_factory=dict)   # what filing the node as a fact needs: a way's route, node cap, nodes used, what it waits on and what breaks it


@dataclass
class Edge:
    """src -> dst. needs: the attack needs this unit to arrive; blocked_by: the route is stopped by that unit; if_removed: the follow-up question on an obstacle and what holds if it were gone;
    stopped_by: the owner's answer that refutes; waits_on: what a waiting way waits on; breaks: an answer of the owner that would stop it."""
    src: int
    dst: int
    kind: str
    because: str = ""
    gone: str = ""                     # the unit whose absence the counterfactual speaks of
    then: str = ""                     # what would hold if it were gone


class Graph:
    """Plain lists of nodes and typed edges; every level of a question adds to it."""

    def __init__(self):
        self.nodes: list = []
        self.edges: list = []
        self.index: dict = {}              # a key the recorder chooses -> node id (a node is made once)

    def node(self, kind: str, asks: str, key=None, **kw) -> Node:
        if key is not None and key in self.index:
            n = self.nodes[self.index[key]]
            for name, value in kw.items():
                if value not in ("", None, frozenset()):
                    setattr(n, name, value)
            return n
        n = Node(len(self.nodes), kind, asks, **kw)
        self.nodes.append(n)
        if key is not None:
            self.index[key] = n.id
        return n

    def link(self, src: Node, dst: Node, kind: str, because: str = "", gone: str = "", then: str = "") -> Edge:
        assert kind in EDGE_KINDS, kind
        for e in self.edges:
            if e.src == src.id and e.dst == dst.id and e.kind == kind:
                return e
        e = Edge(src.id, dst.id, kind, because, gone, then)
        self.edges.append(e)
        if dst.stake_high is None:                                     # a node's bound starts as its parent's: nothing is known to be worth more
            dst.stake_high = src.stake_high
        return e

    def out(self, node: Node) -> list:
        return [(e, self.nodes[e.dst]) for e in self.edges if e.src == node.id]

    def size(self) -> tuple:
        return len(self.nodes), len(self.edges)


def bound(n: Node, low: "float | None" = None, high: "float | None" = None) -> None:
    """Tighten a node's stake bound (a bound only closes: low never falls, high never rises, unless the node was unset)."""
    if low is not None:
        n.stake_low = max(n.stake_low, low) if n.stake_high is not None else low
    if high is not None:
        n.stake_high = high if n.stake_high is None else min(n.stake_high, high)
    if n.stake_high is not None and n.stake_low > n.stake_high:
        n.stake_low = n.stake_high


def chain_lines(graph: Graph, start: Node, depth: int = 8) -> list:
    """The chain from a starting node down, each line saying "because ..." and, where there is one, "if ... were gone then ..."."""
    out: list = []

    def walk(n: Node, e: "Edge | None", pad: str, seen: tuple):
        worth = "" if n.stake_high is None else (f" (worth between {n.stake_low:g} and {n.stake_high:g})" if n.stake_low != n.stake_high else f" (worth {n.stake_low:g})")
        head = f"{n.asks}" + (f" -> {n.answer}" if n.answer else "") + (f" [{n.verdict}]" if n.verdict else "") + worth + (f" [{n.tried} answers tried{', cut by the node cap' if n.cut else ''}]" if n.tried else "")
        line = f"{pad}{e.kind + ': ' if e else ''}{head}"
        if e and e.because:
            line += f" -- because {e.because}"
        if e and e.then:
            line += f"; if {e.gone or 'it'} were gone then {e.then}"
        out.append(line)
        if depth and len(pad) // 2 < depth:
            for edge2, node2 in graph.out(n):
                if node2.id not in seen:
                    walk(node2, edge2, pad + "  ", seen + (n.id,))
    walk(start, None, "", ())
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
    node: "int | None" = None          # the question's starting node in the sheet's graph
    blocked: int = 0                   # blocked steps the counterfactual has examined for this question
    waiting: list = field(default_factory=list)   # the waiting ways (the records of what each waits on and what breaks it) of the last deep step

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


def _stand(r: Reading, t: Unit, u: Unit, to: tuple) -> tuple:
    """(True, "") when a unit `u` of the asking side could stand on the place `to` (the target's side does not strike it with a cheaper unit, nor strike it while nothing of the asking side
    guards it); else (False, why). The strikes of the other units are read on the board as it stands."""
    hit = [v for v in r.units if v.side == t.side and to in r.strikes_of(v)]
    cheaper = [v for v in hit if r.weight(v) < r.weight(u)]
    backed = [w for w in r.units if w.side == u.side and w != u and to in r.strikes_of(w)]
    if cheaper:
        return False, f"{unit_words(u)} would stand on {place_text(to)}, struck by {unit_words(cheaper[0])}, which is worth less"
    if hit and not backed:
        return False, f"{unit_words(u)} would stand on {place_text(to)}, struck by {unit_words(hit[0])} and not guarded"
    return True, ""


@dataclass(frozen=True)
class Route:
    """The ordered places a unit passes to come to a way's arrival place, and for each the reading "a place to stand" (the same as the arrival place's)."""
    unit: Unit
    places: tuple                      # each a place as the description writes it; the last is the arrival place
    stands: tuple                      # per place: (can stand there, why not)
    what: str = "strike"
    over: tuple = ()                   # per hop: the places passed over (layer one)
    obstacles: tuple = ()              # layer two, as the board stands (a cheap listing; level 2 finds the obstacles by the counterfactual)
    goal: str = ""                     # the place the final strike is on


def routes_of(r: Reading, t: Unit, u: Unit, to: tuple, j: int, params: Params, what: str = "strike") -> list:
    """Every route of exactly `j` moves of `u` to `to` (by the one-move relation, a place on the route read for any placement), the ones whose every place can be stood on first, at most
    `params.max_routes` of them."""
    one = r.go[0] if r.go else {}
    step = lambda p: one.get((u.kind, u.side, p), ())
    layers = [{u.at}]
    for _ in range(j):
        layers.append({y for p in layers[-1] for y in step(p)})
    good: list = [set() for _ in range(j + 1)]
    good[j] = {to} if to in layers[j] else set()
    for i in range(j - 1, -1, -1):
        good[i] = {p for p in layers[i] if set(step(p)) & good[i + 1]}
    found: list = []

    def walk(p, i, path):
        if i == j:
            found.append(tuple(path))
            return
        for y in sorted(step(p)):
            if y in good[i + 1]:
                walk(y, i + 1, path + [y])
    if u.at in good[0]:
        walk(u.at, 0, [])
    out = [Route(u, path, tuple(_stand(r, t, u, p) for p in path), what, (), (), place_text(t.at)) for path in found]
    out.sort(key=lambda x: (sum(1 for ok, _w in x.stands if not ok), [place_text(p) for p in x.places]))
    return out[:params.max_routes]


def route_words(route: Route, t: Unit) -> str:
    """The route in plain words: "my <kind> on X to Y, then to Z, from where it strikes ..."."""
    if not route.places:
        return f"{unit_words(route.unit)} where it stands, from where it {'strikes' if route.what == 'strike' else 'covers an exit of'} {unit_words(t)}"
    steps = " to " + place_text(route.places[0]) + "".join(f", then to {place_text(p)}" for p in route.places[1:])
    return f"{unit_words(route.unit)}{steps}, from where it {'strikes' if route.what == 'strike' else 'covers an exit of'} {unit_words(t)}"


def _link_check(r: Reading, t: Unit, u: Unit, to: tuple, moved: bool, what: str, struck_now: bool) -> tuple:
    """(link that fails or 0, why) for a unit `u` of the asking side that would stand on `to` and strike the target `t` (what = "strike") or cover one of its exits.
    The strikes of the other units are read on the board as it stands: the hypothetical arrival of `u` does not change what any other unit strikes (a line it blocks or opens
    is not followed). A unit that covers an exit gains only if the target is already struck by the asking side, so for it link 3 also asks that."""
    if moved:
        ok, why = _stand(r, t, u, to)
        if not ok:
            return 2, why
    if what != "strike" and not struck_now:
        return 3, f"{unit_words(u)} would only cover an exit of {unit_words(t)} and nothing strikes it yet"
    guards = [v for v in r.units if v.side == t.side and v != t and t.at in r.strikes_of(v)]
    if guards and r.weight(t) <= r.weight(u):
        return 3, f"{unit_words(t)} is guarded by {unit_words(guards[0])} and is worth no more than {unit_words(u)}"
    return 0, ""


@dataclass(frozen=True)
class Obstacle:
    """Anything whose absence would let a step happen or succeed. `kind` is a describing word (open, not a list the code branches on): it stands on a place passed over; it stands on the landing
    place; it strikes the asker's goal unit; time; other (the honest default); none (no single unit accounts for it). `note` is the fact found, e.g. "without X the step can be made"; `hypothetical`
    marks a finding read on a board without a unit. `dead`: nothing the asker can ask about (no unit accounts for it)."""
    kind: str
    unit: "Unit | None"
    place: str
    hop: int                           # the number of the step (counted from 0); the final strike is one past the last hop
    note: str = ""
    hypothetical: bool = False
    dead: bool = False
    joint: tuple = ()                  # the places of the whole set, when this unit is one of several that must all go for the step to be made


def _obstacle_words(o: Obstacle, side_of_asker: str = "") -> str:
    if o.unit is None:
        return f"{o.kind}: {o.note}" if o.note else o.kind
    return f"{unit_words(o.unit)} {o.kind}" + (f" ({o.note})" if o.note else "")


def geometric_routes(r: Reading, t: Unit, u: Unit, arrival: str, goal_place: str, what: str, params: Params) -> list:
    """LAYER TWO's first reading: the shortest geometric routes of `u` to `arrival` (the medium, units ignored), each with its obstacles as the board stands and the place-to-stand reading
    of every landing; the routes with the fewest obstacles first."""
    med = r.env.medium()
    kind, side, a = u.kind, r.absolute(u.side), place_text(u.at)
    at = {place_text(v.at): v for v in r.units if v != u}
    takes = (kind, side) in med.takes
    paths = med.routes(kind, side, a, arrival) if arrival != a else [()]
    out = []
    for path in paths:
        obs, over, prev = [], [], a
        for i, landing in enumerate(path):
            ov = med.over(kind, side, prev, landing)
            over.append(ov)
            obs += [Obstacle("stands on a place passed over", at[x], x, i) for x in ov if x in at]
            v, take_only = at.get(landing), med.hops(kind, side, prev)[landing]
            if v is None:
                obs += [Obstacle("no unit to take there", None, landing, i, dead=True)] if take_only else []
            elif v.side == u.side:
                obs.append(Obstacle("stands on the landing place", v, landing, i))
            elif takes and not take_only:
                obs.append(Obstacle("stands on the landing place and this step may not take it", v, landing, i, dead=True))
            else:
                obs.append(Obstacle("stands on the landing place", v, landing, i))
            prev = landing
        if not takes:
            ov = med.over(kind, side, arrival, goal_place)
            obs += [Obstacle("stands on a place passed over", at[x], x, len(path)) for x in ov if x in at]
        places = tuple(tuple(x) for x in path)
        out.append(Route(u, places, tuple(_stand(r, t, u, p) for p in places), what, tuple(over), tuple(obs), goal_place))
    out.sort(key=lambda x: (sum(1 for o in x.obstacles if o.dead), len(x.obstacles), sum(1 for ok, _w in x.stands if not ok), [place_text(p) for p in x.places]))
    return out


def _old_reach(r: Reading, q: Question, params: Params) -> set:
    """The units (kind, place) of the asking side that the move tables say can come within reach_k moves to a place that strikes the target (the board frozen): what level 1 read before the
    medium; kept to count where the two readings disagree."""
    t, out = q.target, set()
    for u in r.units:
        if u.side != q.asker:
            continue
        for table in r.go[:min(params.reach_k, len(r.go))]:
            if any(t.at in r.acts.get((u.kind, u.side, to), ()) for to in table.get((u.kind, u.side, u.at), ())):
                out.add((u.kind, place_text(u.at)))
    return out


def _level1_geometry(q: Question, r: Reading, params: Params) -> tuple:
    """Level 1 by the medium. Layer one: for each unit of the asker, the places that would strike the target (units ignored) and the hops to each; farther than reach_k is too far in time.
    Layer two: each way's routes with the obstacles on them (listed here, asked about at level 2). The three links then judge the arrival place as before."""
    t, k, side = q.target, params.reach_k, side_word(q.asker)
    med = r.env.medium()
    now = r.strikers_of(t, q.asker)
    exits = r.exits_of(t)
    open_exits = exits - r.struck_by(q.asker)
    goals = [(place_text(t.at), "strike")] + ([(place_text(e), "cover an exit") for e in sorted(open_exits)] if now else [])
    supply: list = [(0, u, u.at, "strike") for u in now]
    routes: dict = {}
    far: list = []
    reach: set = set()
    seen_ways: set = set()
    for u in r.units:
        if u.side != q.asker or u in now:
            continue
        side_abs, a, nearest = r.absolute(u.side), place_text(u.at), None
        for goal, what in goals:
            for p in sorted(med.strike_places(u.kind, side_abs, goal)):
                d = med.distance(u.kind, side_abs, a, p)
                if d is None:
                    continue
                nearest = d if nearest is None or d < nearest else nearest
                if d > k or (u, p, what) in seen_ways or (what == "cover an exit" and (u, p, "strike") in seen_ways):
                    continue
                seen_ways.add((u, p, what))
                found = geometric_routes(r, t, u, p, goal, what, params)
                way = (d, u, tuple(p), what)
                if not found:
                    continue
                supply.append(way)
                routes[way] = found
                reach.add((u.kind, a))
        if nearest is not None and nearest > k:
            far.append((nearest, u))
    detail: dict = {"reach_geo": reach, "reach_old": _old_reach(r, q, params) if params.compare_old else None, "far": sorted(far, key=lambda x: (x[0], x[1].kind, x[1].at))}
    supply.sort(key=lambda x: (x[0], len(routes[x][0].obstacles) if x in routes else 0, x[1].kind, x[1].at, x[2]))
    if len(supply) == len(now):
        if now:
            return IDLE, f"already struck by {_names(now)}; no other unit of {side} can join within {k} hops", {**detail, "supply": supply, "routes": {}}
        if far:
            hops, u = detail["far"][0]
            return CLOSED, f"{unit_words(u)} is {hops} hops away: too far in time (reach is {k})", {**detail, "supply": [], "failed_at": 1, "too_far": hops}
        return CLOSED, f"no unit of {side} can come to a place that strikes it", {**detail, "supply": [], "failed_at": 1}
    passed, failed = [], []
    for w in supply:
        if w[0] == 0 and w not in routes:
            passed.append(w)
            continue
        link, why = _link_check(r, t, w[1], w[2], w[0] > 0, w[3], bool(now))
        (failed.append((link, why, w[3])) if link else passed.append(w))
    detail.update(supply=passed, reach=len(supply), failed=failed, links=[len(supply), sum(1 for f in failed if f[0] > 2) + len(passed), len(passed)], routes={w: routes[w] for w in passed if w in routes},
                  rank={w: n for n, w in enumerate(supply)})
    if not passed:
        pool = [f for f in failed if f[2] == "strike"] or failed
        top = max(f[0] for f in pool)
        link, why, _ = [f for f in pool if f[0] == top][-1]
        detail["failed_at"] = link
        return CLOSED, f"{len(supply)} ways for {side} reach it, none passes all three links; the last fails at {LINKS[link]}: {why}", detail

    def way_words(w) -> str:
        found = detail["routes"].get(w)
        if not found:
            return f"{unit_words(w[1])} could {w[3]} from {place_text(w[2])} in {w[0]}"
        if w[0] == 0:
            return f"{unit_words(w[1])} already bears on it, but {_obstacle_words(found[0].obstacles[0], side)}"
        text = route_words(found[0], t)
        if found[0].obstacles:
            text += f" (in the way: {_obstacle_words(found[0].obstacles[0], side)}{' and ' + str(len(found[0].obstacles) - 1) + ' more' if len(found[0].obstacles) > 1 else ''})"
        return text + (f" (and {len(found) - 1} more routes)" if len(found) > 1 else "")
    shown = "; ".join(way_words(w) for w in passed[:4])
    return LIVE, f"{len(passed)} of {len(supply)} ways for {side} pass reach, a place to stand and a strike that gains: {shown}", detail


def weak_places(q: Question, r: Reading, params: Params) -> list:
    """For the goal unit's own place and each place next to it (where it could move on the empty board, not held by its own side): the units of its side that guard the place and the units of
    the asker that strike it now. Where the guards are not fewer than the strikers, the question is whether (guards - strikers + 1) more units can be brought to strike it: layer one names the
    asker's units and their hops to a place that strikes it. -> [{place, guards, strikers, need, candidates: [(hops, unit, arrival)]}]."""
    med = r.env.medium()
    if med is None:
        return []
    t, k = q.target, params.reach_k
    own = {u.at for u in r.units if u.side == t.side and u != t}
    near = [place_text(t.at)] + sorted(med.hops(t.kind, r.absolute(t.side), place_text(t.at)))
    out = []
    for p in near:
        pt = tuple(p)
        if pt in own:
            continue
        guards = [u for u in r.units if u.side == t.side and u != t and pt in r.strikes_of(u)]
        strikers = [u for u in r.units if u.side == q.asker and pt in r.strikes_of(u)]
        if len(guards) < len(strikers):
            continue
        cands = []
        for u in r.units:
            if u.side != q.asker or u in strikers:
                continue
            best = None
            for a in sorted(med.strike_places(u.kind, r.absolute(u.side), p)):
                d = med.distance(u.kind, r.absolute(u.side), place_text(u.at), a)
                if d is not None and 0 < d <= k and (best is None or d < best[0]):
                    best = (d, a)
            if best:
                cands.append((best[0], u, best[1]))
        cands.sort(key=lambda x: (x[0], x[1].kind, x[1].at))
        out.append({"place": p, "guards": guards, "strikers": strikers, "need": len(guards) - len(strikers) + 1, "candidates": cands})
    return out


def level1(q: Question, r: Reading, params: Params) -> tuple:
    status, text, detail = _level1(q, r, params)
    if q.goal and params.weak_places and r.env.medium() is not None:
        detail["weak"] = weak_places(q, r, params)
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
    if params.geometry and params.links and r.env.medium() is not None:
        return _level1_geometry(q, r, params)
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
    detail["routes"] = {w: routes_of(r, t, w[1], w[2], w[0], params, w[3]) for w in passed if w[0] >= 1}

    def way_words(w) -> str:
        found = detail["routes"].get(w)
        if w[0] >= 2 and found:
            return route_words(found[0], t) + (f" (and {len(found) - 1} more routes)" if len(found) > 1 else "")
        return f"{unit_words(w[1])} could {w[3]} from {place_text(w[2])} in {w[0]}"
    shown = "; ".join(way_words(w) for w in passed[:4])
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
    return {"value": value, "line": line, "source": source, "open": fs.stopped > 0, "nodes": fs.nodes}


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


# ---------------------------------------------------------------------------------------------------------------------------------------------- routes: the whole way, answer by answer

WON, WAITING_V, CONFIRMED_V, REFUTED_V, UNSETTLED_V = "won", "waiting", "confirmed", "refuted", "not settled"
ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth"}


class _Capped(Exception):
    """The node cap of one way was reached: nothing more is read, the way is not settled."""


@dataclass
class Stop:
    """One answer of the owner (or the board as it stands) that stops a route. `kind` hard: it costs the owner nothing (the unit is taken for nothing, the exchange at the end pays
    nothing, the game ends); delay: the step cannot be made now (the place is taken, the line is shut, the unit may not move) but nothing is lost."""
    kind: str
    answer: "str | None"               # the owner's move; None: the board as it stands
    after: int                         # steps made when the answer came
    step: "int | None"                 # delay: the number of the step that cannot be made
    reason: str                        # unit_taken, game_over, exchange_stopped, place_taken, line_shut, may_not_move
    places: tuple                      # the places named by the reason
    words: str
    mover: tuple = ()                  # (origin, destination) of the answer


@dataclass
class RouteResult:
    route: Route
    verdict: str
    value: float
    line: tuple
    stops: list
    nodes: int
    tried: int = 0
    capped: bool = False


class _Walk:
    """Follow one route for real from `base` (the asker to move): the asker plays each step of the route, the owner answers every legal move (the likely stoppers first), and after the last
    step and one more answer the asker's real exchange frame runs on the target (followed where it goes). Value to the asker: the least over the owner's answers; a step that cannot be
    made after an answer is a DELAY (kept in `stops`, not counted in the least); an answer that brings the value to 0 or below is a HARD stop and ends that branch."""

    def __init__(self, r: Reading, t: Unit, asker: str, route: Route, params: Params, base: str, cap: int):
        self.r, self.t, self.asker, self.route, self.params, self.base, self.cap = r, t, asker, route, params, base, cap
        self.codec = r.env.codec
        self.places = [place_text(p) for p in route.places]
        self.origin = place_text(route.unit.at)
        self.unit_worth = r.worth_of(route.unit.kind)
        self.stops: list = []
        self.nodes = 0
        self.tried = 0                                                 # the owner's answers examined
        self.mine = "my" if t.side == SELF else "their"                  # the owner of the target answers: its units are "their" when the asker is me
        self._frames: dict = {}

    def tick(self, n: int = 1) -> None:
        self.nodes += n
        if self.nodes > self.cap:
            raise _Capped()

    def frame(self, where: str) -> "_RealExchange":
        if where not in self._frames:
            self._frames[where] = _RealExchange(self.r, tuple(where))
        return self._frames[where]

    def after_words(self, after: int) -> str:
        return "after the last step" if after >= len(self.places) else f"after the {ORDINALS.get(after, str(after) + 'th')} step"

    def answer_words(self, pos: str, move: str, after: int) -> str:
        info = self.codec.move_info(pos, move)
        verb = f"takes on {info['destination']}" if info["captured"] else f"goes to {info['destination']}"
        return f"{self.mine} {info['kind']} {verb} {self.after_words(after)}"

    def hard(self, pos: str, move: str, after: int, reason: str, text: str, places: tuple = ()) -> None:
        self.stops.append(Stop("hard", move, after, None, reason, places, f"{self.answer_words(pos, move, after)}{text}", self.codec.move_ends(move)))

    def run(self) -> RouteResult:
        capped, value, line = False, None, ()
        try:
            value, line = self.step(self.base, 0, self.origin, place_text(self.t.at), None, ())
        except _Capped:
            capped = True
        hard = [x for x in self.stops if x.kind == "hard"]
        if hard:
            verdict = REFUTED_V
        elif capped:
            verdict = UNSETTLED_V
        elif self.stops:
            verdict = WAITING_V
        else:
            verdict = CONFIRMED_V
        return RouteResult(self.route, verdict, 0.0 if value is None or hard else value, line, self.stops, self.nodes, self.tried, capped)

    def step(self, pos: str, i: int, at: str, where: str, by: "tuple | None", line: tuple) -> tuple:
        """-> (value to the asker or None when the step cannot be made, the line). `pos`: the asker to move, its unit on `at`, the target on `where`; `by`: (position, move) of the answer that came before."""
        r, codec, dest = self.r, self.codec, self.places[i]
        mv = codec.find_move(pos, at, dest)
        if mv is None:
            reason, named = codec.why_not(pos, at, dest)
            what = {"place_taken": f"the place {dest} is taken", "line_shut": f"the line to {dest} is shut by {', '.join(named)}", "may_not_move": f"the unit on {at} may not move",
                    "unit_gone": f"no unit of mine stands on {at}"}[reason]
            came = (self.answer_words(by[0], by[1], i) + ": ") if by else "as the board stands: "
            self.stops.append(Stop("delay", by[1] if by else None, i, i + 1, reason, tuple(named), f"{came}the {ORDINALS.get(i + 1, str(i + 1) + 'th')} step cannot be made, {what}", codec.move_ends(by[1]) if by else ()))
            return None, line
        line = line + (mv,)
        child = r.state_of(pos, mv)
        if child[3] is not None:
            return (float(BIG_VALUE) if 100 in list(child[3]) else 0.0), line
        if dest == where:                                              # the step itself takes the target: the owner may take back
            ex = _solve(r, self.frame(where), child[0], lambda p, m: m, self.params, "given")
            r.frame_log.pop()
            self.tick(ex["nodes"])
            return max(0.0, r.worth_of(self.t.kind) - ex["value"]), line + tuple(ex["line"])
        worst: "tuple | None" = None
        for a in codec.rank_answers(child[0], list(child[2]), dest, self.places[i + 1:], where):
            self.tried += 1
            self.tick()
            nxt = r.state_of(child[0], a)
            ofrom, oto = codec.move_ends(a)
            if nxt[3] is not None:
                self.hard(child[0], a, i + 1, "game_over", ", and the game ends")
                return 0.0, line + (a,)
            if oto == dest:                                            # the travelling unit is taken: what does it cost the owner
                ex = _solve(r, self.frame(dest), nxt[0], lambda p, m: m, self.params, "given")
                r.frame_log.pop()
                self.tick(ex["nodes"])
                v, ln = ex["value"] - self.unit_worth, (a,) + tuple(ex["line"])
                if v <= 0:
                    self.hard(child[0], a, i + 1, "unit_taken", ", and nothing is taken back" if ex["value"] <= 0 else ", and what is taken back is worth less")
                    return 0.0, line + ln
            else:
                where2 = oto if ofrom == where else where
                if i + 1 < len(self.places):
                    v, ln = self.step(nxt[0], i + 1, dest, where2, (child[0], a), line + (a,))
                    if v is None:
                        continue
                    if v <= 0:
                        return 0.0, ln
                    ln = ln[len(line):]
                elif codec.kind_at(nxt[0], where2) is None:
                    self.hard(child[0], a, i + 1, "exchange_stopped", ", and the target is gone")
                    return 0.0, line + (a,)
                else:
                    ex = _solve(r, self.frame(where2), nxt[0], lambda p, m: m, self.params, "given")
                    r.frame_log.pop()
                    self.tick(ex["nodes"])
                    v, ln = ex["value"], (a,) + tuple(ex["line"])
                    if v <= 0:
                        pos0, mv0, after0 = child[0], a, i + 1
                        if by is not None:                             # was the exchange already dead before this last answer (the kernel's pass gives the asker the move)? then the answer before it is the one that bites
                            got = CE.passed(r.env.variant, child[0], [])
                            if got:
                                ex0 = _solve(r, self.frame(where), got[0], lambda p, m: m, self.params, "given")
                                r.frame_log.pop()
                                self.tick(ex0["nodes"])
                                if ex0["value"] <= 0:
                                    pos0, mv0, after0 = by[0], by[1], i
                        self.hard(pos0, mv0, after0, "exchange_stopped", ", and the exchange on the target then pays nothing")
                        return 0.0, line + ln
            if worst is None or v < worst[0]:
                worst = (v, ln)
        if worst is None:
            return None, line
        return worst[0], line + worst[1]


def _conditions(stops: list, route: Route) -> list:
    """What a waiting route waits on, from the delays the search saw: [(kind, places, words)]. A place vacated, a line cleared, the unit able to move again, and for a later step the
    steps before it made."""
    out, seen = [], set()
    for x in stops:
        if x.kind != "delay":
            continue
        if x.reason == "place_taken":
            cond = ("vacated", x.places, f"the place {', '.join(x.places)} is vacated")
        elif x.reason == "line_shut":
            cond = ("cleared", x.places, f"the line is cleared: {', '.join(x.places)} empty")
        else:
            cond = ("freed", x.places, f"the unit on {', '.join(x.places)} may move again")
        for c in ([("step", (place_text(route.places[x.step - 2]),), f"the unit has made the first {x.step - 1} steps") ] if x.step and x.step > 1 else []) + [cond]:
            if c[:2] not in seen:
                seen.add(c[:2])
                out.append(c)
    return out


def search_way(r: Reading, t: Unit, asker: str, way: tuple, routes: list, params: Params) -> "dict | None":
    """Follow the routes of a way (at most `params.max_routes`, the best-ordered first) from the base position, within `params.route_nodes` nodes over all of them. None when the asker is
    not to move and the kernel refuses its pass (the caller reads the way after its first move, marked). Otherwise a dict: verdict (CONFIRMED_V, REFUTED_V, WAITING_V or UNSETTLED_V),
    value, line, route, the answer that stops it, `alts` (what it waits on), `breaks` (the answers that stop it), the places it read, nodes and seconds."""
    base = r.position if asker == SELF else r.passed_position()
    if base is None:
        return None
    t0, used, results = time.perf_counter(), 0, []
    for route in routes[:params.max_routes]:
        res = _Walk(r, t, asker, route, params, base, max(params.route_nodes - used, 0)).run()
        used += res.nodes
        results.append(res)
        if res.verdict in (CONFIRMED_V, UNSETTLED_V):
            break
    pick = next((x for x in results if x.verdict == CONFIRMED_V), None) or next((x for x in results if x.verdict == WAITING_V), None) \
        or next((x for x in results if x.verdict == UNSETTLED_V), None) or (results[0] if results else None)
    if pick is None:
        return {"verdict": REFUTED_V, "value": 0.0, "line": (), "route": None, "answer": "no route can be followed", "alts": [], "breaks": [], "nodes": 0, "seconds": time.perf_counter() - t0,
                "reads": frozenset(), "routes_tried": 0, "stops": []}
    stops = pick.stops
    hard = [x for x in stops if x.kind == "hard"]
    alts = _conditions(stops, pick.route) if pick.verdict == WAITING_V else []
    breaks = [x for x in stops if x.kind == "hard" or pick.verdict == WAITING_V]
    reads = {place_text(t.at), place_text(pick.route.unit.at)} | {place_text(p) for p in pick.route.places}
    for c in alts:
        reads |= set(c[1])
    for x in breaks:
        reads |= set(x.mover) | set(x.places)
    answer = hard[0].words if hard else (stops[0].words if stops and pick.verdict == WAITING_V else "")
    return {"verdict": pick.verdict, "value": pick.value, "line": tuple(pick.line), "route": pick.route, "answer": answer, "alts": alts, "breaks": breaks, "nodes": used,
            "tried": sum(x.tried for x in results), "cut": int(any(x.capped for x in results)), "node_cap": params.route_nodes,
            "seconds": time.perf_counter() - t0, "reads": frozenset(reads), "routes_tried": len(results), "stops": stops}


# ---------------------------------------------------------------------------------------------------------------------------------------------- waiting across moves

class Pool:
    """THE SHEET'S MEMORY: discovery/facts.py's FactStore (October's event-driven store), used as it stands. A way is filed as a `claim` (confirmed), a `kill` (refuted) or a `gap` (waiting, or cut by
    the node cap or the follow-up depth, with `resume`: the route, the node cap and the nodes used); the read set is the places the finding depended on and the signature is built from the codec's
    `cell_reader`. A move is the event: `advance` calls `invalidate(diff, permanent=True)` with the codec's diff, which flags every fact that read a changed place. After it, a fact not flagged is served
    as it is; a flagged one is read again only when a live question asks for it (it is served after the cheap comparison of its cells if they are in fact unchanged: `revalidated`); `queue()` lists the
    flagged ones by chain stake. The whole graph is filed too (`file_graph`): a node a fact, the typed edges in `just` and in the payload, the stake bounds in the payload, `stake` the high bound.
    The store is imported lazily (facts.py imports a game library at module level). Two things of the store do not fit and are bridged here: its `lookup` never serves a `gap` (they are for
    resuming), so a waiting way is fetched by `_find` through the store's own groups and signatures; and its `advance(board)` takes a game board, so `advance` here is the codec's diff + `invalidate`."""

    def __init__(self, codec):
        from catspace.incubator.analyst.discovery import facts as FS
        self.store = FS.FactStore()
        self._signature = FS.signature
        self.codec = codec
        self.position: "str | None" = None
        self._cell = None
        self.ply = -1
        self.kept = self.revalidated = self.reread = self.new = 0      # asked for: served unflagged, served after a flagged fact's cells matched, read again, never seen
        self.flagged = self.queued = 0
        self.audited = self.audit_differed = 0
        self.audit_every = 0                                           # n > 0: every n-th served fact is read afresh and compared (the measure's test of the read sets)
        self._would_keep = 0
        self.born_log: list = []                                       # (ply, key, verdict word, value) of every way filed
        self.queue_reread = 0                                          # facts read again from the queue by a budgeted refresh (0 unless `refresh` is called)

    def cell(self, position: str):
        if self._cell is None or self._cell[0] != position:
            self._cell = (position, self.codec.cell_reader(position))
        return self._cell[1]

    def _find(self, key: tuple, cell):
        """The newest fact of `key` (any kind, a gap too) whose read cells on this board equal its signature: the store's `lookup` without its filter on gaps."""
        groups = self.store.groups.get((key, True))
        best = None
        for (reads, _occ), sg in (groups or {}).items():
            f = sg.get(self._signature(cell, reads))
            if f is not None and (best is None or f.id > best.id):
                best = f
        return best, bool(groups)

    def lookup(self, key: tuple, position: str) -> tuple:
        """-> (fact or None, "kept" | "revalidated" | "reread" | "new")."""
        f, seen = self._find(key, self.cell(position))
        if f is None:
            if seen:
                self.reread += 1
                return None, "reread"
            self.new += 1
            return None, "new"
        if f.id in self.store.stale:
            self.store.stale.discard(f.id)
            self.revalidated += 1
            return f, "revalidated"
        self.kept += 1
        return f, "kept"

    def add(self, key: tuple, question: tuple, verdict: dict, position: str, just: tuple = ()):
        """File a way: claim, kill or gap by its verdict."""
        reads = frozenset(verdict["reads"]) or frozenset({key[0]})
        sig = self._signature(self.cell(position), sorted(reads))
        word = verdict["verdict"]
        kind = "claim" if word == CONFIRMED_V else ("kill" if word == REFUTED_V else "gap")
        route = verdict.get("route")
        resume = ({"route": [place_text(p) for p in route.places] if route else [], "node_cap": verdict.get("node_cap"), "nodes_used": verdict.get("nodes"), "position": position,
                   "step": min((x.step for x in verdict.get("stops", []) if x.step), default=None)} if kind == "gap" else None)
        hi = max(float(verdict["value"]), 0.0)
        f = self.store.add(kind, key, True, f"{word}: {verdict.get('answer', '')}"[:120], value=int(round(verdict["value"])), reads=reads, sig=sig, just=just, stake=int(round(hi)), under="way", pk=position,
                           payload={"way": True, "question": question, "verdict": verdict, "stake_low": 0.0 if kind != "claim" else hi, "stake_high": hi, "waits_on": [c[2] for c in verdict.get("alts", [])],
                                    "breaks": [x.words for x in verdict.get("breaks", []) if hasattr(x, "words")]}, resume=resume)
        self.born_log.append((self.ply, key, word, verdict["value"]))
        return f

    def ways(self) -> list:
        """The live (not superseded) way facts."""
        return [f for f in self.store.facts.values() if not f.superseded and f.payload.get("way")]

    def invalidate(self, diff) -> list:
        new = self.store.invalidate(diff, permanent=True)
        self.flagged += len(new)
        return new

    def advance(self, position: str) -> list:
        """The game moves on: the codec's diff of the two positions is the event; what read a changed place is flagged stale for good."""
        new = []
        if self.position is not None:
            new = self.invalidate(self.codec.changed_places(self.position, position))
            self.queued += len(self.store.queue())
        self.position = position
        self.ply += 1
        return new

    def queue(self) -> list:
        """The stale facts and open gaps, by chain stake, highest first (the store's own order)."""
        return self.store.queue()

    def file_graph(self, sheet) -> int:
        """File every node of the sheet's graph as a fact: kind by its verdict, `just` the facts of the nodes it follows from, the edge types, the bounds and the way's conditions in the payload,
        `stake` the high bound, the read set the places of the unit and the way's route. -> facts filed."""
        r = sheet.reading
        cell = self.cell(r.position)
        ids: dict = {}
        for n in r.graph.nodes:
            word = n.verdict
            kind = "kill" if word in (REFUTED_V, "dead", CLOSED) else ("gap" if word in (WAITING_V, UNSETTLED_V, UNREAD, IDLE, "cheap layers only") else "claim")
            reads = frozenset(n.reads) | (frozenset({place_text(n.target.at)}) if n.target is not None else frozenset())
            incoming = [(e.kind, e.src) for e in r.graph.edges if e.dst == n.id]
            just = tuple(ids[src] for _k, src in incoming if src in ids)
            hi = n.stake_high if n.stake_high is not None else 0.0
            f = self.store.add(kind, ("node", n.kind, n.asks), True, n.asks[:100], value=int(round(n.stake_low)), reads=reads or None, sig=self._signature(cell, sorted(reads)) if reads else (), just=just,
                               stake=int(round(hi)), under=n.kind, pk=r.position,
                               payload={"node": n.kind, "answer": n.answer, "verdict": n.verdict, "edges": incoming, "stake_low": n.stake_low, "stake_high": n.stake_high, "chance": n.chance,
                                        "tried": n.tried, "cut": n.cut, **n.data},
                               resume=({"route": n.data.get("route"), "node_cap": n.data.get("node_cap"), "nodes_used": n.data.get("nodes")} if kind == "gap" and n.data.get("route") is not None else None))
            ids[n.id] = f.id
        return len(ids)

    def stats(self) -> dict:
        return {"kept": self.kept, "revalidated": self.revalidated, "reread": self.reread, "new": self.new, "flagged": self.flagged, "queued": self.queued, "audited": self.audited,
                "audit_differed": self.audit_differed, "facts": len(self.store.facts), "stale_now": len(self.store.stale)}


def way_verdict(r: Reading, t: Unit, asker: str, way: tuple, routes: list, params: Params) -> "dict | None":
    """The verdict of a way of more than one move, through the pool (the fact store) when the reading has one: a fact not flagged by a move is served as it is (marked `kept`), a flagged one whose
    cells are in fact unchanged is served after the comparison (`revalidated`), one whose cells changed is read again (`reread`), a new way is read; every verdict is filed (claim, kill or gap)."""
    pool = r.pool
    key = (place_text(t.at), r.absolute(t.side), place_text(way[1].at), place_text(way[2]), way[3])
    state = None
    if pool is not None:
        fact, state = pool.lookup(key, r.position)
        if fact is not None:
            stored = fact.payload["verdict"]
            if pool.audit_every:
                pool._would_keep += 1
                if pool._would_keep % pool.audit_every == 0:
                    fresh = search_way(r, t, asker, way, routes, params)
                    if fresh is not None:
                        pool.audited += 1
                        pool.audit_differed += fresh["verdict"] != stored["verdict"]
            return dict(stored, pool=state)
    v = search_way(r, t, asker, way, routes, params)
    if v is None:
        return None
    if pool is not None:
        v["pool"] = state
        pool.add(key, (place_text(t.at), r.absolute(t.side), r.absolute(asker)), v, r.position)
    return v


# ---------------------------------------------------------------------------------------------------------------------------------------------- layer two: the obstacles, one follow-up question each

UNREAD = "not read"


def question_node(r: Reading, q: Question) -> Node:
    n = r.graph.node("question", q.label(), key=("q", id(q)), target=q.target, asker=q.asker)
    if n.stake_high is None:
        n.stake_high = q.stake
    q.node = n.id
    return n


def way_node(r: Reading, q: Question, way: tuple) -> Node:
    g, u = r.graph, way[1]
    arrive = "already bears on it" if way[0] == 0 else f"{way[0]} hop{'s' if way[0] != 1 else ''} to {place_text(way[2])}"
    n = g.node("way", f"can {unit_words(u)} come to bear on {unit_words(q.target)} ({arrive})", key=("w", id(q), way[:4]), target=q.target, asker=q.asker)
    g.link(question_node(r, q), n, "needs", because=f"the attack needs {unit_words(u)} to arrive" if way[0] else f"{unit_words(u)} is the unit that bears on it")
    return n


def obstacle_node(r: Reading, q: Question, way: tuple, ob: Obstacle) -> Node:
    g = r.graph
    n = g.node("obstacle", _obstacle_words(ob, q.asker), key=("o", id(q), ob.kind, ob.place))
    g.link(way_node(r, q, way), n, "blocked_by", because=_obstacle_words(ob, q.asker), gone=unit_words(ob.unit) if ob.unit else "the obstacle",
             then=f"the route would take {way[0]} hop{'s' if way[0] != 1 else ''}" if way[0] else "it would bear on it at once")
    return n


def _by_place(r: Reading) -> dict:
    return {place_text(u.at): u for u in r.units}


def _move_aside(r: Reading, pos: str, h: str, avoid: set, v: "Unit | None", params: Params) -> tuple:
    """The counterfactual for a unit the kernel will not read the board without: stand it on another place instead. The candidates are the empty places its own hops reach (the medium),
    none of them a place of the route or passed over by it; those the other side does not strike come first (the board is read as the kernel takes it, so a struck place may be refused or
    leave the step unmade), at most `params.aside_places` tried. -> ((place, position) of the first the kernel takes as a position, or None, kernel calls used)."""
    med = r.env.medium()
    if v is None or med is None or params.aside_places < 1:
        return None, 0
    codec, side = r.env.codec, r.absolute(v.side)
    struck = {place_text(x) for x in r.struck_by(OTHER if v.side == SELF else SELF)}
    cands = sorted((p for p in med.hops(v.kind, side, h) if p not in avoid and codec.kind_at(pos, p) is None), key=lambda p: (p in struck, p))
    calls = 0
    for p in cands[:params.aside_places]:
        moved = codec.relocate(pos, h, p)
        if moved is None:
            continue
        calls += 1
        st = r.try_state(moved)
        if st is not None and st[3] is None:
            return (p, moved), calls
    return None, calls


def _without_all(r: Reading, pos: str, group: list, avoid: set, by_place: dict, params: Params) -> tuple:
    """The board with every unit of `group` (places) out of the way, as one hypothetical question to the kernel: each is removed, and a unit whose removal the kernel refuses as not a position
    (a goal unit) is moved aside instead (the same machinery as the single removal). -> (position or None when one cannot be got out of the way, {place: where it was moved aside}, kernel calls)."""
    codec, calls, aside = r.env.codec, 0, {}
    for h in group:
        gone = codec.without(pos, h)
        calls += 1
        if r.try_state(gone) is not None:
            pos = gone
            continue
        moved_to, extra = _move_aside(r, pos, h, avoid | set(group), by_place.get(h), params)
        calls += extra
        if moved_to is None:
            return None, aside, calls
        aside[h] = moved_to[0]
        pos = moved_to[1]
    return pos, aside, calls


def counterfactual_obstacles(r: Reading, q: Question, way: tuple, route: Route, params: Params, base: str) -> tuple:
    """The obstacles of a route, FOUND BY THE COUNTERFACTUAL, not listed by shape. Each step of the route (a hop, then the final strike on the goal place) is asked of the kernel on `base` (for a
    later step on a hypothetical board with the unit already on the step's first place). A step the kernel offers is not blocked. For a step it does not offer, every other unit is removed in
    turn (on a hypothetical board: never a position of the game) and the kernel is asked again: each unit whose removal lets the step be made is an obstacle, whatever the reason ("without X the
    step can be made"), described by a word read from facts already held (it stands on a place passed over / on the landing place / strikes the asker's goal unit) with `other` as the default.
    If no single removal does it, pairs up to `params.pair_cap` are tried, else the step is recorded as one no single unit accounts for. A question examines at most `params.max_blocked_steps`
    blocked steps; the rest are returned as cut (not read). -> (obstacles, cut)."""
    codec, variant, med = r.env.codec, r.env.variant, r.env.medium()
    origin, unit = place_text(route.unit.at), route.unit
    side = r.absolute(unit.side)
    by_place = _by_place(r)
    goal = route.goal or place_text(q.target.at)
    arrival = place_text(route.places[-1]) if route.places else origin
    steps = [(origin if i == 0 else place_text(route.places[i - 1]), place_text(p), route.over[i] if i < len(route.over) else (), i) for i, p in enumerate(route.places)]
    steps.append((arrival, goal, med.over(unit.kind, side, arrival, goal) if (unit.kind, side) not in med.takes else (), len(route.places)))
    out, cut = [], False
    asker_goal = r.goal_unit(q.asker)
    for a_place, b_place, over, idx in steps:
        t0 = time.perf_counter()
        calls = 1
        pos = base if a_place == origin else codec.relocate(base, origin, a_place)
        if pos is None:
            continue
        try:
            st = r.state_of(pos)
        except Exception:
            continue
        if st[3] is not None or codec.find_move(pos, a_place, b_place) in list(st[2]):
            continue                                                   # the kernel offers the step: nothing blocks it
        if q.blocked >= params.max_blocked_steps:
            cut = True
            r.cf_cut += 1
            continue
        q.blocked += 1
        found, rejected, aside = [], 0, {}
        route_places = {place_text(p) for p in route.places} | {x for ov in route.over for x in ov} | {origin, a_place, b_place, goal}
        for h in codec.units(pos):
            if h == a_place:
                continue
            gone = codec.without(pos, h)
            st2 = r.try_state(gone)
            calls += 1
            if st2 is None:
                rejected += 1                                          # the kernel rejects the board without it (a goal unit): not a position
                moved_to, extra = _move_aside(r, pos, h, route_places, by_place.get(h), params)
                calls += extra
                if moved_to is not None:
                    st3 = r.state_of(moved_to[1])
                    calls += 1
                    if st3[3] is None and codec.find_move(moved_to[1], a_place, b_place) in list(st3[2]):
                        found.append(h)
                        aside[h] = moved_to[0]
                continue
            if st2[3] is None and codec.find_move(gone, a_place, b_place) in list(st2[2]):
                found.append(h)
        if not found and params.pair_cap > 0:
            import itertools
            tried = 0
            for first, second in itertools.combinations([h for h in codec.units(pos) if h != a_place], 2):
                if tried >= params.pair_cap:
                    break
                tried += 1
                calls += 1
                try:
                    pos2 = codec.without(codec.without(pos, first), second)
                    st2 = r.state_of(pos2)
                except Exception:
                    continue
                if st2[3] is None and codec.find_move(pos2, a_place, b_place) in list(st2[2]):
                    found = [first, second]
                    break
        joint_n, joint_state = 0, ""
        if not found and params.joint:
            named = [h for h in dict.fromkeys(list(over) + ([b_place] if idx < len(route.places) else [])) if h in by_place and h != a_place]
            if len(named) >= 2:
                joint_n = len(named)
                if joint_n > params.max_joint:
                    joint_state = "cap"
                    out.append(Obstacle("more units in the way than the cap", None, b_place, idx,
                                        f"more than {params.max_joint} units in the way of the step from {a_place} to {b_place} ({joint_n}: {', '.join(unit_words(by_place[h]) for h in named)}); too costly to clear, so the route is refuted by cost", True, True))
                    words = ["more than the cap"]
                    r.cf_log.append({"ms": 1000 * (time.perf_counter() - t0), "calls": calls, "found": 0, "words": words, "pair": False, "rejected": rejected, "aside": len(aside), "joint": joint_n, "joint_state": joint_state})
                    continue
                gone_pos, moved_aside, extra = _without_all(r, pos, named, route_places, by_place, params)
                calls += extra
                st2 = r.try_state(gone_pos) if gone_pos is not None else None
                if st2 is not None and st2[3] is None and codec.find_move(gone_pos, a_place, b_place) in list(st2[2]):
                    joint_state = "joint"
                    members = sorted(named, key=lambda h: (by_place[h].side != q.asker, r.worth_of(by_place[h].kind), h))        # cheapest first: my own units (a move) before theirs (a taking), the lighter first
                    for h in members:
                        v = by_place[h]
                        how = f"moved aside to {moved_aside[h]}" if h in moved_aside else "removed"
                        out.append(Obstacle("one of several units, each of which must go", v, h, idx,
                                            f"the step from {a_place} to {b_place} can be made only with all {len(named)} of {', '.join(unit_words(by_place[x]) for x in members)} out of the way, read on the board with them {how}", True, False, tuple(members)))
                    r.cf_log.append({"ms": 1000 * (time.perf_counter() - t0), "calls": calls, "found": 0, "words": ["several, each must go"] * len(named), "pair": False, "rejected": rejected, "aside": len(aside), "joint": joint_n, "joint_state": joint_state})
                    continue
                joint_state = "failed"
        words = []
        for h in found:
            v = by_place.get(h)
            if v is None:
                continue
            if h in over:
                kind = "stands on a place passed over"
            elif h == b_place:
                kind = "stands on the landing place"
            elif asker_goal is not None and v != asker_goal and asker_goal.at in r.strikes_of(v):
                kind = "strikes the asker's goal unit"
            else:
                kind = "other"
            words.append(kind)
            if h in aside:
                out.append(Obstacle(kind, v, h, idx, f"with {unit_words(v)} out of the way the step from {a_place} to {b_place} can be made, read on the board with {unit_words(v)} moved aside to {aside[h]}", True))
            else:
                out.append(Obstacle(kind, v, h, idx, f"without {unit_words(v)} the step from {a_place} to {b_place} can be made, read on the board without it", True))
        if not words:
            out.append(Obstacle("none", None, b_place, idx, f"no single unit accounts for the step from {a_place} to {b_place}" + (" (pairs tried)" if params.pair_cap else "") + (f"; {rejected} removals were rejected by the kernel as not a position" if rejected else ""), True, True))
            words = ["none"]
        r.cf_log.append({"ms": 1000 * (time.perf_counter() - t0), "calls": calls, "found": len(found) if found else 0, "words": words, "pair": len(found) == 2, "rejected": rejected, "aside": len(aside),
                         "joint": joint_n, "joint_state": joint_state})
    return out, cut


def goal_sign(r: Reading, side: str, params: Params) -> bool:
    """The goal unit of `side` is under the sign: struck now (level 0 live) or at most `max_open_exits` exits open with an asker unit near (level 1 live), as the sheet reads it."""
    g = r.goal_unit(side)
    if g is None:
        return False
    qq = Question(g, OTHER if side == SELF else SELF, 1.0, 0, goal=True)
    return level0(qq, r, params)[0] == LIVE or level1(qq, r, params)[0] == LIVE


def _leaving_child(r: Reading, q: Question, u: Unit, mv: "str | None", base: "str | None") -> tuple:
    """(child position or None, mark): the board after the unit's step `mv` played for real from `base` (the other side then to move: one kernel call), or, when the step is not legal, the
    board without the unit with the other side given the move by the kernel's pass."""
    codec, variant, origin = r.env.codec, r.env.variant, place_text(u.at)
    if base is None:
        return None, "read after the step"
    if mv is not None:
        st = r.state_of(base, mv)
        return (st[0] if st[3] is None else None), "read after the step"
    mark = "read on the board without it (its step is not legal)"
    pos = codec.without(base, origin)
    if r.try_state(pos) is None:
        return None, mark
    key = ("pass", pos)
    if key not in r._frames:
        got = CE.passed(variant, pos, [])
        r._frames[key] = got[0] if got else None
    return r._frames[key], mark


def _lost_after(r: Reading, q: Question, v: Unit, child: str, params: Params) -> "float | None":
    """The value of the other side's exchange on `v`'s place in `child` when that is a newly lost unit (it was not lost where it stands before), else None."""
    if r.env.codec.kind_at(child, place_text(v.at)) is None:
        return None
    if exchange_stands(r, v, q.owner, params)["value"] > 0:
        return None                                                    # it was lost before: not a duty of this unit
    ex = _solve(r, _RealExchange(r, v.at), child, lambda p, m: m, params, "given")
    r.frame_log.pop()
    return ex["value"] if ex["value"] > 0 else None


def _goal_newly_under_sign(r: Reading, q: Question, child: str, params: Params) -> bool:
    try:
        after_reading = r._frames.get(("reading", child))
        if after_reading is None:
            after_reading = Reading(r.env, child, r.worth)
            after_reading._goal_kind = r._goal_kind if r._goal_kind is not None else (r.goal_kind(),)      # the goal's kind is the game's, not the position's: not asked of the kernel again
            r._frames[("reading", child)] = after_reading
        if ("sign", q.asker) not in r._frames:
            r._frames[("sign", q.asker)] = goal_sign(r, q.asker, params)
        return goal_sign(after_reading, OTHER, params) and not r._frames[("sign", q.asker)]
    except ValueError:
        return False


def _leaving_candidates(r: Reading, q: Question, u: Unit, mv: "str | None", params: "Params | None" = None) -> tuple:
    """(units of the asker's side whose standing the leaving can change, whether the goal unit's sign may change). The leaving empties the unit's place and fills its new place (the step's
    end), nothing else. A unit of its side is newly open if the unit guarded it (it strikes it now), shielded it (a unit of the other side has a hop to the place of that unit passing over the
    unit's place: read from the medium), or if the new place lies on the line of a guard of that unit (the step shuts the guard). The goal sign may change when the goal unit is among those, or the
    unit stood, or its step ends, one hop from it or on any line to it, or the unit strikes (now, or from the step's end) a place one hop from it."""
    med = r.env.medium()
    origin = place_text(u.at)
    new = r.env.codec.move_ends(mv)[1] if mv else None
    own = [v for v in r.units if v.side == q.asker and v != u]
    foes = [w for w in r.units if w.side != q.asker]
    strikes = r.strikes_of(u)
    got: set = set()
    for v in own:
        if v.at in strikes:
            got.add(v)                                                 # it guarded it
    if med is None:
        return set(own), True
    for v in own:
        for w in foes:
            if origin in med.over(w.kind, r.absolute(w.side), place_text(w.at), place_text(v.at)):
                got.add(v)                                             # it shielded it from a line
                break
        if new is not None and v not in got:
            for g in own:
                if g != v and v.at in r.strikes_of(g) and new in med.over(g.kind, r.absolute(g.side), place_text(g.at), place_text(v.at)):
                    got.add(v)                                         # its step shuts a guard's line
                    break
    gu = r.goal_unit(q.asker)
    touched = [x for x in (origin, new) if x is not None]
    sign = False
    if gu is not None and gu != u:
        near = set(med.hops(gu.kind, r.absolute(gu.side), place_text(gu.at)))                                  # beside it
        lines = med.lines_into(place_text(gu.at)) if params is not None and params.leaving_wide_sign else set()   # on a line any unit could come along to it (wide: costs a whole reading of the child for most steps)
        side_u = r.absolute(u.side)
        strikes_near = any(place_text(x) in near for x in strikes) or (new is not None and any(new in med.strike_places(u.kind, side_u, p) for p in near))
        sign = gu in got or strikes_near or any(x in near or x in lines for x in touched) or any(x in lines for x in touched)
    return got, sign


def _cost_leaving_full(r: Reading, q: Question, u: Unit, child: str) -> list:
    """The earlier form: the exchange frame for every weighty unit of the asker's side, and the goal sign from a whole reading of the child."""
    duties = []
    for v in r.units:
        if v.side != q.asker or v == u or not r.weighty(v) or v == r.goal_unit(q.asker):
            continue
        value = _lost_after(r, q, v, child, r.params)
        if value is not None:
            duties.append({"unit": v, "stake": r.weight(v), "kind": "newly lost", "value": value})
    g = r.goal_unit(q.asker)
    if g is not None and g != u and _goal_newly_under_sign(r, q, child, r.params):
        duties.append({"unit": g, "stake": r.weight(g), "kind": "goal unit newly under the sign", "value": 0.0})
    return duties


def _cost_leaving_narrow(r: Reading, q: Question, u: Unit, mv: "str | None", child: str, params: Params) -> list:
    cands, sign = _leaving_candidates(r, q, u, mv, params)
    goal = r.goal_unit(q.asker)
    duties = []
    for v in r.units:
        if v not in cands or not r.weighty(v) or v == goal:
            continue
        value = _lost_after(r, q, v, child, params)
        if value is not None:
            duties.append({"unit": v, "stake": r.weight(v), "kind": "newly lost", "value": value})
    if goal is not None and goal != u and sign and _goal_newly_under_sign(r, q, child, params):
        duties.append({"unit": goal, "stake": r.weight(goal), "kind": "goal unit newly under the sign", "value": 0.0})
    return duties


def cost_of_leaving(r: Reading, q: Question, u: Unit, mv: "str | None", base: "str | None", params: Params) -> dict:
    """What changes for the asker's side when `u` is no longer on its place. Its first step `mv` is played for real from `base` (the other side then to move); when the step is not legal the board
    without the unit is used (marked) and the other side is given the move by the kernel's pass. In that position the exchange frame `lost where it stands` is read for the asker's side's units
    that the leaving can change (the narrow reading: the units it guarded or shielded, or whose guard line its step shuts; with `params.leaving_full` every weighty unit and a whole reading of the
    child), and the goal unit's sign when it may have changed. Each unit newly lost, or goal unit newly under the sign, is a DUTY the unit was holding.
    -> {read, mark, duties: [{unit, stake, kind, value}]}"""
    key = ("duty", q.asker, place_text(u.at), mv, params.leaving_full)
    if key in r._frames:
        return r._frames[key]
    r.params = params
    out = {"read": False, "mark": "", "duties": []}
    child, mark = _leaving_child(r, q, u, mv, base)
    if child is None:
        r._frames[key] = out
        return out
    out.update(read=True, mark=mark)
    if params.leaving_full:
        out["duties"] = _cost_leaving_full(r, q, u, child)
    else:
        out["duties"] = _cost_leaving_narrow(r, q, u, mv, child, params)
        if params.leaving_audit_every > 0:
            r._audit_n += 1
            if r._audit_n % params.leaving_audit_every == 0:
                full = _cost_leaving_full(r, q, u, child)
                narrow_set = {(d["unit"], d["kind"]) for d in out["duties"]}
                full_set = {(d["unit"], d["kind"]) for d in full}
                cands, sign = _leaving_candidates(r, q, u, mv, params)
                r.leaving_audit.append({"unit": unit_words(u), "mv": mv, "position": r.position, "narrow": len(narrow_set), "full": len(full_set), "differ": narrow_set != full_set,
                                        "missed": [f"{unit_words(x)} ({k})" for x, k in sorted(full_set - narrow_set, key=lambda z: (z[0].kind, z[0].at))],
                                        "extra": [f"{unit_words(x)} ({k})" for x, k in sorted(narrow_set - full_set, key=lambda z: (z[0].kind, z[0].at))], "candidates": len(cands)})
    r._frames[key] = out
    return out


def takeover_words(r: Reading, q: Question, u: Unit, duty: dict, params: Params) -> str:
    """The cheap layers' answer to "can another unit of the asker take over the duty": the asker's other units that could come, within reach_k hops, to a place that strikes the held unit's place;
    the deeper reading is left open."""
    med = r.env.medium()
    held = place_text(duty["unit"].at)
    if med is None:
        return "not read: no medium"
    got = []
    for v in r.units:
        if v.side != q.asker or v in (u, duty["unit"]):
            continue
        side = r.absolute(v.side)
        if any(0 < (med.distance(v.kind, side, place_text(v.at), p) or 99) <= params.reach_k for p in med.strike_places(v.kind, side, held)):
            got.append(v)
    return (f"{len(got)} other unit{'s' if len(got) != 1 else ''} of {side_word(q.asker)} could come to bear on {held} within {params.reach_k} hops: {_names(got[:3])}" if got
            else f"no other unit of {side_word(q.asker)} can come to bear on {held} within {params.reach_k} hops") + "; whether it would hold is not read"


def ask_obstacle(r: Reading, q: Question, way: tuple, route: Route, ob: Obstacle, params: Params, base: "str | None") -> dict:
    """The follow-up question on one obstacle, by the kernel's real moves from the base position (the asker to move). An own unit in the way: does it have a move that leaves the route
    clear, and what does leaving cost (the other side's exchange frame on the place it lands and on each own unit it was guarding, and the duties it holds, read after the move; at least what that
    costs, the board as it stands)? The other side's unit: will not move for us; can it be taken (the exchange frame on it, started by the asker)? -> {verdict: movable | takeable | dead | not read,
    text, then, cost, more, duties}."""
    if ob.dead or ob.unit is None:
        return {"verdict": "dead", "text": ob.note or "no single unit accounts for it", "then": "", "cost": 0.0, "more": 0, "duties": []}
    if params.obstacle_depth < 1 or base is None:
        return {"verdict": UNREAD, "text": "the follow-up question is not read (depth cap or no base position)", "then": "", "cost": 0.0, "more": 0, "duties": []}
    key = ("obstacle", q.asker, ob.place, tuple(place_text(p) for p in route.places), ob.kind)
    if key in r._frames:
        return r._frames[key]
    codec, u = r.env.codec, ob.unit
    if u.side == q.asker:
        legal = list(r.state_of(base)[2])
        keep = {place_text(p) for p in route.places} | {x for ov in route.over for x in ov} | {place_text(route.unit.at)}
        cands = [m for m in codec.moves_of(ob.place, legal) if codec.move_ends(m)[1] not in keep]
        if not cands:
            out = {"verdict": "dead", "text": f"{unit_words(u)} has no move that leaves the route clear", "then": "", "cost": 0.0, "more": 0, "duties": []}
        else:
            guarded = [g for g in r.units if g.side == u.side and g != u and g.at in r.strikes_of(u)]
            best = None
            for m in cands[:params.max_obstacle_moves]:
                child = r.state_of(base, m)
                if child[3] is not None:
                    cost = 0.0
                else:
                    cost = 0.0
                    for place in [codec.move_ends(m)[1]] + [place_text(g.at) for g in guarded]:
                        if codec.kind_at(child[0], place) is None:
                            continue
                        ex = _solve(r, _RealExchange(r, tuple(place)), child[0], lambda pos, mv: mv, params, "given")
                        r.frame_log.pop()
                        cost = max(cost, ex["value"])
                if best is None or cost < best[0]:
                    best = (cost, m)
                if cost <= 0:
                    break
            duties = cost_of_leaving(r, q, u, best[1], base, params) if params.duties else {"duties": []}
            held = "; it holds " + ", ".join(unit_words(d["unit"]) for d in duties["duties"]) if duties["duties"] else ""
            more = "it can move away, one move more" + (f", at a cost of at least {best[0]:g}" if best[0] > 0 else "") + held
            out = {"verdict": "movable", "text": f"{unit_words(u)}: {more} ({best[1]})", "then": f"the route holds in {way[0]} hops, {way[0] + 1} moves with the one that clears it", "cost": best[0], "more": 1,
                   "duties": duties["duties"]}
    elif u == r.goal_unit(u.side):
        out = {"verdict": "dead", "text": f"{unit_words(u)} will not move for us and cannot be taken (it is the goal unit of the other side)", "then": "", "cost": 0.0, "more": 0, "duties": []}
    else:
        ex = _solve(r, _RealExchange(r, tuple(ob.place)), base, lambda pos, mv: mv, params, "given")
        r.frame_log.pop()
        if ex["value"] > 0:
            out = {"verdict": "takeable", "text": f"{unit_words(u)} will not move for us, but it can be taken: the exchange there is worth {ex['value']:g} to us, one move more",
                   "then": f"the route holds in {way[0]} hops, {way[0] + 1} moves with the taking", "cost": 0.0, "more": 1, "duties": []}
        else:
            out = {"verdict": "dead", "text": f"{unit_words(u)} will not move for us and cannot be taken: the exchange on {ob.place} pays nothing", "then": "", "cost": 0.0, "more": 0, "duties": []}
    r._frames[key] = out
    return out


def _unit_node(r: Reading, q: Question, u: Unit) -> Node:
    return r.graph.node("unit", unit_words(u), key=("u", id(q), u), target=u, asker=q.asker)


def _stop_unit(r: Reading, base: str, res: dict) -> "tuple | None":
    """(unit, place) of the unit that made the refuting answer, read on `base` (None when the answer is the target's own move or the unit is not found)."""
    hard = [x for x in res.get("stops", []) if x.kind == "hard" and x.mover]
    if not hard:
        return None
    place = hard[0].mover[0]
    v = _by_place(r).get(place)
    return (v, place) if v is not None else None


def _settle_way(r: Reading, q: Question, way: tuple, params: Params) -> dict:
    """The verdict of one way at level 2, as a result entry. A clear route of one move is read after its first move (the earlier form). Otherwise every route (the best-ordered first) has its
    obstacles found by the counterfactual and asked about, one follow-up question each; a route with a dead obstacle is dead (that unit is the reason), one with an obstacle not read is not read,
    and the routes left, clear or with answerable obstacles, are followed whole by the route reasoner. A way is never refuted for what was not read."""
    t, asker, j = q.target, q.asker, way[0]
    routes = list(q.findings[1].detail.get("routes", {}).get(way[:4]) or [])
    g = r.graph
    wn = way_node(r, q, way)
    if not params.routes or (j == 1 and (not routes or any(not x.obstacles for x in routes))) or (j >= 1 and not routes):
        key = ("after", t, asker, way[:4])
        if key not in r._frames:
            r._frames[key] = exchange_after(r, t, asker, way, params)
        v = r._frames[key]
        verdict = CONFIRMED_V if v["value"] > 0 else REFUTED_V
        wn.verdict, wn.answer = verdict, f"after its first move the exchange is worth {v['value']:g}" if v["value"] > 0 else "after its first move the exchange pays nothing"
        return {"way": way, "move": v["line"][0] if (v["line"] and v["source"] == "given") else None, "status": verdict, "route": routes[0] if routes else None, **v}
    base = r.position if asker == SELF else r.passed_position()
    if params.duties and j >= 1 and base is not None and routes and routes[0].places:
        early = _duty_reading(r, q, way, routes[0], base, params)             # the first question: what changes if this unit leaves
        if early["duties"] and max(d["stake"] for d in early["duties"]) > q.stake:
            big = max(early["duties"], key=lambda d: d["stake"])
            for ask in ("what obstacles does it meet on its way", "what is the other side likely to do"):
                nn = g.node("follow_up", ask, key=("f", id(q), way[:4], ask), answer=f"not read: leaving drops {unit_words(big['unit'])} (stake {big['stake']:g}), more than the attack could win ({q.stake:g})",
                            verdict=UNREAD, target=q.target, asker=asker)
                g.link(wn, nn, "needs", because="asked after the cost of leaving, which already settles it")
            rt0 = routes[0]
            entry = {"way": way, "move": None, "status": WAITING_V, "verdict": WAITING_V, "value": 0.0, "line": (), "source": "route", "open": False, "route": rt0, "nodes": 0, "seconds": 0.0,
                     "answer": f"{unit_words(way[1])} holds {unit_words(big['unit'])} (stake {big['stake']:g}), more than the attack could win ({q.stake:g}); its route was not read",
                     "alts": [("duty", (place_text(d["unit"].at),), f"something else holds {unit_words(d['unit'])}") for d in early["duties"]], "breaks": [], "obstacles": [], "stops": [],
                     "reads": frozenset({place_text(t.at), place_text(way[1].at)} | {place_text(p) for p in rt0.places} | {place_text(d["unit"].at) for d in early["duties"]}),
                     "duties": early["duties"], "duty_mark": early["mark"], "held_back": True, "skipped": True, "first_obstacles": [], "tried": 0, "cut": 0}
            wn.verdict, wn.answer = WAITING_V, entry["answer"]
            for kind, text, because in _graph_edges(entry):
                g.link(wn, g.node("answer" if kind != "waits_on" else "condition", text, key=("a", id(q), way[:4], kind, text)), kind, because=because)
            for d in early["duties"]:
                fu = g.node("follow_up", "can another unit of mine take over the duty", key=("f", id(q), "takeover", d["unit"]), answer=takeover_words(r, q, way[1], d, params), verdict="cheap layers only", asker=asker)
                g.link(wn, fu, "if_removed", because="the duty relieved", gone=unit_words(way[1]), then="it could travel")
            return entry
    walk, dead, unread, followed, first_obs = [], [], [], [], None
    for rt in routes[:params.max_routes]:
        if base is not None and r.env.medium() is not None:
            obs_list, cut = counterfactual_obstacles(r, q, way, rt, params, base)
        else:
            obs_list, cut = list(rt.obstacles), False
        if first_obs is None:
            first_obs = [(o.unit.side if o.unit else None, o.kind, o.dead) for o in obs_list]
        answers = []
        for n, ob in enumerate(obs_list):
            on = obstacle_node(r, q, way, ob)
            if ob.joint and any(other.joint == ob.joint and reply["verdict"] == "dead" for other, reply in answers):
                on.verdict = "not asked"                                # a member of the set is dead: the route is dead, the rest of the set is not asked
                continue
            ans = ask_obstacle(r, q, way, rt, ob, params, base)
            answers.append((ob, ans))
            fu = g.node("follow_up", "can it be taken" if ob.unit and ob.unit.side != asker else "can it move away", key=("f", id(q), ob.kind, ob.place), answer=ans["text"], verdict=ans["verdict"],
                        target=ob.unit, asker=asker)
            g.link(on, fu, "if_removed", because=ans["text"], gone=unit_words(ob.unit) if ob.unit else "it", then=ans["then"])
            if ans["verdict"] == "dead":
                bound(fu, 0.0, 0.0)
                bound(on, 0.0, 0.0)
            followed.append((ob, ans))
        for members in dict.fromkeys(o.joint for o in obs_list if o.joint):
            asked = [(o, a) for o, a in answers if o.joint == members]
            dead_one = next(((o, a) for o, a in asked if a["verdict"] == "dead"), None)
            total = sum(a["more"] for _o, a in asked)
            gn = g.node("obstacle", f"several units in the way, each of which must go: {', '.join(unit_words(o.unit) for o, _a in asked)}", key=("og", id(q), way[:4], members))
            if dead_one:
                g.link(wn, gn, "blocked_by", because=f"{unit_words(dead_one[0].unit)} cannot be cleared, so the route is dead: {dead_one[1]['text']}", gone="all of them", then="the route would be clear")
                bound(gn, 0.0, 0.0)
            elif len(asked) == len(members):
                g.link(wn, gn, "blocked_by", because=f"{len(members)} units stand in the way and each must go", gone="all of them", then=f"the route holds in {way[0]} hops, {way[0] + total} moves with the ones that clear it (the moves add up)")
            r.joint_log.append({"members": len(members), "asked": len(asked), "dead": dead_one is not None, "moves_more": total})
        if cut:
            cutob = Obstacle("other", None, "", len(rt.places), "blocked steps beyond the cap were not read")
            answers.append((cutob, {"verdict": UNREAD, "text": "not read: more blocked steps than the cap", "then": "", "cost": 0.0, "more": 0, "duties": []}))
        bad = next(((ob, a) for ob, a in answers if a["verdict"] == "dead"), None)
        if bad:
            dead.append((rt, bad))
        elif any(a["verdict"] == UNREAD for _o, a in answers):
            unread.append(rt)
        else:
            walk.append(rt)
    entry: dict
    if j == 0:                                                         # it bears on the target in place, but a unit stands in the line
        if dead:
            ob, a = dead[0][1]
            verdict, text = REFUTED_V, f"{a['text']}"
        elif unread:
            verdict, text = UNSETTLED_V, "the obstacles were not read"
        else:
            verdict, text = WAITING_V, "; ".join(a["text"] for _o, a in followed[:2]) if followed else "nothing stands in the line"
        alts = [("vacated" if a["verdict"] == "movable" else "taken", (ob.place,), f"{unit_words(ob.unit)} {'moves away' if a['verdict'] == 'movable' else 'is taken'}") for ob, a in followed if ob.unit] if verdict == WAITING_V else []
        entry = {"way": way, "move": None, "status": verdict, "verdict": verdict, "value": 0.0, "line": (), "source": "route", "open": verdict == UNSETTLED_V, "answer": text, "alts": alts, "breaks": [],
                 "route": routes[0] if routes else None, "nodes": 0, "seconds": 0.0, "reads": frozenset(ob.place for ob, _a in followed), "obstacles": followed, "stops": []}
    else:
        v = way_verdict(r, t, asker, way, walk, params) if walk else None
        if walk and v is None:                                         # the asker not to move and no pass: read after the first move, marked
            key = ("after", t, asker, way[:4])
            if key not in r._frames:
                r._frames[key] = exchange_after(r, t, asker, way, params)
            v0 = r._frames[key]
            verdict = CONFIRMED_V if v0["value"] > 0 else REFUTED_V
            return {"way": way, "move": None, "status": verdict, "route": routes[0], **v0}
        if v is None:
            if unread:
                v = {"verdict": UNSETTLED_V, "value": 0.0, "line": (), "route": unread[0], "answer": "the obstacles on the routes were not read", "alts": [], "breaks": [], "nodes": 0, "seconds": 0.0,
                     "reads": frozenset(), "routes_tried": 0, "stops": []}
            else:
                rt, (ob, a) = dead[0]
                v = {"verdict": REFUTED_V, "value": 0.0, "line": (), "route": rt, "answer": a["text"], "alts": [], "breaks": [], "nodes": 0, "seconds": 0.0, "reads": frozenset(),
                     "routes_tried": 0, "stops": []}
        elif v["verdict"] == REFUTED_V and unread:
            v = dict(v, verdict=UNSETTLED_V, answer=v["answer"] + f" (the obstacles on {len(unread)} other routes were not read)")
        entry = {**v, "way": way, "status": v["verdict"], "move": v["line"][0] if v["line"] else None, "source": "route", "open": v["verdict"] == UNSETTLED_V, "obstacles": followed,
                 "dead_routes": len(dead), "unread_routes": len(unread)}
    entry["first_obstacles"] = first_obs or []
    wn.verdict, wn.answer = entry["status"], entry.get("answer", "")
    _bound_way(wn, entry)
    for kind, text, because in _graph_edges(entry):
        g.link(wn, g.node("answer" if kind != "waits_on" else "condition", text, key=("a", id(q), way[:4], kind, text)), kind, because=because)
    _refute_counterfactual(r, q, way, entry, base, params)
    return entry


def _duty_reading(r: Reading, q: Question, way: tuple, route: Route, base: str, params: Params) -> dict:
    """The first question about a unit that is to join: what changes if it leaves (the cost of leaving), read after its first step on the route (or on the board without it when the step is not
    legal). Recorded in the graph: the unit holds each duty found (`holds`)."""
    u, codec = way[1], r.env.codec
    mv = codec.find_move(base, place_text(u.at), place_text(route.places[0]))
    if mv is not None and mv not in list(r.state_of(base)[2]):
        mv = None
    cost = cost_of_leaving(r, q, u, mv, base, params)
    if cost["duties"]:
        g = r.graph
        un = _unit_node(r, q, u)
        for d in cost["duties"]:
            dn = g.node("duty", f"it holds {unit_words(d['unit'])}", key=("d", id(q), u, d["unit"]), answer=f"stake {d['stake']:g}, {d['kind']}, {cost['mark']}", target=d["unit"], asker=q.asker)
            g.link(un, dn, "holds", because=f"{unit_words(u)} {d['kind']} when it leaves ({cost['mark']}); stake {d['stake']:g}", gone=unit_words(u), then=f"{unit_words(d['unit'])} would stay safe")
            bound(dn, d["stake"], d["stake"])
    return cost


def _bound_way(wn: Node, entry: dict) -> None:
    """The way node's bound from its verdict: confirmed, both ends at its value; refuted, both at 0; waiting, 0 to what it would be worth if its conditions held; not settled, as it was."""
    wn.tried, wn.cut = entry.get("tried", 0), entry.get("cut", 0)
    wn.reads = frozenset(entry.get("reads", ()))
    rt = entry.get("route")
    wn.data = {"route": [place_text(p) for p in rt.places] if rt is not None else None, "node_cap": entry.get("node_cap"), "nodes": entry.get("nodes"),
               "waits_on": [c[2] for c in entry.get("alts", [])], "breaks": [x.words for x in entry.get("breaks", []) if hasattr(x, "words")]}
    st = entry["status"]
    if st == CONFIRMED_V:
        bound(wn, entry["value"], entry["value"])
    elif st == REFUTED_V:
        bound(wn, 0.0, 0.0)
    elif st == WAITING_V and entry.get("value", 0) > 0:
        bound(wn, 0.0, entry["value"])


def _refute_counterfactual(r: Reading, q: Question, way: tuple, entry: dict, base: "str | None", params: Params) -> None:
    """A refuted route: the refuting answer's unit is the obstacle. Would the route be confirmed without that unit (the route's search rerun on the board without it, the same node cap)? If so the
    graph gets an if_removed edge "without X this is worth V", and the follow-up "can it be taken"; "driven off" and "drawn away" stay open nodes marked not read."""
    if not (params.refute_counterfactual and entry["status"] == REFUTED_V and base is not None and entry.get("route") is not None and entry.get("stops")):
        return
    found = _stop_unit(r, base, entry)
    if found is None or found[1] == place_text(q.target.at):
        return
    u, place = found
    codec = r.env.codec
    try:
        pos = codec.without(base, place)
        r.state_of(pos)
    except Exception:
        return
    res = _Walk(r, q.target, q.asker, entry["route"], params, pos, params.route_nodes).run()
    entry["without_stopper"] = {"unit": u, "verdict": res.verdict, "value": res.value}
    if res.verdict != CONFIRMED_V:
        return
    g = r.graph
    ans = g.node("answer", entry["answer"], key=("a", id(q), way[:4], "stopped_by", entry["answer"]))
    taken = ask_obstacle(r, q, way, entry["route"], Obstacle("answers and stops the route", u, place, 0), params, base)
    fu = g.node("follow_up", "can it be taken", key=("f", id(q), "stopper", place), answer=taken["text"], verdict=taken["verdict"], target=u, asker=q.asker)
    g.link(ans, fu, "if_removed", because=f"read on the board without {unit_words(u)}", gone=unit_words(u), then=f"without {unit_words(u)} this is worth {res.value:g}")
    for what in ("driven off", "drawn away"):
        g.link(ans, g.node("follow_up", f"can it be {what}", key=("f", id(q), what, place), answer="not read", verdict=UNREAD, target=u, asker=q.asker), "if_removed", because="left open",
               gone=unit_words(u), then=f"without {unit_words(u)} this is worth {res.value:g}")
    entry["if_removed_value"] = res.value


def time_obstacle(r: Reading, q: Question, way: tuple, entry: dict) -> None:
    """A confirmed way that takes n moves, while a confirmed or waiting way of the OTHER side on a unit of at least the same stake, already held (the sheet's other questions, the pool), needs
    fewer: the obstacle `time`. Read only from what is held, nothing searched."""
    n = way[0]
    best = None
    for q2 in getattr(r, "questions", []):
        if q2.quiet or q2.target is None or q2.asker == q.asker or q2.stake < q.stake:
            continue
        deep = next((x for x in q2.findings if x.level >= 2 and x.detail.get("results")), None)
        for x in (deep.detail["results"] if deep else ()):
            w = x.get("way")
            if w is not None and w[0] >= 1 and x["status"] in (CONFIRMED_V, WAITING_V) and w[0] < n and (best is None or w[0] < best[0]):
                best = (w[0], unit_words(q2.target), "their" if q2.asker == OTHER else "my")
    if best is None and r.pool is not None:
        mine = r.absolute(q.asker)
        for fact in r.pool.ways():
            question, verdict = fact.payload["question"], fact.payload["verdict"]
            rt = verdict.get("route")
            if question[2] != mine and verdict["verdict"] in (CONFIRMED_V, WAITING_V) and rt is not None and 1 <= len(rt.places) < n:
                kind = r.env.codec.kind_at(r.position, question[0])
                if kind is not None and r.worth_of(kind) >= r.worth_of(q.target.kind):
                    best = (len(rt.places), f"the unit on {question[0]}", "their" if question[2] == r.absolute(OTHER) else "my")
    if best is None:
        return
    note = f"{best[2]} way on {best[1]} lands first (in {best[0]} moves)"
    entry["time"] = note
    g = r.graph
    ob = g.node("obstacle", f"time: {note}", key=("time", id(q), way[:4]))
    g.link(way_node(r, q, way), ob, "blocked_by", because=note, gone="that way", then=f"this one would be the first (in {n} moves)")


def apply_duties(r: Reading, q: Question, way: tuple, entry: dict, params: Params) -> None:
    """THE COST OF LEAVING for a confirmed way: the unit that travels may hold something where it stands. The route's value is the target's frame less the largest stake among the duties dropped;
    if that is not above 0 the way is WAITING (`waits_on`: something else holds the unit), with the follow-up question whether another unit can take over."""
    if not params.duties or entry["status"] != CONFIRMED_V or way[0] < 1:
        return
    base = r.position if q.asker == SELF else r.passed_position()
    if base is None:
        return
    routes = q.findings[1].detail.get("routes", {}).get(way[:4]) or []
    route = entry.get("route") or (routes[0] if routes else None)
    if route is None or not route.places:
        return
    u = way[1]
    cost = _duty_reading(r, q, way, route, base, params)
    entry["duties"] = cost["duties"]
    entry["duty_mark"] = cost["mark"]
    if not cost["duties"]:
        return
    g = r.graph
    wn = way_node(r, q, way)
    biggest = max(cost["duties"], key=lambda d: d["stake"])
    entry["held_value"] = entry["value"]
    if entry["value"] - biggest["stake"] > 0:
        wn.stake_low = wn.stake_high = entry["value"] - biggest["stake"]       # a confirmed frame less the duty dropped
        return
    wn.stake_low, wn.stake_high = 0.0, entry["value"]                         # it would be worth the frame if the duty were taken over
    entry["status"] = entry["verdict"] = WAITING_V
    entry["alts"] = list(entry.get("alts", [])) + [("duty", (place_text(d["unit"].at),), f"something else holds {unit_words(d['unit'])}") for d in cost["duties"]]
    entry["answer"] = f"{unit_words(u)} holds {unit_words(biggest['unit'])} (stake {biggest['stake']:g}, more than the {entry['value']:g} the route wins)"
    entry["breaks"] = []
    entry["held_back"] = True
    wn.verdict, wn.answer = WAITING_V, entry["answer"]
    for d in cost["duties"]:
        cn = g.node("condition", f"something else holds {unit_words(d['unit'])}", key=("a", id(q), way[:4], "waits_on", d["unit"]))
        g.link(wn, cn, "waits_on", because="it would hold if another unit took over the duty")
        fu = g.node("follow_up", "can another unit of mine take over the duty", key=("f", id(q), "takeover", d["unit"]), answer=takeover_words(r, q, u, d, params), verdict="cheap layers only", asker=q.asker)
        g.link(g.nodes[cn.id], fu, "if_removed", because="the duty relieved", gone=unit_words(u), then="it could travel")


def settle_way(r: Reading, q: Question, way: tuple, params: Params) -> dict:
    """`_settle_way` with the cost of leaving and the time obstacle applied, and the time it took (obstacle questions and route walks both) and the nodes the route walks used."""
    key = ("settled", q.target, q.asker, way[:4])
    if key not in r._frames:                                          # a deeper level asks again: the same position, the same answer, and the pool is asked once
        t0 = time.perf_counter()
        entry = _settle_way(r, q, way, params)
        apply_duties(r, q, way, entry, params)
        if entry["status"] == CONFIRMED_V:
            time_obstacle(r, q, way, entry)
        way_chance(r, q, way, entry, params)
        entry["seconds_total"] = time.perf_counter() - t0
        entry.setdefault("nodes", 0)
        r._frames[key] = entry
    return r._frames[key]


def _graph_edges(entry: dict) -> list:
    """(edge kind, node text, because) for a verdict: the owner's answer that refutes (stopped_by), what a waiting way waits on (waits_on) and the answers that would stop it (breaks)."""
    out = []
    if entry["status"] == REFUTED_V and entry.get("answer"):
        out.append(("stopped_by", entry["answer"], "the route cannot be completed against it"))
    if entry["status"] == WAITING_V:
        out += [("waits_on", c[2], "it would hold if this were so") for c in entry.get("alts", [])]
        out += [("breaks", x.words, "that answer stops it") for x in entry.get("breaks", []) if hasattr(x, "words")]
    return out


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


SOURCE_WORDS = {"route": "the whole route followed, every answer, real moves", "real": "real moves", "given": "the way's first move played, every answer, real moves", "static": "read from the board as it stands, lines behind not read"}


def level_frames(q: Question, r: Reading, params: Params) -> tuple:
    """Level 2 of a non-goal unit by the exchange frame: the asker's frame on the target as it stands, then each way that passed the three links, judged by `settle_way` (a clear way of one move
    after its first move; otherwise the obstacles asked about and the routes followed whole). The verdict of a way: CONFIRMED, REFUTED, WAITING (it waits on a condition, with what breaks it) or
    NOT SETTLED. The question is LIVE when a way is confirmed, IDLE (the waiting records attached) when a way waits or is not settled, closed only when every way is refuted."""
    t0 = len(r.frame_log)
    supply = q.findings[1].detail.get("supply", []) if len(q.findings) > 1 else []
    rank = q.findings[1].detail.get("rank", {}) if len(q.findings) > 1 else {}
    ways = sorted(supply, key=lambda w: (w[0], rank.get(w, 0), w[1].kind, w[1].at, w[2]))[:params.max_ways]
    stands = exchange_stands(r, q.target, q.asker, params)
    results = [{"way": None, "move": None, "status": "confirmed" if stands["value"] > 0 else "refuted", **stands}]
    for way in ways:
        if way[0] == 0 and way not in q.findings[1].detail.get("routes", {}):
            continue                                                  # a unit that strikes it now is the frame as it stands
        results.append(settle_way(r, q, way, params))
    conf = [x for x in results if x["status"] == CONFIRMED_V]
    wait = [x for x in results if x["status"] == WAITING_V]
    unset = [x for x in results if x["status"] == UNSETTLED_V]
    side = side_word(q.asker)
    detail = {"mode": "frame", "results": results, "confirmed": len(conf), "refuted": len(results) - len(conf) - len(wait) - len(unset), "waiting": len(wait), "open": len(unset),
              "lost_where_it_stands": stands["value"] > 0, "sources": sorted({x["source"] for x in results}), "frame_steps": len(r.frame_log) - t0, "waiting_ways": wait}
    q.waiting = wait
    if stands["value"] > 0:
        detail.update(cornering(r, q.target, q.asker, params))
    if conf:
        best = max(conf, key=lambda x: x["value"])
        detail.update(value=best["value"], line=tuple(best["line"]))
        where = "where it stands" if best["way"] is None else f"after {unit_words(best['way'][1])} comes to {place_text(best['way'][2])}"
        corner = ", and cornered" if detail.get("cornered") else ""
        return LIVE, f"the exchange {where} is worth {best['value']:g} to {side}: {' '.join(best['line']) or 'no taking'} ({SOURCE_WORDS[best['source']]}){corner}", detail
    if wait or unset:
        bits = [f"{x['way'][1].kind} to {place_text(x['way'][2])} waits: {x['answer']}" for x in wait[:2]] + [f"{x['way'][1].kind} to {place_text(x['way'][2])} not settled: {x['answer']}" for x in unset[:1]]
        return IDLE, f"{len(wait)} ways wait and {len(unset)} are not settled, none confirmed; " + "; ".join(bits), detail
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


def record_graph(q: Question, r: Reading, f: "Finding", params: Params) -> None:
    """Add what a step found to the graph: the question's starting node, its ways (needs) and their obstacles (blocked_by) at level 1, the goal unit's weak places, the verdict at level 2."""
    g = r.graph
    qn = question_node(r, q)
    d = f.detail
    qn.verdict = f.status
    qn.answer = f.text
    if f.level >= 2 and f.status == LIVE and f.value is not None:
        bound(qn, f.value, f.value)
    elif f.level >= 2 and f.status == CLOSED:
        bound(qn, 0.0, 0.0)
    if f.level == 1 and not q.quiet:
        for w in (d.get("supply") or [])[:params.max_ways]:
            if w[0] == 0 and w not in (d.get("routes") or {}):
                continue
            way_node(r, q, w)
            found = (d.get("routes") or {}).get(w[:4]) or []
            for ob in (found[0].obstacles if found else ())[:params.max_obstacles]:
                obstacle_node(r, q, w, ob)
        if d.get("too_far"):
            hops, u = d["far"][0]
            n = g.node("answer", f"{unit_words(u)} is {hops} hops away", key=("far", id(q)), verdict="too far", target=q.target, asker=q.asker)
            g.link(qn, n, "blocked_by", because=f"{unit_words(u)} is {hops} hops away, too far in time (reach is {params.reach_k})", gone=unit_words(u), then=f"it would be within {params.reach_k} hops")
        for wp in d.get("weak", ()):
            k = len(wp["candidates"])
            need = wp["need"]
            verdict = "can be brought" if k >= need else "cannot be brought"
            ans = (f"{k} unit{'s' if k != 1 else ''} of {side_word(q.asker)} can come within {params.reach_k} hops" + (f": {_names([c[1] for c in wp['candidates'][:need]])}" if k else "")
                   + "; not read by the reasoner (no unit stands on the place to exchange)")
            wn = g.node("weak_place", f"can {side_word(q.asker)} bring {need} more unit{'s' if need != 1 else ''} to strike {wp['place']} ({len(wp['guards'])} guard it, {len(wp['strikers'])} strike it now)",
                        key=("wp", id(q), wp["place"]), answer=ans, verdict=verdict, target=q.target, asker=q.asker)
            g.link(qn, wn, "needs", because=f"{wp['place']} is guarded by {len(wp['guards'])} and struck by {len(wp['strikers'])}, so {need} more must strike it")
            for hops, u, arrival in wp["candidates"][:need]:
                cn = g.node("candidate", f"can {unit_words(u)} come to {arrival} ({hops} hops) and strike {wp['place']}", key=("wc", id(q), wp["place"], u),
                            target=u, asker=q.asker, verdict="reaches")
                g.link(wn, cn, "needs", because=f"the attack needs {unit_words(u)} to arrive")
                found = geometric_routes(r, q.target, u, arrival, wp["place"], "strike", params)
                for ob in (found[0].obstacles if found else ())[:params.max_obstacles]:
                    on = g.node("obstacle", _obstacle_words(ob, q.asker), key=("wo", id(q), wp["place"], u, ob.place))
                    g.link(cn, on, "blocked_by", because=_obstacle_words(ob, q.asker), gone=unit_words(ob.unit) if ob.unit else "the obstacle", then=f"it would take {hops} hops")


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
    if not q.quiet:
        record_graph(q, r, f, params)
        assign_chances(q, r, params)
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
    moves: "MoveSheet | None" = None       # the question of each move, when `Params.move_question` is on

    @property
    def graph(self) -> Graph:
        return self.reading.graph


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


def stake_bounds(q: Question, r: "Reading | None") -> tuple:
    """(low, high) of the question's stake: its node's bounds in the graph as they now stand (tightened as answers came), else 0 to the stake it was given."""
    n = r.graph.nodes[q.node] if (r is not None and q.node is not None) else None
    high = q.stake if (n is None or n.stake_high is None) else n.stake_high
    low = 0.0 if n is None else min(n.stake_low, high)
    return low, high


def stake_at_risk(q: Question, r: "Reading | None", params: Params) -> float:
    """THE TOLERANCE FOR RISK. The stake a question's next step is weighed by. A unit of the side to move (mine) is at risk of being lost: the stake is `risk` times the high bound plus
    (1 - risk) times the midpoint of the bounds, so risk 1 is the worst case, 0 the expected value. A unit of the other side is a gain to be had, never a loss: the midpoint (the expected size
    of the stake; the chance that it is real enters once, as the other factor of the priority, not twice)."""
    low, high = stake_bounds(q, r)
    mid = 0.5 * (low + high)
    if q.owner == SELF:
        return params.risk * high + (1.0 - params.risk) * mid
    return mid


def chance_priority(q: Question, costs: _Costs, params: Params, table: "Chances | None", r: "Reading | None") -> float:
    """THE CLOCK BY CHANCE: chance that the next step finds something real x the stake at risk, over the expected cost of the step (the running mean of the measured costs of its level).
    The chance is the table's, for the question as its last step left it (its verdict kind, so a question that waits sinks and one lost where it stands rises); with no table it is 1 (the
    order is then stake at risk over cost). Urgency is not used: what it stood for is in the chance."""
    chance = 1.0 if table is None else table.chance(question_key(q), params.chance_horizon)
    return chance * stake_at_risk(q, r, params) / costs.expect(q.level)


def _record(sheet: Sheet, f: Finding, costs: _Costs, params: Params) -> None:
    q = f.question
    sheet.steps.append(Step(q, f.level, f.status, f.text, f.cost, f.detail.get("mode", "")))
    costs.add(f.level if not q.quiet else 0, f.cost)
    if f.status == LIVE and f.new:
        q.urgency *= params.urgency_up
    elif f.status != CLOSED:
        q.urgency *= params.urgency_down
        if f.status == IDLE and q.waiting and f.level >= 2:
            q.urgency *= params.waiting_urgency
    if q.state == "open" and not q.quiet and q.level > params.max_level:
        q.state, q.closed_at, q.reason = "closed", f.level, "asked to the deepest level"


def run(position: str, worth: dict, *, max_steps: "int | None" = None, budget_seconds: "float | None" = None, budget_cost: "float | None" = None, env: "Env | None" = None, game: "str | None" = None,
        params: Params = Params(), pool: "Pool | None" = None, on_step=None) -> Sheet:
    """The anytime schedule over the position's questions (see the module). Stops at `max_steps` steps, at `budget_seconds` of wall time (the reading of the position counted), or when the
    summed measured cost of the steps taken reaches `budget_cost` (seconds; the reproducible form of the budget); default 40 steps when none is given. A step is started only while
    the budget is not used up; one step already begun is finished, so the time may overrun by that step. `params.clock` is the order: "present" (every question's level 0 first, then stake x
    urgency over cost), "chance" (from the start by chance x stake at risk over cost, see `chance_priority`), "plain" (the first order, each question taken to its full depth before the next).
    `on_step(sheet, step, elapsed)` is called after every step."""
    if env is None:
        if game is None:
            raise ValueError("run needs an Env or a game name")
        env = Env(game)
    t0 = time.perf_counter()
    r = Reading(env, position, worth)
    r.params_k = params.reach_k
    r.pool = pool
    questions = first_order(r, params)
    sheet = Sheet(position, questions, read_seconds=r.read_seconds, reading=r)
    r.questions = sheet.questions
    if max_steps is None and budget_seconds is None and budget_cost is None:
        max_steps = 40
    costs = _Costs(params)
    table = load_chances(params.chances) if params.chances else None
    spent = [0.0]

    def over() -> bool:
        return ((max_steps is not None and len(sheet.steps) >= max_steps) or (budget_seconds is not None and time.perf_counter() - t0 >= budget_seconds)
                or (budget_cost is not None and spent[0] >= budget_cost))

    def take(q: Question) -> None:
        f = step(q, r, params)
        _record(sheet, f, costs, params)
        spent[0] += f.cost
        if on_step is not None:
            on_step(sheet, sheet.steps[-1], time.perf_counter() - t0)

    if params.clock == "present":
        for q in questions:                                  # every question's level 0 first, in the first order
            if over():
                break
            take(q)
    quiet: "Question | None" = None
    lowest = min((q.stake for q in questions), default=params.min_stake)
    while not over():
        if quiet is None and questions and all(q.level > 0 and (q.state == "closed" or q.status == IDLE) for q in questions):
            quiet = quiet_question(r, len(questions) + 1, params, lowest)
            if quiet is not None:
                sheet.questions.append(quiet)
        pool = [q for q in sheet.questions if q.state == "open"]
        if not pool:
            break
        if params.clock == "chance":
            q = max(pool, key=lambda x: (chance_priority(x, costs, params, table, r), -x.rank))
        elif params.clock == "plain":
            q = min(pool, key=lambda x: x.rank)
        else:
            q = max(pool, key=lambda x: (priority(x, costs), -x.rank))
        take(q)
    if params.move_question:
        left = None if budget_seconds is None else max(0.0, budget_seconds - (time.perf_counter() - t0))
        sheet.moves = move_sheet(env, r, params, budget_seconds=left)
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
        deep = next((x for x in q.findings if x.level >= 2 and x.detail.get("results")), None)
        for x in (deep.detail["results"] if deep else ()):
            if x.get("way") is None or not x.get("route"):
                continue
            line = f"       way: {route_words(x['route'], q.target)}: {x['status'].upper()}"
            if x["status"] == CONFIRMED_V:
                line += f", worth {x['value']:g} against every answer"
            elif x.get("answer"):
                line += f": {x['answer']}"
            if x["status"] == WAITING_V:
                line += "; waits on " + "; ".join(c[2] for c in x.get("alts", [])) + "; broken by " + ("; ".join(b.words for b in x.get("breaks", []) if hasattr(b, "words")) or "nothing seen")
            out.append(line)
    if sheet.moves is not None:
        out.append(f"the question of the move: {len(sheet.moves.results)} of {len(sheet.moves.order)} moves asked in {sheet.moves.seconds:.3f} s")
        for res in sheet.moves.results:
            if res["verdict"] != SAFE_V:
                out += explain_move(res)
    return out


def explain_chain(sheet: Sheet, question: Question) -> list:
    """One question's chain of claims from its starting node down: every line says "because ..." and, where an obstacle gives one, "if ... were gone then ..."."""
    g = sheet.graph
    if question.node is None:
        return [f"{question.label()}: nothing recorded yet"]
    return chain_lines(g, g.nodes[question.node])


# ---------------------------------------------------------------------------------------------------------------------------------------------- the question of the move

SAFE_V, LOSES_V, MOVE_UNSETTLED = "SAFE", "LOSES", "NOT SETTLED"


def child_reading(r: Reading, child: str, params: Params) -> Reading:
    """The reading of the position `child` reached from `r`'s position by one move, made once and shared (the goal's kind is the game's, not the position's)."""
    got = r._frames.get(("reading", child))
    if got is None:
        got = Reading(r.env, child, r.worth)
        got._goal_kind = r._goal_kind if r._goal_kind is not None else (r.goal_kind(),)
        r._frames[("reading", child)] = got
    got.params_k, got.passes = params.reach_k, params.passes
    return got


def _taken_by(r: Reading, child_r: Reading) -> list:
    """The units of the other side that the move took: the other side's units before the move that are not the child's units of the side to move (a multiset difference, so every kind of taking is
    found, the place of a taken unit need not be the place the move ended on)."""
    have: dict = {}
    for u in child_r.units:
        if u.side == SELF:
            have[(u.kind, u.at)] = have.get((u.kind, u.at), 0) + 1
    out = []
    for u in r.units:
        if u.side != OTHER:
            continue
        if have.get((u.kind, u.at), 0) > 0:
            have[(u.kind, u.at)] -= 1
        else:
            out.append(u)
    return out


def _threat(unit: Unit, at_before: str, part: str, kind: str, value: float, line: tuple, open_: bool, words: str) -> dict:
    return {"unit": unit, "place": at_before, "part": part, "kind": kind, "value": float(value), "line": tuple(line), "open": bool(open_), "words": words}


def move_question(env: Env, reading: Reading, move: str, params: Params = Params()) -> dict:
    """THE QUESTION OF THE MOVE (plan 78.42): the side to move plays `move`; is something of its own lost for it? Three answers, each a node of the graph with stake bounds, asked in this order:
      a. WHAT IT LEAVES: the cost of leaving (narrow form) for the moving unit and this step: which of the mover's units are newly lost where they stand, or the goal unit newly under the sign,
         because the unit is no longer where it was.
      b. WHERE IT LANDS: on the position after the move (real move through the kernel), the other side to move: is the moved unit lost where it now stands (the exchange frame as it stands, real
         moves)? When the move took something, the answer is the balance: what it took minus what the frame says is lost.
      c. WHAT THE OTHER SIDE THEN HAS: of the mover's other units, those on places whose count of units bearing on them (or whose content) changed and that the other side now strikes, not read by (a),
         and not lost before: the frame on each; and the goal unit's sign when it or its exits are on a changed place.
    The verdict: SAFE (nothing is lost), LOSES (the unit, and the amount as bounds: the worst single exchange less what the move took, up to the sum of the exchanges less what it took, with the line of real
    moves the frame found), or NOT SETTLED (a frame left open by its node budget, the goal unit newly under the sign without a loss, or the move or position refused by the kernel).
    -> {move, unit, dest, took, took_units, answers: {a, b, c}, threats, verdict, named, low, high, line, because, ms: {a, read, b, c, total}, nodes}"""
    t_all = time.perf_counter()
    r = reading
    r.params, r.params_k, r.passes = params, params.reach_k, params.passes
    origin, dest = env.codec.move_ends(move)
    u = next((x for x in r.units if x.side == SELF and place_text(x.at) == origin), None)
    if u is None:
        raise ValueError(f"{move} is not a move of a unit of the side to move")
    ms = {"a": 0.0, "read": 0.0, "b": 0.0, "c": 0.0, "total": 0.0}
    out = {"move": move, "unit": u, "dest": dest, "took": 0.0, "took_units": [], "answers": {"a": "", "b": "", "c": ""}, "threats": [], "verdict": SAFE_V, "named": None, "low": 0.0, "high": 0.0,
           "line": (), "because": "", "ms": ms, "nodes": {}, "unsettled": []}
    st = r.try_state(r.position, move)
    if st is None:
        out.update(verdict=MOVE_UNSETTLED, because="the kernel does not accept the move")
        ms["total"] = 1000 * (time.perf_counter() - t_all)
        return out
    if st[3] is not None:
        out.update(because="the move ends the game; nothing is left to lose", answers={"a": "not read: the game is over", "b": "not read: the game is over", "c": "not read: the game is over"})
        ms["total"] = 1000 * (time.perf_counter() - t_all)
        return out
    child = st[0]
    weight = r.weight(u)
    top = max([r.weight(x) for x in r.units if x.side == SELF] + [params.min_stake])
    g = r.graph
    root = g.node("move", f"is the move {move} of {unit_words(u)} safe for {side_word(SELF)}", key=("m", move), target=u, asker=SELF, stake_high=top)

    # the position after the move, read once for the answers that need it (and the goal sign of a)
    t0 = time.perf_counter()
    try:
        child_r = child_reading(r, child, params)
    except ValueError:
        out.update(verdict=MOVE_UNSETTLED, because="the position after the move cannot be read")
        ms["total"] = 1000 * (time.perf_counter() - t_all)
        return out
    mine_now = {place_text(x.at): x for x in child_r.units if x.side == OTHER}
    took_units = _taken_by(r, child_r)
    took = sum(r.worth_of(x.kind) for x in took_units)
    out["took"], out["took_units"] = took, took_units
    ms["read"] = 1000 * (time.perf_counter() - t0)

    # a. what it leaves
    t0 = time.perf_counter()
    q = Question(u, SELF, weight, 0)
    leave = cost_of_leaving(r, q, u, move, r.position, params)
    cands = _leaving_candidates(r, q, u, move, params)[0] if leave["read"] else set()
    threats, unsettled = [], []
    for d in leave["duties"]:
        v = d["unit"]
        if d["kind"] == "newly lost":
            threats.append(_threat(v, place_text(v.at), "a", "left unguarded", d["value"], (), False, f"{unit_words(v)} is lost where it stands once {unit_words(u)} has gone"))
        else:
            unsettled.append({"unit": v, "part": "a", "words": f"{unit_words(v)} comes newly under the sign once {unit_words(u)} has gone", "stake": d["stake"]})
    out["answers"]["a"] = ("; ".join(t["words"] for t in threats + [dict(words=x["words"]) for x in unsettled]) or f"leaving its place newly loses nothing ({len(cands)} unit{'s' if len(cands) != 1 else ''} it held were read)") if leave["read"] else "not read"
    ms["a"] = 1000 * (time.perf_counter() - t0)

    # b. where it lands
    t0 = time.perf_counter()
    land = mine_now.get(dest)
    if land is None:
        out["answers"]["b"] = "not read: the moved unit is not where the move ends"
    elif not child_r.weighty(land):
        out["answers"]["b"] = f"my {land.kind} on {dest} is worth the least of all, not asked" + (f"; it took {_names(took_units)} worth {took:g}" if took_units else "")
    else:
        ex = exchange_stands(child_r, land, SELF, params)
        mine = f"my {land.kind} on {dest}"
        where = f"the other side's exchange on {mine} is worth {ex['value']:g} to it" if ex["value"] > 0 else f"{mine} is not lost where it lands"
        if took_units:
            where += f"; the move took {_names(took_units)} worth {took:g}, so the balance is {took - ex['value']:+g}"
        out["answers"]["b"] = where + f" ({SOURCE_WORDS[ex['source']]})"
        if ex["value"] > 0:
            threats.append(_threat(u, origin, "b", "lands", ex["value"], ex["line"], ex["open"], where))
        elif ex["open"]:
            unsettled.append({"unit": u, "part": "b", "words": f"the exchange where it lands was not read to its end (node cap)", "stake": weight})
    ms["b"] = 1000 * (time.perf_counter() - t0)

    # c. what the other side then has
    t0 = time.perf_counter()
    changed = env.codec.changed_places(r.position, child)
    read_units, newly = [], []
    for v in r.units:
        if v.side != SELF or v == u or v in cands or not r.weighty(v) or v == r.goal_unit(SELF) or place_text(v.at) not in changed:
            continue
        vc = mine_now.get(place_text(v.at))
        if vc is None or not child_r.strikers_of(vc, SELF):
            continue
        read_units.append(v)
        if exchange_stands(r, v, OTHER, params)["value"] > 0:
            continue                                                   # lost before the move: not newly
        ex = exchange_stands(child_r, vc, SELF, params)
        if ex["value"] > 0:
            words = f"{unit_words(v)} is lost where it stands after the move: worth {ex['value']:g} to the other side"
            threats.append(_threat(v, place_text(v.at), "c", "newly struck", ex["value"], ex["line"], ex["open"], words))
            newly.append(words)
        elif ex["open"]:
            unsettled.append({"unit": v, "part": "c", "words": f"the exchange on {unit_words(v)} was not read to its end (node cap)", "stake": r.weight(v)})
    goal = r.goal_unit(SELF)
    if goal is not None and goal != u and not any(x["unit"] == goal for x in unsettled) and changed & ({place_text(goal.at)} | {place_text(x) for x in r.exits_of(goal)}):
        if _goal_newly_under_sign(r, q, child, params):
            unsettled.append({"unit": goal, "part": "c", "words": f"{unit_words(goal)} comes newly under the sign", "stake": r.weight(goal)})
    out["answers"]["c"] = ("; ".join(newly + [x["words"] for x in unsettled if x["part"] == "c"])
                           or f"{len(read_units)} unit{'s' if len(read_units) != 1 else ''} on changed places read; none newly lost ({len(changed)} places changed)")
    ms["c"] = 1000 * (time.perf_counter() - t0)

    # the verdict
    out["threats"], out["unsettled"] = threats, unsettled
    sure = sorted((t for t in threats if t["value"] - took > 0), key=lambda t: (-t["value"], t["part"]))
    if sure:
        worst = sure[0]
        lo = worst["value"] - took
        hi = max(lo, sum(t["value"] for t in threats) - took)
        out.update(verdict=LOSES_V, named=worst, low=lo, high=hi, line=worst["line"],
                   because=f"{worst['words']}" + (f"; the move took {_names(took_units)} worth {took:g}, which leaves {lo:g}" if took_units else ""))
    elif unsettled:
        first = unsettled[0]
        out.update(verdict=MOVE_UNSETTLED, named={"unit": first["unit"], "place": place_text(first["unit"].at), "part": first["part"], "words": first["words"]},
                   low=0.0, high=max(x["stake"] for x in unsettled), because=first["words"])
    else:
        out.update(because="nothing of its own is lost: " + "; ".join(out["answers"][k] for k in "abc"))
    # the nodes: one question, three answers, each with its bounds
    ans_nodes = {}
    for key, asks in (("a", f"what does {unit_words(u)} leave by going to {dest}"), ("b", f"is it lost where it lands on {dest}"), ("c", "what does the other side then have")):
        n = g.node("answer", asks, key=("m" + key, move), answer=out["answers"][key])
        g.link(root, n, "needs", because=out["answers"][key])
        part_threats = [t for t in threats if t["part"] == key]
        part_open = [x for x in unsettled if x["part"] == key]
        hi = sum(t["value"] for t in part_threats) + sum(x["stake"] for x in part_open)
        n.stake_high = hi
        bound(n, low=max([t["value"] for t in part_threats] + [0.0]), high=hi)
        ans_nodes[key] = n.id
    root.answer, root.verdict = out["because"], out["verdict"]
    bound(root, low=out["low"], high=out["high"] if out["verdict"] != SAFE_V else 0.0)
    out["nodes"] = {"move": root.id, **ans_nodes}
    ms["total"] = 1000 * (time.perf_counter() - t_all)
    return out


def explain_move(res: dict) -> list:
    """The verdict of one move in plain words, every line with its "because": the move, what it leaves, where it lands, what the other side then has, and the verdict with the unit, the bounds and the line."""
    u = res["unit"]
    taking = f", taking {_names(res['took_units'])} (worth {res['took']:g})" if res["took_units"] else ""
    out = [f"the move {res['move']}: {unit_words(u)} goes to {res['dest']}{taking}: {res['verdict']}"]
    out.append(f"  what it leaves -- because {res['answers']['a']}")
    out.append(f"  where it lands -- because {res['answers']['b']}")
    out.append(f"  what the other side then has -- because {res['answers']['c']}")
    if res["verdict"] == LOSES_V:
        n = res["named"]
        out.append(f"  it LOSES {unit_words(n['unit'])} -- because {res['because']}; worth to lose between {res['low']:g} and {res['high']:g}" + (f"; the line: {' '.join(res['line'])}" if res["line"] else ""))
    elif res["verdict"] == MOVE_UNSETTLED:
        out.append(f"  NOT SETTLED -- because {res['because']}" + (f"; at stake up to {res['high']:g}" if res["high"] else ""))
    else:
        out.append(f"  SAFE -- because {res['because']}")
    return out


@dataclass
class MoveSheet:
    """The verdicts of a position's moves: `results` in the order asked, `unread` the moves not reached (cut by the budget), `order` the whole order planned."""
    position: str
    results: list
    unread: list
    order: list
    seconds: float = 0.0

    def verdicts(self) -> dict:
        return {x["move"]: x["verdict"] for x in self.results}

    def loses(self) -> list:
        return [x for x in self.results if x["verdict"] == LOSES_V]


def move_order(reading: Reading, moves: list, params: Params = Params()) -> list:
    """The moves in the order they are asked: by the stake of the moving unit times whether its landing place is struck by the other side in the present reading (struck places first, the heavier
    unit first), then the rest, the heavier unit first; ties in the kernel's own order."""
    r = reading
    struck = {place_text(x) for x in r.struck_by(OTHER)}
    codec = r.env.codec
    by_place = {place_text(x.at): x for x in r.units if x.side == SELF}

    def key(item):
        i, m = item
        o, d = codec.move_ends(m)
        stake = r.weight(by_place[o]) if o in by_place else 0.0
        return (-(stake * (1 if d in struck else 0)), -stake, i)
    return [m for _, m in sorted(enumerate(moves), key=key)]


def move_sheet(env: Env, reading: Reading, params: Params = Params(), moves: "list | None" = None, budget_seconds: "float | None" = None, budget_cost: "float | None" = None,
               max_moves: "int | None" = None) -> MoveSheet:
    """The question of the move for each legal move of the position (or for the given list), in `move_order`, stopped when the wall-clock budget, the budget in summed measured cost (seconds)
    or the number of moves is used up; one move already begun is finished. The moves not reached are `unread`."""
    t0 = time.perf_counter()
    moves = list(reading.legal if moves is None else moves)
    order = move_order(reading, moves, params)
    results, spent = [], 0.0
    for n, m in enumerate(order):
        if ((budget_seconds is not None and time.perf_counter() - t0 >= budget_seconds) or (budget_cost is not None and spent >= budget_cost) or (max_moves is not None and n >= max_moves)):
            break
        res = move_question(env, reading, m, params)
        results.append(res)
        spent += res["ms"]["total"] / 1000.0
    return MoveSheet(reading.position, results, order[len(results):], order, time.perf_counter() - t0)


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
    routes: list = field(default_factory=list)  # measure H: one dict per unit above the lowest worth with ways of exactly two moves and none shorter
    reach: list = field(default_factory=list)   # (units reached within reach_k by the medium, by the move tables) per non-goal question at level 1
    too_far: int = 0                            # questions closed at level 1 because every unit of the asker is farther than reach_k hops
    graph: tuple = (0, 0)                       # (nodes, edges) of the position's graph
    chains: list = field(default_factory=list)  # printed chains (lines) of questions with an obstacle and its follow-up
    cf: list = field(default_factory=list)      # the counterfactual's log, one dict per blocked step; and the number cut by the cap in `cf_cut`
    cf_cut: int = 0
    joint: list = field(default_factory=list)   # the sets of units that must all go: one dict per set (members, asked, dead, moves more)
    leaving_audit: list = field(default_factory=list)   # the audited costs of leaving: narrow reading against the whole reread
    duty_rows: list = field(default_factory=list)   # one dict per way held back by a duty (or read for one): skipped, traveller moved in the game, held unit taken within 4 plies
    bounds: dict = field(default_factory=dict)  # node kind -> [nodes whose bound closed (low = high), nodes]
    way_costs: list = field(default_factory=list)   # (hops, ms, nodes, verdict) of every way of more than one move settled at level 2


def sample_chains(sheet: Sheet, limit: int = 1) -> list:
    """The printed chain of the first question whose graph holds an obstacle with its follow-up (for the report's samples)."""
    out = []
    for q in sheet.questions:
        if q.quiet or q.node is None:
            continue
        lines = chain_lines(sheet.graph, sheet.graph.nodes[q.node])
        if any("if_removed" in ln for ln in lines):
            out.append(lines)
            if len(out) >= limit:
                break
    return out


def node_bounds(graph: Graph) -> dict:
    """{node kind: [nodes whose bound closed (low equals high), nodes]}."""
    out: dict = {}
    for n in graph.nodes:
        c = out.setdefault(n.kind, [0, 0])
        c[1] += 1
        c[0] += n.stake_high is not None and n.stake_low == n.stake_high
    return out


def format_counterfactual(records: list) -> list:
    """OBS lines: what the counterfactual found at the blocked steps, the cost of leaving, and the bounds that closed."""
    out = []
    cf = [x for r in records for x in r.cf]
    if cf:
        one = sum(1 for x in cf if x["found"] == 1)
        many = sum(1 for x in cf if x["found"] > 1 or x.get("joint_state") == "joint")
        none = sum(1 for x in cf if x["found"] == 0 and x.get("joint_state") != "joint")
        words: dict = {}
        for x in cf:
            for w in x["words"]:
                words[w] = words.get(w, 0) + 1
        ms = [x["ms"] for x in cf]
        calls = [x["calls"] for x in cf]
        out.append(f"OBS (H) counterfactual: blocked steps examined {len(cf)} (in {sum(1 for r in records if r.cf)} of {len(records)} positions); explained by one unit {one} ({_pct(one, len(cf))}), by several {many} ({_pct(many, len(cf))}), "
                   f"by none (no single unit) {none} ({_pct(none, len(cf))}); not examined because of the cap {sum(r.cf_cut for r in records)}; removals rejected as not a position {sum(x['rejected'] for x in cf)}; of those, the unit moved aside instead and the step then made {sum(x.get('aside', 0) for x in cf)}")
        former = [x for x in cf if x["found"] == 0]
        sized = {}
        for x in former:
            if x.get("joint_state") == "joint":
                sized[x["joint"]] = sized.get(x["joint"], 0) + 1
        sets = [x for r in records for x in r.joint]
        out.append(f"OBS (H) steps no single removal explains (the former \"none\"): {len(former)}; explained by removing the set of units the first layer names together {sum(sized.values())} ({_pct(sum(sized.values()), len(former))}"
                   + (", by size " + ", ".join(f"{k} units {v}" for k, v in sorted(sized.items())) if sized else "") + f"); more than the cap of units in the way {sum(1 for x in former if x.get('joint_state') == 'cap')}; "
                   f"a set named but the step still not possible {sum(1 for x in former if x.get('joint_state') == 'failed')}; fewer than two units named (still none) {sum(1 for x in former if not x.get('joint_state'))}")
        out.append(f"OBS (H) sets of units that must all go: {len(sets)}; follow-ups asked {sum(x['asked'] for x in sets)} of {sum(x['members'] for x in sets)} members; sets killed by a dead member {sum(1 for x in sets if x['dead'])}; "
                   f"sets all asked and none dead {sum(1 for x in sets if not x['dead'] and x['asked'] == x['members'])}, moves more (summed over those) {sum(x['moves_more'] for x in sets if not x['dead'] and x['asked'] == x['members'])}")
        out.append("OBS (H) counterfactual, describing words of the units found: " + ", ".join(f"{w} {n}" for w, n in sorted(words.items(), key=lambda kv: -kv[1])))
        out.append(f"OBS (H) counterfactual cost per blocked step: mean {sum(ms) / len(ms):.2f} ms, max {max(ms):.2f} ms; kernel calls mean {sum(calls) / len(calls):.1f}, max {max(calls)}")
    audit = [x for r in records for x in r.leaving_audit]
    if audit:
        miss = [x for x in audit if x["missed"]]
        out.append(f"OBS (H) leaving audit: narrow reading against the whole reread on {len(audit)} costs of leaving; duties differ in {sum(1 for x in audit if x['differ'])} ({_pct(sum(1 for x in audit if x['differ']), len(audit))}); "
                   f"the whole reread found {sum(x['full'] for x in audit)} duties, the narrow {sum(x['narrow'] for x in audit)}; narrow missed {sum(len(x['missed']) for x in audit)}"
                   + ("".join(f"; missed by {x['unit']} {x['mv']}: {', '.join(x['missed'])} (position {x['position']})" for x in miss[:5])) + f"; narrow found what the reread did not {sum(len(x['extra']) for x in audit)}")
    rows = [d for r in records for d in r.duty_rows]
    if rows:
        held = [d for d in rows if d["held_back"]]
        known = [d for d in held if not d["skipped"]]
        moved = [d for d in held if d["traveller_moved"] and d["taken4"] is not None]
        hit = sum(1 for d in moved if d["taken4"])
        out.append(f"OBS (H) duties: ways with a duty found {len(rows)}; held back {len(held)} (route read and confirmed, then held back: {len(known)}; route not read because leaving drops more than the attack could win: {len(held) - len(known)}); "
                   f"of the held-back ways, those where the travelling unit did move at that ply of the game and the held unit can be followed: {len(moved)}, the held unit taken within 4 plies in {hit} ({_pct(hit, len(moved))})"
                   + ("; the count is too small to mean anything" if len(moved) < 20 else ""))
    kinds: dict = {}
    for r in records:
        for k, (c, n) in r.bounds.items():
            kk = kinds.setdefault(k, [0, 0])
            kk[0] += c
            kk[1] += n
    if kinds:
        out.append("OBS (H) bounds closed (stake low equals high) after the levels run, by node kind: " + ", ".join(f"{k} {c}/{n} ({_pct(c, n)})" for k, (c, n) in sorted(kinds.items())) + f"; all nodes {sum(c for c, _ in kinds.values())}/{sum(n for _, n in kinds.values())}")
    return out


def route_unit(q: Question, sheet: Sheet, params: Params, taken_of) -> "dict | None":
    """Measure H: for a unit above the lowest worth with at least one way of exactly two moves and none shorter, its verdict by the routes (confirmed, waiting, not settled, refuted), its
    verdict under the earlier form (every way judged after its first move only), the group of its best-ordered way (clear route; an obstacle of the asker's own; an obstacle of the other side),
    whether the exact unit was taken within 4 and 8 plies, and what the ways cost."""
    if len(q.findings) < 3 or q.findings[1].status != LIVE:
        return None
    det1 = q.findings[1].detail
    ways = list(det1.get("supply", []))
    two = [w for w in ways if w[0] == 2]
    if not two or any(w[0] < 2 for w in ways):
        return None
    deep = next((x for x in q.findings if x.level >= 2 and x.detail.get("results")), None)
    if deep is None:
        return None
    r = sheet.reading
    got = {x["way"][:4]: x for x in deep.detail["results"] if x.get("way") is not None}
    read = [w for w in two if w[:4] in got]
    if not read:
        return None
    rank = det1.get("rank", {})
    first = sorted(read, key=lambda w: (rank.get(w, 0), w[1].kind, w[1].at, w[2]))[0]
    obs = got[first[:4]].get("first_obstacles")
    if obs is None:
        best = (det1.get("routes", {}).get(first[:4]) or [None])[0]
        obs = [(o.unit.side if o.unit else None, o.kind, o.dead) for o in (best.obstacles if best else ())]
    group = "clear" if not obs else ("other" if any((side is not None and side != q.asker) or dead for side, _k, dead in obs) else "own")
    order = (CONFIRMED_V, WAITING_V, UNSETTLED_V, REFUTED_V)
    statuses = [got[w[:4]]["status"] for w in read]
    new = next(v for v in order if v in statuses)
    old_ms, old_conf = [], False
    for w in read:
        t0 = time.perf_counter()
        v = exchange_after(r, q.target, q.asker, w, params)
        old_ms.append(1000 * (time.perf_counter() - t0))
        old_conf = old_conf or v["value"] > 0
    place = place_text(q.target.at)
    return {"owner": q.owner, "new": new, "old": CONFIRMED_V if old_conf else REFUTED_V, "group": group, "taken4": taken_of(place, 4) if taken_of else None,
            "taken8": taken_of(place, 8) if taken_of else None, "ways": len(read), "new_ms": [1000 * got[w[:4]].get("seconds_total", 0.0) for w in read], "new_nodes": [got[w[:4]].get("nodes", 0) for w in read],
            "old_ms": old_ms, "ways_by": [got[w[:4]]["status"] for w in read], "pool": [got[w[:4]].get("pool") for w in read]}


def summarize_sheet(key: tuple, bad: bool, fell: "bool | None", sheet: Sheet, moved: "str | None" = None, taken_of=None, weight: float = 0.0, follow_all: bool = False,
                    params: Params = Params()) -> Record:
    ranks, ways, answered, units, lost_units = [], [0, 0, 0], 0, [], []
    routes, reach, too_far, way_costs, duty_rows = [], [], 0, [], []
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
            if result == "not settled" and l2.detail.get("waiting"):
                result = "waiting"
            units.append((q.owner, group, result, taken_of(place) if taken_of else None))
            ex = exchange_stands(sheet.reading, q.target, q.asker)
            lost_units.append((q.owner, ex["value"] > 0, taken_of(place) if taken_of else None))
        if follow_all and not q.goal and 1 in f:
            rec_geo, rec_old = f[1].detail.get("reach_geo"), f[1].detail.get("reach_old")
            if rec_geo is not None and rec_old is not None:
                reach.append((rec_geo, rec_old))
            too_far += bool(f[1].detail.get("too_far"))
        if follow_all and not q.goal:
            deep_f = next((x for x in q.findings if x.level >= 2 and x.detail.get("results")), None)
            for x in (deep_f.detail["results"] if deep_f else ()):
                if x.get("way") is not None and x.get("duties"):
                    held = x["duties"][0]["unit"]
                    mover_here = q.asker == SELF and moved is not None and place_text(x["way"][1].at) == moved
                    duty_rows.append({"held_back": bool(x.get("held_back")), "skipped": bool(x.get("skipped")), "traveller_moved": mover_here, "had_value": x.get("held_value"),
                                      "taken4": taken_of(place_text(held.at), 4) if (taken_of and mover_here) else None})
                if x.get("way") is not None and x["way"][0] >= 2 and x.get("source") == "route":
                    way_costs.append((x["way"][0], 1000 * x.get("seconds_total", 0.0), x.get("nodes", 0), x["status"]))
        if follow_all and not q.goal and ranks[-1].weighty:
            ru = route_unit(q, sheet, params, taken_of)
            if ru is not None:
                routes.append(ru)
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
                  frames=list(sheet.reading.frame_log) if sheet.reading else [], passes=list(sheet.reading.pass_log) if sheet.reading else [], routes=routes, reach=reach, too_far=too_far, way_costs=way_costs, duty_rows=duty_rows, joint=list(sheet.reading.joint_log) if sheet.reading else [], cf=list(sheet.reading.cf_log) if sheet.reading else [], cf_cut=sheet.reading.cf_cut if sheet.reading else 0, leaving_audit=list(sheet.reading.leaving_audit) if sheet.reading else [],
                  bounds=node_bounds(sheet.graph) if sheet.reading else {},
                  graph=sheet.graph.size() if sheet.reading else (0, 0), chains=sample_chains(sheet) if follow_all else [])


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


def measure(items: list, runner, samples: int = 3, codec=None, objector=None, alternatives=None, alt_limit: int = 100, follow_all: bool = False, params: Params = Params()) -> tuple:
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
        taken_of = (lambda place, k=4, it=it: codec.taken_within(start, it.full, it.key[1], place, k)) if codec else None
        rec = summarize_sheet(it.key, it.bad, it.fell, sheet, it.moved, taken_of, it.weight, follow_all, params)
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


def format_routes(records: list, params: Params) -> list:
    """OBS lines of measure H (and the geometry's counts): the units above the lowest worth with ways of exactly two moves and none shorter, by verdict, with the rate at which that exact unit
    is taken within 4 and 8 plies, beside the same groups under the earlier form; split by the best-ordered route's group; the cost; the medium against the move tables; the graph."""
    out = []
    rows = [(r.key, u) for r in records for u in r.routes]
    n = len(rows)
    sides = {SELF: "mine", OTHER: "theirs"}

    def cell(sel, k):
        read = [u for _k, u in sel if u[f"taken{k}"] is not None]
        hit = sum(1 for u in read if u[f"taken{k}"])
        return f"{hit} of {len(read)} ({_pct(hit, len(read))})"

    def lines(label, sel, field_name, verdicts):
        for v in verdicts:
            grp = [(k, u) for k, u in sel if u[field_name] == v]
            out.append(f"OBS (H) {label}, {v}: {len(grp)} units ({_pct(len(grp), len(sel))}); taken within 4 plies {cell(grp, 4)}; within 8 plies {cell(grp, 8)}")
    out.append(f"OBS (H) units read {n} (mine {sum(1 for _k, u in rows if u['owner'] == SELF)}, theirs {sum(1 for _k, u in rows if u['owner'] == OTHER)}): units above the lowest worth with a way of exactly two moves and none shorter, "
               f"level 2 read; reach_k {params.reach_k}, geometry {params.geometry}, routes {params.routes}, max_routes {params.max_routes}, route_nodes {params.route_nodes}")
    for side, word in sides.items():
        sel = [(k, u) for k, u in rows if u["owner"] == side]
        lines(f"{word}, new (the routes followed whole)", sel, "new", (CONFIRMED_V, REFUTED_V, WAITING_V, UNSETTLED_V))
        lines(f"{word}, present code (each way judged after its first move)", sel, "old", (CONFIRMED_V, REFUTED_V))
    for grp_name, text in (("clear", "a clear route"), ("own", "a route with an obstacle of the asker's own"), ("other", "a route with an obstacle of the other side")):
        for side, word in sides.items():
            sel = [(k, u) for k, u in rows if u["owner"] == side and u["group"] == grp_name]
            out.append(f"OBS (H) {word}, best-ordered route is {text}: {len(sel)} units")
            if not sel:
                continue
            lines(f"{word}, {text}, new", sel, "new", (CONFIRMED_V, REFUTED_V, WAITING_V, UNSETTLED_V))
            lines(f"{word}, {text}, present code", sel, "old", (CONFIRMED_V, REFUTED_V))
    new_ms = [x for _k, u in rows for x in u["new_ms"]]
    nodes = [x for _k, u in rows for x in u["new_nodes"]]
    old_ms = [x for _k, u in rows for x in u["old_ms"]]
    if new_ms:
        out.append(f"OBS (H) cost per two-move way, new: {len(new_ms)} ways, mean {sum(new_ms) / len(new_ms):.2f} ms, max {max(new_ms):.2f} ms; nodes mean {sum(nodes) / len(nodes):.1f}, max {max(nodes)}; "
                   f"present code: mean {sum(old_ms) / len(old_ms):.2f} ms, max {max(old_ms):.2f} ms")
        by = {}
        for _k, u in rows:
            for v in u["ways_by"]:
                by[v] = by.get(v, 0) + 1
        out.append("OBS (H) two-move ways by verdict (new): " + ", ".join(f"{v} {c}" for v, c in sorted(by.items())))
        pools = [x for _k, u in rows for x in u["pool"] if x]
        if pools:
            out.append(f"OBS (H) two-move ways through the pool: " + ", ".join(f"{v} {pools.count(v)}" for v in sorted(set(pools))))
    costs = [x for r in records for x in r.way_costs]
    for hops in sorted({c[0] for c in costs}):
        sel = [c for c in costs if c[0] == hops]
        by = {}
        for c in sel:
            by[c[3]] = by.get(c[3], 0) + 1
        out.append(f"OBS (H) all ways of {hops} moves settled by the routes (every unit, not only the group above): {len(sel)} ways, mean {sum(c[1] for c in sel) / len(sel):.2f} ms, max {max(c[1] for c in sel):.2f} ms; "
                   f"nodes mean {sum(c[2] for c in sel) / len(sel):.1f}, max {max(c[2] for c in sel)}; " + ", ".join(f"{v} {n}" for v, n in sorted(by.items())))
    pairs = [x for r in records for x in r.reach]
    if pairs:
        geo_only = sum(len(g - o) for g, o in pairs)
        old_only = sum(len(o - g) for g, o in pairs)
        both = sum(len(g & o) for g, o in pairs)
        differ = sum(1 for g, o in pairs if g != o)
        out.append(f"OBS (H) reach within {params.reach_k}, the medium against the move tables: {len(pairs)} questions read; units reached by both {both}, by the medium only {geo_only} (what the frozen board hid), "
                   f"by the move tables only {old_only}; questions where the two differ {differ} ({_pct(differ, len(pairs))})")
    out.append(f"OBS (H) questions closed at level 1 as too far in time (every unit of the asker farther than {params.reach_k} hops): {sum(r.too_far for r in records)} of {len(pairs)} non-goal questions read")
    sizes = [r.graph for r in records if r.graph != (0, 0)]
    if sizes:
        nn, ee = [a for a, _b in sizes], [b for _a, b in sizes]
        out.append(f"OBS (H) graph per position: nodes mean {sum(nn) / len(nn):.1f}, max {max(nn)}; edges mean {sum(ee) / len(ee):.1f}, max {max(ee)} ({len(sizes)} positions)")
    chains = [c for r in records for c in r.chains]
    for i, c in enumerate(chains[::max(1, len(chains) // 3)][:3], start=1):
        out.append(f"OBS (H) printed chain {i}:")
        out += [f"OBS   {ln}" for ln in c[:24]]
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


# ---------------------------------------------------------------------------------------------------------------------------------------------- measure I: anticipation over whole games

BUCKETS = (("never", 0, 0), ("1-2", 1, 2), ("3-4", 3, 4), ("5-8", 5, 8), ("more", 9, 10 ** 9))


def read_position(env: Env, worth: dict, params: Params, position: str, pool: "Pool | None" = None, max_steps: "int | None" = None) -> Sheet:
    """One position read as the measures read it: the schedule, level 1 for every question, level 2 for every non-goal unit open after level 1; the pool (if any) carries the ways."""
    sheet = run(position, worth, max_steps=max_steps, env=env, params=params, pool=pool)
    finish_levels(sheet, 1, params)
    settle_flagged(sheet, params)
    return sheet


def _bucket(plies: int) -> str:
    return next(name for name, lo, hi in BUCKETS if lo <= plies <= hi)


def anticipation(directory: str, games: int, env: Env, worth: dict, params: Params, audit_every: int = 5, max_plies: "int | None" = None) -> list:
    """Measure I. Every position of the first `games` games, the pool carried from ply to ply. For every taking of a unit above the lowest worth (not the goal unit): how many plies before it
    (counted to the taking move: the position just before the taking is 1) was the unit first under a confirmed or waiting verdict of one of its ways that stayed on until the taking, read
    as mine (its owner to move: the positions an even number of plies back from the one before the taking) and as theirs (the other positions), and in either reading. Then the false alarms
    (records of the pool that ended with the unit not taken within 8 plies) and the pool's saving."""
    codec = env.codec
    dist = {kind: {b[0]: 0 for b in BUCKETS} for kind in ("mine", "theirs", "either", "mine+stands", "theirs+stands", "either+stands")}
    takings = unread = 0
    born_all, stats = [], {"kept": 0, "revalidated": 0, "reread": 0, "new": 0, "flagged": 0, "queued": 0, "audited": 0, "audit_differed": 0, "facts": 0, "stale_now": 0}
    store_sizes: list = []
    sizes, seconds, positions = [], 0.0, 0
    for line in (Path(directory) / "games.jsonl").read_text().splitlines()[:games]:
        if not line.strip():
            continue
        g = json.loads(line)
        v, start, moves = g.get("variant") or env.variant, g.get("start"), g["moves"]
        by_ply, taken = codec.tracks(start, moves)
        pool = Pool(codec)
        pool.audit_every = audit_every
        per_ply: list = []
        for ply in range(len(moves) if max_plies is None else min(len(moves), max_plies)):
            pos = CE.state(v, start, moves[:ply])[0]
            pool.advance(pos)
            t0 = time.perf_counter()
            try:
                sheet = read_position(env, worth, params, pos, pool)
            except ValueError:
                per_ply.append(None)
                continue
            seconds += time.perf_counter() - t0
            pool.file_graph(sheet)
            positions += 1
            sizes.append(sheet.graph.size())
            rec = {}
            for q in sheet.questions:
                if q.quiet or q.goal or not sheet.reading.weighty(q.target):
                    continue
                uid = by_ply[ply].get(place_text(q.target.at), (None,))[0]
                deep = next((x for x in q.findings if x.level >= 2 and x.detail.get("results")), None)
                res = deep.detail["results"] if deep else []
                ways_on = any(x.get("way") is not None and x["status"] in (CONFIRMED_V, WAITING_V) for x in res)
                stands_on = bool(res) and res[0]["status"] == CONFIRMED_V
                rec[uid] = (q.owner == SELF, ways_on, ways_on or stands_on)
            per_ply.append(rec)
        for k, vv in pool.stats().items():
            stats[k] += vv
        store_sizes.append((len(pool.store.facts), sum(1 for f in pool.store.facts.values() if not f.superseded), len(pool.ways())))
        born_all.append((by_ply, taken, list(pool.born_log), len(per_ply)))
        for p, (uid, _side, _kind) in taken.items():
            if p >= len(per_ply):
                continue
            if per_ply[p] is None or uid not in per_ply[p]:
                unread += 1                                            # the unit was not weighty, or the position was not readable
                continue
            takings += 1
            for kind, idx in (("", 1), ("+stands", 2)):
                runs = {"mine": [], "theirs": [], "either": []}
                for name, start_q, step_q in (("theirs", p, 2), ("mine", p - 1, 2), ("either", p, 1)):
                    q_at = start_q
                    first = None
                    while q_at >= 0 and q_at < len(per_ply) and per_ply[q_at] is not None and uid in per_ply[q_at] and per_ply[q_at][uid][idx]:
                        first = q_at
                        q_at -= step_q
                    runs[name] = 0 if first is None else p - first + 1
                for name in runs:
                    dist[name + kind][_bucket(runs[name])] += 1
    out = [f"OBS (I) games {games}, positions read {positions} in {seconds:.1f} s; takings of a unit above the lowest worth, not the goal unit: {takings} (plus {unread} of a unit not weighty or a position not readable)"]
    for kind in dist:
        out.append(f"OBS (I) plies before the taking that the unit was first under a confirmed or waiting verdict that stayed on until the taking, read as {kind}: "
                   + ", ".join(f"{name} {dist[kind][name]} ({_pct(dist[kind][name], takings)})" for name, _lo, _hi in BUCKETS))
    for verdict in (CONFIRMED_V, WAITING_V):
        n = hit = 0
        for by_ply, taken, born, _n in born_all:
            for ply, key, vd, _value in born:
                if vd != verdict or ply >= len(by_ply):
                    continue
                uid = by_ply[ply].get(key[0], (None,))[0]
                n += 1
                hit += any(i in taken and taken[i][0] == uid for i in range(ply, ply + 8))
        out.append(f"OBS (I) false alarms, records filed as {verdict}: {n}; the unit taken within 8 plies of the record's reading in {hit} ({_pct(hit, n)}); not taken (false alarm) {n - hit} ({_pct(n - hit, n)})")
    total = stats["kept"] + stats["revalidated"] + stats["reread"] + stats["new"]
    out.append(f"OBS (I) memory (the fact store): ways asked for {total}: served as filed, not flagged by any move {stats['kept']} ({_pct(stats['kept'], total)}); flagged stale but its read cells unchanged, served after the comparison "
               f"{stats['revalidated']} ({_pct(stats['revalidated'], total)}); read again (flagged, cells changed) {stats['reread']} ({_pct(stats['reread'], total)}); new {stats['new']} ({_pct(stats['new'], total)}); "
               f"facts flagged by moves {stats['flagged']}; offered by the queue in chain-stake order (counted at each move) {stats['queued']}; read again from the queue by a budgeted refresh 0 (not run: the live questions of the next position ask for what they need)")
    out.append(f"OBS (I) memory audit: every 5th served fact read afresh: {stats['audited']} compared, the verdict differed in {stats['audit_differed']} ({_pct(stats['audit_differed'], stats['audited'])})")
    out.append("OBS (I) store size per game (facts filed in all, not superseded at the end, way facts live): " + "; ".join(f"{a} / {b} / {c}" for a, b, c in store_sizes) + f"; stale at the ends {stats['stale_now']}")
    if sizes:
        out.append(f"OBS (I) graph per position: nodes mean {sum(a for a, _b in sizes) / len(sizes):.1f}, max {max(a for a, _b in sizes)}; edges mean {sum(b for _a, b in sizes) / len(sizes):.1f}, max {max(b for _a, b in sizes)}")
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------- B: the chance of a node, fitted at sleep from our own games, read at wake

CHANCE_FORMAT = "question-chances-1"
KEY_NAMES = ("owner", "verdict", "moves", "route")
HORIZONS = (4, 8)                      # plies within which the exact unit is taken; both are counted, the table is read at `Params.chance_horizon`
DANGER = ("live0", "lost", "confirmed")    # the verdict kinds that say danger: live at level 0, lost where it stands, confirmed after an arrival
SMOOTH = 2.0                           # pseudo-observations a cell borrows from its next coarser key


def verdict_kind(q: Question, upto: "int | None" = None) -> str:
    """The kind of verdict the question has reached by its deepest finding up to level `upto` (None: all). Level 2 and on: `lost` (the exchange where it stands already wins), `confirmed`
    (a way after an arrival wins), `waiting`, `unsettled`, `refuted`. Before that: `live0` (live at level 0, kept as the danger sign until a deeper level speaks), `closed1`..`closed3` (closed
    at level 1 by which link failed), `live1`, `idle1`, `idle0`, and `unread` when nothing has been asked."""
    fs = [f for f in q.findings if upto is None or f.level <= upto]
    if not fs:
        return "unread"
    deep = next((f for f in reversed(fs) if f.level >= 2), None)
    if deep is not None:
        res = deep.detail.get("results")
        if not res:
            return {LIVE: "confirmed", CLOSED: "refuted"}.get(deep.status, "waiting")
        if res[0].get("way") is None and res[0]["status"] == CONFIRMED_V:
            return "lost"
        states = [x["status"] for x in res if x.get("way") is not None]
        for kind, word in ((CONFIRMED_V, "confirmed"), (WAITING_V, "waiting"), (UNSETTLED_V, "unsettled")):
            if kind in states:
                return word
        return "refuted"
    lv0 = next((f for f in fs if f.level == 0), None)
    lv1 = next((f for f in fs if f.level == 1), None)
    if lv0 is not None and lv0.status == LIVE:
        return "live0"
    if lv1 is not None:
        if lv1.status == CLOSED:
            return f"closed{lv1.detail.get('failed_at', 0)}"
        return "live1" if lv1.status == LIVE else "idle1"
    return "idle0"


def _moves_word(q: Question, kind: str, upto: "int | None") -> str:
    """The number of moves the way needs: 0 (lost where it stands), 1, 2, 3+ (the fewest among the ways that decided the verdict, else among those level 1 listed), `-` when no way is listed."""
    if kind == "lost":
        return "0"
    fs = [f for f in q.findings if upto is None or f.level <= upto]
    deep = next((f for f in reversed(fs) if f.level >= 2 and f.detail.get("results")), None)
    ways = [x["way"] for x in deep.detail["results"] if x.get("way") is not None and x["status"] == kind] if (deep is not None and kind in (CONFIRMED_V, WAITING_V)) else []
    if not ways:
        lv1 = next((f for f in fs if f.level == 1), None)
        ways = list(lv1.detail.get("supply") or []) if lv1 is not None else []
    if not ways:
        return "-"
    n = min(w[0] for w in ways)
    return str(n) if n < 3 else "3+"


def _route_word(q: Question, upto: "int | None") -> str:
    """`clear` (the best-ordered way's route has nothing in the way), `own` (an obstacle of the asker's own side only) or `theirs` (an obstacle of the other side, or one nothing can clear);
    `-` before level 1 or with no way listed. The same grouping as measure H."""
    lv1 = next((f for f in q.findings if f.level == 1 and (upto is None or f.level <= upto)), None)
    supply = list(lv1.detail.get("supply") or []) if lv1 is not None else []
    if not supply:
        return "-"
    rank = lv1.detail.get("rank", {})
    first = sorted(supply, key=lambda w: (w[0], rank.get(w, 0), w[1].kind, w[1].at, w[2]))[0]
    routes = (lv1.detail.get("routes") or {}).get(first[:4]) or []
    obs = routes[0].obstacles if routes else ()
    if not obs:
        return "clear"
    return "theirs" if any((o.unit is not None and o.unit.side != q.asker) or o.dead for o in obs) else "own"


def question_key(q: Question, upto: "int | None" = None) -> tuple:
    """The KEY of a question's record: (whose unit: mine = the side to move / theirs, the verdict kind, the moves the way needs, the route's group). Small on purpose: about 250 cells at most.
    The coarser keys are its prefixes, in this order: (owner, verdict, moves), (owner, verdict), (owner), ()."""
    kind = verdict_kind(q, upto)
    return ("mine" if q.owner == SELF else "theirs", kind, _moves_word(q, kind, upto), _route_word(q, upto))


class Chances:
    """A table of counts key -> {horizon: [taken, total]} and the chance read from it. CHANCE = (taken + a) / (total + a + b), with a = SMOOTH x p and b = SMOOTH x (1 - p), where p is the chance of
    the next coarser key (its prefix) computed the same way; at the root p is (taken + 1) / (total + 2). A key never seen has total 0 and so the chance of its coarser key."""

    def __init__(self, cells: dict, header: "dict | None" = None, smooth: float = SMOOTH):
        self.cells = {tuple(k): {int(h): list(v) for h, v in c.items()} for k, c in cells.items()}
        self.header, self.smooth = header or {}, smooth
        self.agg: dict = {}
        for key, c in self.cells.items():
            for n in range(len(key) + 1):
                slot = self.agg.setdefault(key[:n], {})
                for h, (taken, total) in c.items():
                    t = slot.setdefault(h, [0, 0])
                    t[0] += taken
                    t[1] += total
        self._memo: dict = {}

    def counts(self, key: tuple, horizon: int = 4) -> tuple:
        taken, total = self.agg.get(tuple(key), {}).get(horizon, (0, 0))
        return taken, total

    def chance(self, key: tuple, horizon: int = 4) -> float:
        key = tuple(key)[:len(KEY_NAMES)]
        memo = (key, horizon)
        if memo not in self._memo:
            p = 0.0
            for n in range(len(key) + 1):
                taken, total = self.counts(key[:n], horizon)
                p = (taken + 1.0) / (total + 2.0) if n == 0 else (taken + self.smooth * p) / (total + self.smooth)
            self._memo[memo] = p
        return self._memo[memo]


_TABLES: dict = {}


def load_chances(path: str) -> Chances:
    """The table written by `save_chances`, read once per path (a wake loads it; nothing in it came from a teacher)."""
    if path not in _TABLES:
        blob = json.loads(Path(path).read_text())
        if blob.get("format") != CHANCE_FORMAT:
            raise ValueError(f"{path} is not a chance table")
        _TABLES[path] = Chances({tuple(k.split("|")): c for k, c in blob["cells"].items()}, blob.get("header"))
    return _TABLES[path]


def chances_path(tag: str) -> str:
    return paths.derived("question_chances", f"{tag}.json")


def save_chances(path: str, cells: dict, header: dict) -> None:
    Path(path).write_text(json.dumps({"format": CHANCE_FORMAT, "header": header, "cells": {"|".join(k): {str(h): v for h, v in c.items()} for k, c in sorted(cells.items())}}, indent=1))


def cells_from_records(records: list, horizons: tuple = HORIZONS) -> dict:
    """Counts from records [{"keys": [key, ...], "out": {horizon: taken (True/False) or None when the game ended too soon to say}}]: every key a question passed through (one per distinct state)
    adds one to the total of its cell and, when the exact unit was taken within the horizon, one to the taken. Outcomes None are not counted."""
    cells: dict = {}
    for rec in records:
        for key in rec["keys"]:
            c = cells.setdefault(tuple(key), {h: [0, 0] for h in horizons})
            for h in horizons:
                o = rec["out"].get(h, rec["out"].get(str(h)))
                if o is None:
                    continue
                c[h][1] += 1
                c[h][0] += bool(o)
    return cells


def merge_cells(parts: list) -> dict:
    """The sum of several tables of counts (the fit run in several processes over different games)."""
    out: dict = {}
    for cells in parts:
        for key, c in cells.items():
            slot = out.setdefault(tuple(key), {})
            for h, (taken, total) in c.items():
                t = slot.setdefault(int(h), [0, 0])
                t[0] += taken
                t[1] += total
    return out


def load_games(directory: str) -> list:
    """[{variant, start, moves}] of a directory's own games. Only the moves are read: the judge's labels and balances in the other files are not."""
    out = []
    for line in (Path(directory) / "games.jsonl").read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            out.append({"variant": g.get("variant"), "start": g.get("start"), "moves": g["moves"]})
    return out


def pick_positions(dirs: list, stride: int = 1, limit: "int | None" = None) -> list:
    """[(directory index, game index, ply)] for every ply of every game, every `stride`-th, then at most `limit` evenly spread."""
    allp = []
    for di, d in enumerate(dirs):
        for gi, g in enumerate(load_games(d)):
            allp += [(di, gi, ply) for ply in range(len(g["moves"]))]
    allp = allp[::max(1, stride)]
    if limit is not None and limit < len(allp):
        step_ = len(allp) / limit
        allp = [allp[int(i * step_)] for i in range(limit)]
    return allp


class Games:
    """The games of some directories with their unit tracks, each read once."""

    def __init__(self, dirs: list, env: Env):
        self.env, self.dirs = env, dirs
        self.games = [load_games(d) for d in dirs]
        self._tracks: dict = {}

    def tracks(self, di: int, gi: int) -> tuple:
        if (di, gi) not in self._tracks:
            g = self.games[di][gi]
            self._tracks[(di, gi)] = self.env.codec.tracks(g["start"], g["moves"])
        return self._tracks[(di, gi)]

    def position(self, di: int, gi: int, ply: int) -> str:
        g = self.games[di][gi]
        return CE.state(g["variant"] or self.env.variant, g["start"], g["moves"][:ply])[0]

    def events(self, di: int, gi: int) -> list:
        """The game's moves as events (see `unit_loss`), read once."""
        if ("events", di, gi) not in self._tracks:
            g = self.games[di][gi]
            self._tracks[("events", di, gi)] = self.env.codec.events(g["start"], g["moves"])
        return self._tracks[("events", di, gi)]

    def losses(self, di: int, gi: int, ply: int, place: str, worth_of, horizons: tuple = HORIZONS) -> "dict | None":
        """The outcome by LOSS: {horizon: was the exact unit that stands on `place` LOST (by `unit_loss`) within that many plies (True/False; an even exchange or a gain is False), None when it was not
        taken and the game ended before the horizon}, None for no unit there. The same shape as `outcomes`."""
        got = self.tracks(di, gi)[0][ply].get(place)
        if got is None:
            return None
        n = len(self.games[di][gi]["moves"])
        ev = self.events(di, gi)
        out = {}
        for h in horizons:
            res = unit_loss(ev, got[0], ply, h, worth_of)
            out[h] = (res["status"] == LOST) if res else (None if ply + h > n else False)
        return out

    def outcomes(self, di: int, gi: int, ply: int, place: str, horizons: tuple = HORIZONS) -> "dict | None":
        """{horizon: was the exact unit that stands on `place` taken within that many plies (True/False), None when not taken and the game ended before the horizon}, None for no unit there."""
        by_ply, taken = self.tracks(di, gi)
        got = by_ply[ply].get(place)
        if got is None:
            return None
        n = len(self.games[di][gi]["moves"])
        out = {}
        for h in horizons:
            hit = any(i in taken and taken[i][0] == got[0] for i in range(ply, min(ply + h, n)))
            out[h] = True if hit else (None if ply + h > n else False)
        return out


LOST, EVEN, GAINED = "lost", "even", "gained"


def unit_loss(events: list, unit_id, start: int, horizon: int, worth_of) -> "dict | None":
    """THE LOSS OF A UNIT, followed through a game's real moves (plan 78.42, replacing "was it taken" as the outcome).
    `events[i]` is the i-th move of the game: {mover: (id, side, kind), dest, taken: (id, side, kind) | None, taken_at}. The unit `unit_id` is looked for among the takings of the plies
    `start` .. `start + horizon - 1`. None when it is not taken there. Otherwise let X be the place it was taken on, and the RUN be
      (1) the plies, from the one that takes the unit back to `start` and forward to the end of the game, that are consecutive takings on X (the exchange on that place: the forward end is not
          cut by the horizon, the horizon decides only when the unit goes), and
      (2) the unit's own arriving capture: when the last move of the unit before it was taken ended on X and took something, that move too, wherever its ply lies (not before `start`).
    The BALANCE is, over the run, (worth of the units taken that belong to the unit's side) minus (worth of the units taken that belong to the other side), by `worth_of(kind)`.
    The unit is LOST when the balance is above 0, EVEN when it is 0, GAINED when it is below 0.
    -> {status, balance, place, ply (the ply it was taken), run (the plies, in order), arrival (the ply of its own arriving capture or None)}"""
    end = min(start + horizon, len(events))
    hit = next((i for i in range(max(start, 0), end) if events[i]["taken"] and events[i]["taken"][0] == unit_id), None)
    if hit is None:
        return None
    side, place = events[hit]["taken"][1], events[hit]["taken_at"]
    run = [hit]
    j = hit - 1
    while j >= start and events[j]["taken"] and events[j]["taken_at"] == place:
        run.insert(0, j)
        j -= 1
    k = hit + 1
    while k < len(events) and events[k]["taken"] and events[k]["taken_at"] == place:
        run.append(k)
        k += 1
    arrival = None
    for j in range(hit - 1, start - 1, -1):
        if events[j]["mover"][0] == unit_id:
            if events[j]["dest"] == place and events[j]["taken"]:
                arrival = j
                if j not in run:
                    run = sorted(run + [j])
            break
    balance = sum((1 if events[i]["taken"][1] == side else -1) * worth_of(events[i]["taken"][2]) for i in run)
    return {"status": LOST if balance > 0 else (EVEN if balance == 0 else GAINED), "balance": balance, "place": place, "ply": hit, "run": run, "arrival": arrival}


def question_records(sheet: Sheet, outcomes_of) -> list:
    """The records of one read position: for every non-goal question of a unit above the lowest worth, the keys it passed through (before any step, and after each level) and the outcomes.
    `outcomes_of(place) -> {horizon: bool|None} | None`. Each record also holds its final key, its owner and the unit's kind."""
    out = []
    for q in sheet.questions:
        if q.quiet or q.goal or not sheet.reading.weighty(q.target):
            continue
        oc = outcomes_of(place_text(q.target.at))
        if oc is None:
            continue
        keys = [question_key(q, -1)] + [question_key(q, f.level) for f in q.findings]
        keys = [k for i, k in enumerate(keys) if i == 0 or k != keys[i - 1]]
        out.append({"keys": [list(k) for k in keys], "final": list(keys[-1]), "owner": keys[-1][0], "unit": q.target.kind, "out": dict(oc)})
    return out


def fit_chances(dirs: list, env: Env, worth: dict, params: Params, *, stride: int = 1, limit: "int | None" = None, shard: tuple = (0, 1), say=print, outcome: str = "taken") -> tuple:
    """THE SLEEP SIDE. Read the positions of our own games (every `stride`-th, at most `limit`, those whose index is `shard[0]` modulo `shard[1]`), run the sheet as the measures do (no table,
    the present order), and for every question record its key and whether the exact unit was taken within 4 and within 8 plies (`outcome` "taken", the default) or LOST within them
    (`outcome` "loss": the balance of worth over the exchange on the place, see `unit_loss`). Nothing from a teacher or a judge is read. -> (cells, header)."""
    if outcome not in ("taken", "loss"):
        raise ValueError(f"outcome must be taken or loss, not {outcome!r}")
    worth_of = env.worth_fn(worth)
    params = Params(**{**params.__dict__, "chances": None, "clock": "present"})
    games = Games(dirs, env)
    todo = pick_positions(dirs, stride, limit)
    todo = [x for i, x in enumerate(todo) if i % shard[1] == shard[0]]
    records, unreadable, t0 = [], 0, time.perf_counter()
    for n, (di, gi, ply) in enumerate(todo):
        try:
            sheet = read_position(env, worth, params, games.position(di, gi, ply))
        except ValueError:
            unreadable += 1
            continue
        if outcome == "loss":
            records += question_records(sheet, lambda place, di=di, gi=gi, ply=ply: games.losses(di, gi, ply, place, worth_of))
        else:
            records += question_records(sheet, lambda place, di=di, gi=gi, ply=ply: games.outcomes(di, gi, ply, place))
        if say and (n + 1) % 25 == 0:
            say(f"OBS fit progress: shard {shard[0]}/{shard[1]}: {n + 1} of {len(todo)} positions, {len(records)} records, {time.perf_counter() - t0:.0f} s")
    header = {"format": CHANCE_FORMAT, "key": list(KEY_NAMES), "horizons": list(HORIZONS), "smoothing": SMOOTH, "directories": [Path(d).name for d in dirs], "stride": stride, "limit": limit,
              "positions": len(todo) - unreadable, "unreadable": unreadable, "records": len(records), "seconds": time.perf_counter() - t0, "shards": [list(shard)], "outcome": outcome,
              "source": "our own engine's games: the moves only; the outcome is whether the exact unit was " + ("lost (balance of worth over the exchange on its place)" if outcome == "loss" else "taken") + " in the game; no teacher number"}
    return cells_from_records(records), header


def merge_headers(headers: list) -> dict:
    out = dict(headers[0])
    for h in headers[1:]:
        for k in ("positions", "unreadable", "records", "seconds"):
            out[k] = out.get(k, 0) + h.get(k, 0)
        out["shards"] = out.get("shards", []) + h.get("shards", [])
    return out


def format_chances(table: Chances, horizon: int = 4) -> list:
    """OBS lines: the fit's size and every cell of the table (key, taken, total, chance), the coarser keys after."""
    h = table.header
    out = [f"OBS chances: fit sample {h.get('positions', '?')} positions read ({h.get('unreadable', '?')} unreadable) of {h.get('directories')}, {h.get('records', '?')} question records, {sum(t for _k, c in table.cells.items() for t in [c[horizon][1]])} "
           f"state records counted at horizon {horizon}; smoothing: chance = (taken + a) / (total + a + b), a = {table.smooth:g} x p, b = {table.smooth:g} x (1 - p), p the chance of the next coarser key (the root: (taken + 1) / (total + 2))"]
    for n in range(len(KEY_NAMES), -1, -1):
        keys = sorted({k[:n] for k in table.cells})
        for key in keys:
            taken, total = table.counts(key, horizon)
            out.append(f"OBS chance {'(' + ', '.join(f'{a}={b}' for a, b in zip(KEY_NAMES, key)) + ')' if key else '(all)'} horizon {horizon}: taken {taken}, total {total}, chance {table.chance(key, horizon):.4f}"
                       + (f"; horizon 8: taken {table.counts(key, 8)[0]}, total {table.counts(key, 8)[1]}, chance {table.chance(key, 8):.4f}" if 8 in HORIZONS and 8 != horizon else ""))
    return out


CALIBRATION_BANDS = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01))


def format_calibration(records: list, table: Chances, horizon: int = 4, label: str = "") -> list:
    """OBS lines: for each band of the predicted chance of the question's FINAL key, how many questions fell in it, the mean predicted chance and the share whose exact unit was taken. Brier score of the
    table against the base rate of the fit (the same chance for every question)."""
    rows = [(table.chance(tuple(r["final"]), horizon), r["out"].get(horizon, r["out"].get(str(horizon)))) for r in records]
    rows = [(p, bool(o)) for p, o in rows if o is not None]
    out = [f"OBS calibration {label}: {len(rows)} questions, horizon {horizon}, taken {sum(o for _p, o in rows)} ({_pct(sum(o for _p, o in rows), len(rows))})"]
    for lo, hi in CALIBRATION_BANDS:
        sel = [(p, o) for p, o in rows if lo <= p < hi]
        if sel:
            out.append(f"OBS calibration {label}: predicted {lo:g} to {min(hi, 1):g}: {len(sel)} questions, mean predicted {sum(p for p, _o in sel) / len(sel):.4f}, observed {sum(o for _p, o in sel)} taken ({_pct(sum(o for _p, o in sel), len(sel))})")
        else:
            out.append(f"OBS calibration {label}: predicted {lo:g} to {min(hi, 1):g}: 0 questions")
    if rows:
        base = table.chance((), horizon)
        brier = sum((p - o) ** 2 for p, o in rows) / len(rows)
        brier0 = sum((base - o) ** 2 for _p, o in rows) / len(rows)
        out.append(f"OBS calibration {label}: Brier score of the table {brier:.5f}, of the base rate alone ({base:.4f}) {brier0:.5f}")
    return out


def inherit_chances(graph: Graph) -> None:
    """A node with no chance of its own (an obstacle, a follow-up, an answer, a condition) takes the chance of the node it hangs from, in the order the edges were made."""
    for e in graph.edges:
        dst = graph.nodes[e.dst]
        if dst.chance is None and graph.nodes[e.src].chance is not None:
            dst.chance = graph.nodes[e.src].chance


def assign_chances(q: Question, r: Reading, params: Params) -> None:
    """With a table (`params.chances`): the question's node gets the chance of its key as the question now stands; every node below it that has none takes its parent's."""
    if not params.chances or q.node is None or q.quiet:
        return
    table = load_chances(params.chances)
    r.graph.nodes[q.node].chance = table.chance(question_key(q), params.chance_horizon)
    inherit_chances(r.graph)


def way_chance(r: Reading, q: Question, way: tuple, entry: dict, params: Params) -> None:
    """A way node's chance: the table's, for (owner of the target, the way's verdict, its moves, its route's group)."""
    if not params.chances:
        return
    table = load_chances(params.chances)
    obs = entry.get("first_obstacles") or []
    group = "clear" if not obs else ("theirs" if any((side is not None and side != q.asker) or dead for side, _k, dead in obs) else "own")
    kind = {CONFIRMED_V: "confirmed", WAITING_V: "waiting", UNSETTLED_V: "unsettled"}.get(entry["status"], "refuted")
    wn = way_node(r, q, way)
    wn.chance = table.chance(("mine" if q.owner == SELF else "theirs", kind, str(way[0]) if way[0] < 3 else "3+", group), params.chance_horizon)
    inherit_chances(r.graph)


# ---------------------------------------------------------------------------------------------------------------------------------------------- J: the clock against a budget, on games the fit never saw

J_ORDERS = ("plain", "present", "chance")
J_TIMES = (0.02, 0.05, 0.2, 1.0)       # seconds a position, the reading of the position counted: real runs under the budget
J_STEPS = (10, 20, 40, 80)             # steps: read off the one unbudgeted run, exactly
J_COSTS = (0.02, 0.05, 0.2, 1.0)       # summed measured cost of the steps taken, seconds: read off the one unbudgeted run, exactly
J_FIELDS = ("positions", "seconds", "steps", "mine_units", "mine_taken", "mine_taken_flagged", "mine_flagged", "mine_flagged_taken",
            "theirs_units", "theirs_taken", "theirs_taken_flagged", "theirs_flagged", "theirs_flagged_taken",
            "mine_lost", "mine_lost_flagged", "mine_flagged_lost", "theirs_lost", "theirs_lost_flagged", "theirs_flagged_lost")


def _j_tally(acc: dict, cfg: str, kinds: dict, sheet: Sheet, outcome_of, seconds: float, steps: int, loss_of=None) -> None:
    """Add one position read under one configuration: for each non-goal question of a unit above the lowest worth whose unit's fate in 4 plies is known, the verdict kind it had when the
    budget ran out (`kinds`, by question), whether it says danger, and whether the exact unit was taken within 4 plies. With `loss_of` (place -> {horizon: LOST or not}, see `Games.losses`) the
    columns `*_lost`, `*_lost_flagged`, `*_flagged_lost` are the same counts with LOST in place of taken."""
    a = acc.setdefault(cfg, {f: 0 for f in J_FIELDS})
    a["positions"] += 1
    a["seconds"] += seconds
    a["steps"] += steps
    for q in sheet.questions:
        if q.quiet or q.goal or not sheet.reading.weighty(q.target):
            continue
        oc = outcome_of(place_text(q.target.at))
        if oc is None or oc.get(4) is None:
            continue
        side = "mine" if q.owner == SELF else "theirs"
        flagged = kinds.get(id(q), "unread") in DANGER
        taken = bool(oc[4])
        a[f"{side}_units"] += 1
        a[f"{side}_taken"] += taken
        a[f"{side}_taken_flagged"] += taken and flagged
        a[f"{side}_flagged"] += flagged
        a[f"{side}_flagged_taken"] += taken and flagged
        lo = loss_of(place_text(q.target.at)) if loss_of is not None else None
        if lo is not None and lo.get(4) is not None:
            lost = bool(lo[4])
            a[f"{side}_lost"] += lost
            a[f"{side}_lost_flagged"] += lost and flagged
            a[f"{side}_flagged_lost"] += lost and flagged


def j_position(env: Env, worth: dict, params: Params, position: str, outcome_of, acc: dict, table_path: "str | None", orders: tuple = J_ORDERS, timed: bool = True, loss_of=None) -> None:
    """Measure J on one position: for each order, the real runs under the time budgets, then one unbudgeted run whose every step is logged, from which the unbudgeted row and the rows
    for budgets counted in steps and in summed measured step cost are read exactly."""
    for order in orders:
        p = Params(**{**params.__dict__, "clock": order, "chances": table_path if order == "chance" else None})
        for b in (J_TIMES if timed else ()):
            sheet = run(position, worth, budget_seconds=b, env=env, params=p)
            _j_tally(acc, f"{order}|time|{b * 1000:g} ms", {id(q): verdict_kind(q) for q in sheet.questions if not q.quiet}, sheet, outcome_of, sheet.seconds, len(sheet.steps), loss_of)
        log, cur = [], {}
        cost = [0.0]

        def on_step(sheet, st, elapsed, log=log, cur=cur, cost=cost):
            cur[id(st.question)] = verdict_kind(st.question)
            cost[0] += st.cost
            log.append((elapsed, cost[0], len(sheet.steps), dict(cur)))
        sheet = run(position, worth, max_steps=10 ** 6, env=env, params=p, on_step=on_step)
        first = (sheet.read_seconds, 0.0, 0, {})
        log = [first] + log
        _j_tally(acc, f"{order}|none|unbudgeted", log[-1][3], sheet, outcome_of, sheet.seconds, len(sheet.steps), loss_of)
        for n in J_STEPS:
            e = log[min(n, len(log) - 1)]
            _j_tally(acc, f"{order}|steps|{n} steps", e[3], sheet, outcome_of, e[0], e[2], loss_of)
        for b in J_COSTS:
            e = next((x for x in log if x[1] >= b), log[-1])
            _j_tally(acc, f"{order}|cost|{b * 1000:g} ms of step cost", e[3], sheet, outcome_of, e[0], e[2], loss_of)


def measure_j(dirs: list, env: Env, worth: dict, params: Params, table_path: str, *, stride: int = 1, limit: "int | None" = None, shard: tuple = (0, 1), say=print, orders: tuple = J_ORDERS, timed: bool = True) -> dict:
    games = Games(dirs, env)
    worth_of = env.worth_fn(worth)
    todo = [x for i, x in enumerate(pick_positions(dirs, stride, limit)) if i % shard[1] == shard[0]]
    acc: dict = {}
    t0 = time.perf_counter()
    for n, (di, gi, ply) in enumerate(todo):
        try:
            j_position(env, worth, params, games.position(di, gi, ply), lambda place, di=di, gi=gi, ply=ply: games.outcomes(di, gi, ply, place), acc, table_path, orders, timed,
                       loss_of=lambda place, di=di, gi=gi, ply=ply: games.losses(di, gi, ply, place, worth_of))
        except ValueError:
            continue
        if say and (n + 1) % 10 == 0:
            say(f"OBS J progress: shard {shard[0]}/{shard[1]}: {n + 1} of {len(todo)} positions, {time.perf_counter() - t0:.0f} s")
    return acc


def merge_counts(parts: list) -> dict:
    """The sum of several measure-J tallies (positions split over processes)."""
    out: dict = {}
    for acc in parts:
        for cfg, a in acc.items():
            slot = out.setdefault(cfg, {f: 0 for f in J_FIELDS})
            for f in J_FIELDS:
                slot[f] += a.get(f, 0)
    return out


def format_j(acc: dict) -> list:
    """OBS lines, one table per way of counting the budget: for each order and budget, of the units that were taken within 4 plies the share with a danger verdict (mine, theirs), the danger
    verdicts raised a position and the share of them followed by a taking (mine, theirs), the steps and the time used a position."""
    out = []
    families = [("time", "budget in wall-clock time a position (the reading counted; real runs)"), ("steps", "budget counted in steps (read off the unbudgeted run)"),
                ("cost", "budget counted in summed measured step cost (read off the unbudgeted run)"), ("none", "no budget")]
    for fam, title in families:
        out.append(f"OBS (J) {title}:")
        cfgs = sorted({c.split("|", 2)[2] for c in acc if c.split("|")[1] == fam}, key=lambda x: float(x.split()[0]) if x[0].isdigit() else 0.0)
        for budget in cfgs:
            for order in J_ORDERS:
                a = acc.get(f"{order}|{fam}|{budget}")
                if a is None:
                    continue
                n = max(a["positions"], 1)

                def part(side):
                    return (f"{side}: taken in 4 plies {a[side + '_taken']} of {a[side + '_units']} units, with a danger verdict {a[side + '_taken_flagged']} ({_pct(a[side + '_taken_flagged'], a[side + '_taken'])}); "
                            f"danger verdicts {a[side + '_flagged']} ({a[side + '_flagged'] / n:.3f} a position), followed by a taking {a[side + '_flagged_taken']} ({_pct(a[side + '_flagged_taken'], a[side + '_flagged'])})")
                out.append(f"OBS (J) {budget}, order {order}: positions {a['positions']}; {part('mine')}; {part('theirs')}; steps {a['steps'] / n:.1f} and time {1000 * a['seconds'] / n:.1f} ms a position")
    out.append("OBS (J, by LOSS) the same tables with LOST in place of taken (the balance of worth over the exchange on the place where the unit was taken, see `unit_loss`; an even exchange or a gain is not a loss):")
    for fam, title in families:
        cfgs = sorted({c.split("|", 2)[2] for c in acc if c.split("|")[1] == fam}, key=lambda x: float(x.split()[0]) if x[0].isdigit() else 0.0)
        for budget in cfgs:
            for order in J_ORDERS:
                a = acc.get(f"{order}|{fam}|{budget}")
                if a is None:
                    continue

                def lpart(side):
                    return (f"{side}: LOST in 4 plies {a.get(side + '_lost', 0)} of {a[side + '_units']} units, with a danger verdict {a.get(side + '_lost_flagged', 0)} ({_pct(a.get(side + '_lost_flagged', 0), a.get(side + '_lost', 0))}); "
                            f"danger verdicts {a[side + '_flagged']}, followed by a loss {a.get(side + '_flagged_lost', 0)} ({_pct(a.get(side + '_flagged_lost', 0), a[side + '_flagged'])})")
                out.append(f"OBS (J, by LOSS) {budget}, order {order}: positions {a['positions']}; {lpart('mine')}; {lpart('theirs')}")
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------- K: the question of the move against what the game did

K_HORIZON = 4                          # plies within which a unit of the mover is LOST (counted from the move played)


def bad_move_labels(directory: str) -> set:
    """REPORTING ONLY. {(game index, ply)} of the plies the judge's file marks as a bad move (a missed surprise whose jump is below 0). The label is never read by a decision or a fit; measure K prints
    the share of bad moves among the verdicts and nothing more."""
    d = Path(directory)
    ids = [json.loads(ln).get("game") for ln in (d / "games.jsonl").read_text().splitlines() if ln.strip()]
    index = {gid: gi for gi, gid in enumerate(ids)}
    out = set()
    for line in (d / "surprises.jsonl").read_text().splitlines():
        if line.strip():
            sv = json.loads(line)
            if sv.get("kind") == "MISSED" and (sv.get("judge") or {}).get("jump", 0) < 0 and sv.get("game") in index:
                out.add((index[sv["game"]], sv["ply"]))
    return out


def lost_units_of_mover(games: Games, r: Reading, di: int, gi: int, ply: int, worth_of, horizon: int = K_HORIZON) -> dict:
    """{handle before the move: the `unit_loss` result} of the mover's units above the least worth (not the goal unit) that are taken within `horizon` plies of the game from `ply`."""
    by_ply = games.tracks(di, gi)[0][ply]
    events = games.events(di, gi)
    out = {}
    for u in r.units:
        if u.side != SELF or not r.weighty(u) or u == r.goal_unit(SELF):
            continue
        got = by_ply.get(place_text(u.at))
        if got is None:
            continue
        res = unit_loss(events, got[0], ply, horizon, worth_of)
        if res is not None:
            out[place_text(u.at)] = res
    return out


def k_position(env: Env, worth: dict, params: Params, games: Games, di: int, gi: int, ply: int, bad: "set | None" = None) -> "dict | None":
    """Measure K on one position: the move sheet of every legal move, and for the move played in the game its verdict against what the game then did. None when the position cannot be read."""
    pos = games.position(di, gi, ply)
    t0 = time.perf_counter()
    try:
        r = Reading(env, pos, worth)
    except ValueError:
        return None
    r.params_k, r.passes = params.reach_k, params.passes
    read_ms = 1000 * (time.perf_counter() - t0)
    played = games.games[di][gi]["moves"][ply]
    sheet = move_sheet(env, r, params)
    by_move = {x["move"]: x for x in sheet.results}
    if played not in by_move:
        return None
    res = by_move[played]
    by_ply = games.tracks(di, gi)[0][ply]
    lost = lost_units_of_mover(games, r, di, gi, ply, env.worth_fn(worth))
    status = {h: v["status"] for h, v in lost.items()}
    lost_h = sorted(h for h, v in status.items() if v == LOST)
    origin = env.codec.move_ends(played)[0]
    named_h = res["named"]["place"] if res["named"] else None
    parts = {k: sum(x["ms"][k] for x in sheet.results) for k in ("a", "read", "b", "c", "total")}
    return {"di": di, "gi": gi, "ply": ply, "played": played, "verdict": res["verdict"], "named": named_h, "named_part": res["named"]["part"] if res["named"] else None,
            "named_lost": bool(named_h and by_ply.get(named_h) and by_ply[named_h][0] in {by_ply[h][0] for h in lost_h}),
            "lost": lost_h, "statuses": status, "moved": origin, "moved_lost": origin in lost_h, "other_lost": any(h != origin for h in lost_h),
            "bad": None if bad is None else (gi, ply) in bad, "legal": len(r.legal), "loses": len(sheet.loses()), "played_in_loses": res["verdict"] == LOSES_V,
            "verdicts": dict(sheet.verdicts()), "read_ms": read_ms, "sheet_ms": parts["total"], "parts_ms": parts, "max_move_ms": max(x["ms"]["total"] for x in sheet.results),
            "answer": " | ".join(res["answers"].values())[:300]}


def measure_k(dirs: list, env: Env, worth: dict, params: Params, *, stride: int = 1, limit: "int | None" = None, shard: tuple = (0, 1), say=print) -> list:
    games = Games(dirs, env)
    bad = bad_move_labels(dirs[0]) if len(dirs) == 1 else None
    todo = [x for i, x in enumerate(pick_positions(dirs, stride, limit)) if i % shard[1] == shard[0]]
    out, t0 = [], time.perf_counter()
    for n, (di, gi, ply) in enumerate(todo):
        rec = k_position(env, worth, params, games, di, gi, ply, bad)
        if rec is not None:
            out.append(rec)
        if say and (n + 1) % 10 == 0:
            say(f"OBS K progress: shard {shard[0]}/{shard[1]}: {n + 1} of {len(todo)} positions, {time.perf_counter() - t0:.0f} s")
    return out


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def format_k(records: list) -> list:
    """OBS lines of measure K: the verdict on the move played against whether a unit of the mover was LOST within 4 plies, by which unit, the bad-move share, the move sheet's counts and the costs."""
    n = len(records)
    out = [f"OBS (K) positions {n}; the verdict of the question of the move on the move actually played, against whether a unit of the mover above the least worth was LOST (see `unit_loss`) within {K_HORIZON} plies"]
    verdicts = (LOSES_V, SAFE_V, MOVE_UNSETTLED)
    out.append("OBS (K) table: verdict | a unit of the mover LOST | none lost | total")
    for v in verdicts:
        rows = [x for x in records if x["verdict"] == v]
        out.append(f"OBS (K)   {v}: {sum(1 for x in rows if x['lost'])} | {sum(1 for x in rows if not x['lost'])} | {len(rows)}")
    out.append("OBS (K) the same split by which unit was lost (positions with a loss): verdict | the moved unit only | another unit only | both | the verdict named a unit that was lost | named a unit that was not lost | named none")
    for v in verdicts:
        rows = [x for x in records if x["verdict"] == v and x["lost"]]
        both = sum(1 for x in rows if x["moved_lost"] and x["other_lost"])
        out.append(f"OBS (K)   {v}: {sum(1 for x in rows if x['moved_lost'] and not x['other_lost'])} | {sum(1 for x in rows if x['other_lost'] and not x['moved_lost'])} | {both} | "
                   f"{sum(1 for x in rows if x['named_lost'])} | {sum(1 for x in rows if x['named'] and not x['named_lost'])} | {sum(1 for x in rows if not x['named'])}")
    out.append("OBS (K) by which part named the unit (LOSES): " + ", ".join(f"{p}: {sum(1 for x in records if x['verdict'] == LOSES_V and x['named_part'] == p)} named, of which lost in the game "
               f"{sum(1 for x in records if x['verdict'] == LOSES_V and x['named_part'] == p and x['named_lost'])}" for p in "abc"))
    judged = [x for x in records if x["bad"] is not None]
    allbad = sum(1 for x in judged if x["bad"])
    out.append(f"OBS (K) against the judge's bad-move label (reporting only; never used by a decision): bad moves {allbad} of {len(judged)} ({_pct(allbad, len(judged))}); " + "; ".join(
        f"among {v} {sum(1 for x in judged if x['verdict'] == v and x['bad'])} of {sum(1 for x in judged if x['verdict'] == v)} ({_pct(sum(1 for x in judged if x['verdict'] == v and x['bad']), sum(1 for x in judged if x['verdict'] == v))})"
        for v in verdicts))
    out.append(f"OBS (K) the move sheet: legal moves a position {_mean([x['legal'] for x in records]):.1f}; moves it calls LOSES a position {_mean([x['loses'] for x in records]):.2f}; "
               f"positions with at least one LOSES {sum(1 for x in records if x['loses'])} ({_pct(sum(1 for x in records if x['loses']), n)}); the move played is among them in {sum(1 for x in records if x['played_in_loses'])} positions; "
               f"of the positions with at least one LOSES the move played is among them in {sum(1 for x in records if x['loses'] and x['played_in_loses'])} ({_pct(sum(1 for x in records if x['loses'] and x['played_in_loses']), sum(1 for x in records if x['loses']))})")
    legal = sum(x["legal"] for x in records)
    tot = sum(x["parts_ms"]["total"] for x in records)
    out.append(f"OBS (K) cost: a move question mean {tot / max(legal, 1):.1f} ms, max {max([x['max_move_ms'] for x in records] or [0]):.1f} ms ({legal} questions); a whole move sheet mean {_mean([x['sheet_ms'] for x in records]):.0f} ms, "
               f"max {max([x['sheet_ms'] for x in records] or [0]):.0f} ms; the reading of the position itself mean {_mean([x['read_ms'] for x in records]):.1f} ms")
    out.append("OBS (K) cost split of a move question, mean ms a question (share of the total): " + ", ".join(
        f"{k} {sum(x['parts_ms'][k] for x in records) / max(legal, 1):.1f} ({100.0 * sum(x['parts_ms'][k] for x in records) / tot if tot else 0.0:.0f}%)" for k in ("a", "read", "b", "c")) + " (read = the position after the move, read once)")
    return out


def k_missed(env: Env, worth: dict, params: Params, dirs: list, files: list, examples: int = 5, say=print) -> list:
    """The mover's units that the earlier study found taken within 4 plies after having moved, and that no danger verdict foresaw (rows of its `missed_out*.json`): for each, the question of the move on
    the move played at that position. OBS lines: how many the question calls LOSES, how many SAFE where part 1's rule says EVEN or GAINED (not a miss at all), how many remain SAFE yet LOST (true misses),
    with `examples` of the last group explained."""
    games = Games(dirs, env)
    wf = env.worth_fn(worth)
    rows = [e for f in files for e in json.loads(Path(f).read_text())]
    mine = [e for e in rows if e["side"] == "mine" and e["moved"] and not e["flagged"]]
    out = [f"OBS (K, missed) rows read {len(rows)}; mine, no danger verdict foresaw it, the unit had moved, then was taken: {len(mine)}"]
    table: dict = {}
    true_misses, own_first, named_it, not_taken = [], 0, 0, 0
    for e in mine:
        gi, ply = e["game"], e["ply"]
        played = games.games[0][gi]["moves"][ply]
        r = Reading(env, games.position(0, gi, ply), worth)
        r.params_k, r.passes = params.reach_k, params.passes
        res = move_question(env, r, played, params)
        by_ply = games.tracks(0, gi)[0][ply]
        uid = by_ply[e["unit_place"]][0]
        loss = unit_loss(games.events(0, gi), uid, ply, K_HORIZON, wf)
        st = loss["status"] if loss else "not taken"
        table[(res["verdict"], st)] = table.get((res["verdict"], st), 0) + 1
        own_first += played[:2] == e["unit_place"]
        if res["verdict"] == LOSES_V and res["named"] and by_ply.get(res["named"]["place"]) and by_ply[res["named"]["place"]][0] == uid:
            named_it += 1
        if res["verdict"] == SAFE_V and st == LOST:
            true_misses.append((e, res))
    out.append("OBS (K, missed) verdict of the question of the move on the move played | part 1's rule on the unit taken in the game: " + ", ".join(sorted(f"{v}/{s}: {c}" for (v, s), c in table.items())))
    total = len(mine)
    calls = lambda v: sum(c for (vv, _s), c in table.items() if vv == v)       # noqa: E731
    not_miss = sum(c for (vv, s), c in table.items() if vv == SAFE_V and s in (EVEN, GAINED))
    out.append(f"OBS (K, missed) of {total}: called LOSES {calls(LOSES_V)} (it named the very unit in {named_it}); called SAFE {calls(SAFE_V)}, of which the taking was EVEN or GAINED by part 1's rule (not a miss at all) {not_miss}, "
               f"and LOST (true misses) {len(true_misses)}; NOT SETTLED {calls(MOVE_UNSETTLED)}; the move played at that ply was the unit's own move in {own_first}")
    for e, res in true_misses[:examples]:
        out.append(f"OBS (K, missed) example: game {e['game']} ply {e['ply']}, the unit {e['unit_place']} ({e['unit_kind']}) taken after {' '.join(e['seq'])}; position {e['pos']}")
        out += [f"OBS (K, missed)   {ln}" for ln in explain_move(res)]
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
    ap.add_argument("--games-dir", action="append", default=[], help="a surprise meter directory; with --goal-verdict, --fit-chances, --calibrate and --measure-j the flag may repeat")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-links", action="store_true", help="level 1 by reach alone (the first version)")
    ap.add_argument("--no-frames", action="store_true", help="level 2 of a non-goal unit by the earlier forms (the search, the census) instead of the exchange frame")
    ap.add_argument("--no-pass", action="store_true", help="the asker not to move: the static form instead of the kernel's pass and real moves")
    ap.add_argument("--no-goal-near", action="store_true", help="the goal unit's level 1 by the earlier rule (no asker unit needed near)")
    ap.add_argument("--no-geometry", action="store_true", help="level 1 by the move tables, the board frozen (the earlier form), instead of the medium and the obstacles")
    ap.add_argument("--no-routes", action="store_true", help="a way of more than one move judged after its first move only (the earlier form)")
    ap.add_argument("--reach-k", type=int, default=None, help="moves (hops) a unit of the asker may use (default 2)")
    ap.add_argument("--route-nodes", type=int, default=None, help="node cap of one way's route reasoning")
    ap.add_argument("--anticipation", type=int, default=None, help="measure I over the first N games of the (single) --games-dir, the pool carried from ply to ply")
    ap.add_argument("--no-search", action="store_true", help="level 2 by the census alone (the first version)")
    ap.add_argument("--alternatives", type=int, default=100, help="bad-move positions whose other legal moves are read for the sheet's objection (0: none)")
    ap.add_argument("--no-objection", action="store_true", help="skip measure (e)")
    ap.add_argument("--goal-verdict", action="store_true", help="measure F: the goal unit's question against how the games ended (levels 0 and 1 only)")
    ap.add_argument("--flagged", action="store_true", help="measure G: the level 2 result on the flagged units of both sides, followed for 4 plies")
    ap.add_argument("--leaving-full", action="store_true", help="the cost of leaving by a whole new reading of the child position (the earlier form)")
    ap.add_argument("--leaving-wide-sign", action="store_true", help="the cost of leaving reads the goal sign whenever the step touches a line to the goal unit")
    ap.add_argument("--leaving-audit", type=int, default=0, help="every n-th cost of leaving is also read by the whole reread and compared")
    ap.add_argument("--search-depth", type=int, default=None, help="the deepest level and the deepest search (default: the Params'); 2 keeps G to depth 2")
    ap.add_argument("--games", type=int, default=None, help="with --goal-verdict: at most this many games")
    ap.add_argument("--out", default=None, help="a name under the derived data directory to write the OBS lines to")
    ap.add_argument("--no-joint", action="store_true", help="a step no single removal explains is recorded as \"none\" (the earlier form), not tried with the set of units the first layer names")
    ap.add_argument("--max-joint", type=int, default=None, help="units in a set that must all go (default 3); more is refuted by cost")
    ap.add_argument("--fit-chances", metavar="TAG", default=None, help="sleep: fit the table of chances from the games of every --games-dir and write it under the chances directory as TAG.json (with --shard: TAG.shardK.json)")
    ap.add_argument("--merge-chances", metavar="TAG", default=None, help="sum the TAG.shard*.json tables into TAG.json and print them")
    ap.add_argument("--show-chances", metavar="PATH", default=None, help="print a table as OBS lines")
    ap.add_argument("--chances", metavar="PATH", default=None, help="wake: the table of chances every question and node reads its chance from")
    ap.add_argument("--chance-outcome", choices=("taken", "loss"), default="taken", help="with --fit-chances: the outcome the table is fitted on: the exact unit taken (default) or LOST (balance of worth over the exchange on its place)")
    ap.add_argument("--chance-horizon", type=int, default=4, help="the horizon (4 or 8 plies) the chance is read at")
    ap.add_argument("--shard", default="0/1", help="K/N: this process reads the positions whose index is K modulo N")
    ap.add_argument("--stride", type=int, default=1, help="every n-th ply of the games is read (fit, calibrate, measure J)")
    ap.add_argument("--calibrate", metavar="PATH", default=None, help="read the --games-dir games' positions and print the table's predicted chance against what happened, by band; --records-out writes the records for merging")
    ap.add_argument("--records-out", default=None, help="with --calibrate: write the records (final key and outcomes) to this file")
    ap.add_argument("--calibrate-merge", nargs="+", default=None, help="records files written by --records-out: calibration over all of them, against --chances")
    ap.add_argument("--clock", choices=("present", "chance", "plain"), default="present", help="the order of steps: present (stake x urgency over cost), chance (chance x stake at risk over cost), plain (first order, each question to full depth)")
    ap.add_argument("--risk", type=float, default=None, help="tolerance for risk, 0 to 1: the stake at risk of my own unit is risk x worst case + (1 - risk) x expected (default 1)")
    ap.add_argument("--measure-j", action="store_true", help="measure J on the --games-dir games: the three orders against budgets (needs --chances); --j-out writes the raw tallies")
    ap.add_argument("--j-orders", default=",".join(J_ORDERS), help="with --measure-j: the orders to run, comma separated")
    ap.add_argument("--j-no-time", action="store_true", help="with --measure-j: skip the real runs under wall-clock budgets")
    ap.add_argument("--j-out", default=None, help="with --measure-j: write the raw tallies (JSON) here")
    ap.add_argument("--j-merge", nargs="+", default=None, help="tally files written by --j-out: merge them and print the tables")
    ap.add_argument("--measure-k", action="store_true", help="measure K on the --games-dir game: the question of the move on the move played, against what the game then did; --k-out writes the raw records")
    ap.add_argument("--k-out", default=None, help="with --measure-k: write the raw records (JSON) here")
    ap.add_argument("--k-merge", nargs="+", default=None, help="record files written by --k-out: merge them and print the tables")
    ap.add_argument("--k-missed", nargs="+", default=None, help="the study's missed-unit files (missed_out*.json): the question of the move on each mover's unit that moved and was taken, with the --games-dir game")
    ap.add_argument("--move-question", action="store_true", help="the question of each legal move is asked after the questions and kept on the sheet")
    a = ap.parse_args(argv)
    shard = tuple(int(x) for x in a.shard.split("/"))
    env = Env(a.game)
    worth = store_worth(a.game)
    params = Params(links=not a.no_links, search=not a.no_search, frames=not a.no_frames, goal_near=not a.no_goal_near, passes=not a.no_pass, geometry=not a.no_geometry, routes=not a.no_routes)
    params.leaving_full, params.leaving_audit_every, params.leaving_wide_sign = a.leaving_full, a.leaving_audit, a.leaving_wide_sign
    params.joint = not a.no_joint
    if a.max_joint is not None:
        params.max_joint = a.max_joint
    params.chances, params.chance_horizon, params.clock = a.chances, a.chance_horizon, a.clock
    if a.risk is not None:
        params.risk = a.risk
    if a.reach_k:
        params.reach_k = a.reach_k
    if a.route_nodes:
        params.route_nodes = a.route_nodes
    if a.search_depth:
        params.max_level = params.search_depth_cap = a.search_depth
    t0 = time.perf_counter()
    params.move_question = a.move_question
    if a.k_merge:
        for ln in format_k([x for f in a.k_merge for x in json.loads(Path(f).read_text())]):
            print(ln)
        return 0
    if a.merge_chances or a.show_chances or a.j_merge or a.calibrate_merge:
        if a.merge_chances:
            parts = sorted(Path(chances_path(a.merge_chances)).parent.glob(f"{a.merge_chances}.shard*.json"))
            blobs = [json.loads(x.read_text()) for x in parts]
            cells = merge_cells([{tuple(k.split("|")): {int(h): v for h, v in c.items()} for k, c in b["cells"].items()} for b in blobs])
            header = merge_headers([b["header"] for b in blobs])
            save_chances(chances_path(a.merge_chances), cells, header)
            print(f"OBS chances: merged {len(parts)} shard files into {chances_path(a.merge_chances)}")
            a.show_chances = chances_path(a.merge_chances)
        if a.show_chances:
            for ln in format_chances(load_chances(a.show_chances), a.chance_horizon):
                print(ln)
        if a.j_merge:
            for ln in format_j(merge_counts([json.loads(Path(x).read_text()) for x in a.j_merge])):
                print(ln)
        if a.calibrate_merge:
            recs = [r for x in a.calibrate_merge for r in json.loads(Path(x).read_text())]
            for h in HORIZONS:
                for ln in format_calibration(recs, load_chances(a.chances), h, "on the test games"):
                    print(ln)
        return 0
    if not a.games_dir:
        ap.error("--games-dir is needed")
    if a.fit_chances:
        cells, header = fit_chances(a.games_dir, env, worth, params, stride=a.stride, limit=a.limit, shard=shard, say=lambda m: print(m, flush=True), outcome=a.chance_outcome)
        tag = a.fit_chances if shard[1] == 1 else f"{a.fit_chances}.shard{shard[0]}of{shard[1]}"
        header["params"] = {"max_joint": params.max_joint, "joint": params.joint, "reach_k": params.reach_k}
        save_chances(chances_path(tag), cells, header)
        print(f"OBS chances: fitted from {header['positions']} positions in {header['seconds']:.1f} s, {header['records']} question records, written to {chances_path(tag)}")
        return 0
    if a.calibrate:
        games = Games(a.games_dir, env)
        todo = [x for i, x in enumerate(pick_positions(a.games_dir, a.stride, a.limit)) if i % shard[1] == shard[0]]
        recs, unreadable = [], 0
        for n, (di, gi, ply) in enumerate(todo):
            try:
                sheet = read_position(env, worth, Params(**{**params.__dict__, "chances": None, "clock": "present"}), games.position(di, gi, ply))
            except ValueError:
                unreadable += 1
                continue
            recs += question_records(sheet, lambda place, di=di, gi=gi, ply=ply: games.outcomes(di, gi, ply, place))
            if (n + 1) % 25 == 0:
                print(f"OBS calibrate progress: {n + 1} of {len(todo)} positions, {time.perf_counter() - t0:.0f} s", flush=True)
        if a.records_out:
            Path(a.records_out).write_text(json.dumps(recs))
        print(f"OBS calibration: {len(todo) - unreadable} positions read ({unreadable} unreadable), {len(recs)} question records")
        for h in HORIZONS:
            for ln in format_calibration(recs, load_chances(a.calibrate), h, "on the test games"):
                print(ln)
        return 0
    if a.measure_k:
        recs = measure_k(a.games_dir, env, worth, params, stride=a.stride, limit=a.limit, shard=shard, say=lambda m: print(m, flush=True))
        if a.k_out:
            Path(a.k_out).write_text(json.dumps(recs))
        for ln in format_k(recs):
            print(ln)
        print(f"OBS (K) {time.perf_counter() - t0:.1f} s, shard {shard[0]}/{shard[1]}")
        return 0
    if a.k_missed:
        for ln in k_missed(env, worth, params, a.games_dir, a.k_missed):
            print(ln)
        return 0
    if a.measure_j:
        if not a.chances:
            ap.error("--measure-j needs --chances (the table the chance order reads)")
        acc = measure_j(a.games_dir, env, worth, params, a.chances, stride=a.stride, limit=a.limit, shard=shard, say=lambda m: print(m, flush=True), orders=tuple(a.j_orders.split(",")), timed=not a.j_no_time)
        if a.j_out:
            Path(a.j_out).write_text(json.dumps(acc))
        for ln in format_j(acc):
            print(ln)
        print(f"OBS (J) {time.perf_counter() - t0:.1f} s, shard {shard[0]}/{shard[1]}")
        return 0
    if a.goal_verdict:
        rows, endings, positions = goal_verdict(a.games_dir, env, worth, params, a.games)
        lines = format_goal_verdict(rows, endings, positions) + [f"OBS question sheet: goal verdict {time.perf_counter() - t0:.1f} s for {len(endings)} games, {positions} positions"]
        for ln in lines:
            print(ln)
        return 0
    if len(a.games_dir) != 1:
        ap.error("several --games-dir only with --goal-verdict")
    if a.anticipation:
        lines = anticipation(a.games_dir[0], a.anticipation, env, worth, params) + [f"OBS question sheet: anticipation {time.perf_counter() - t0:.1f} s"]
        for ln in lines:
            print(ln, flush=True)
        return 0
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
                                        alternatives=None if (a.no_objection or not a.alternatives) else alternatives, alt_limit=a.alternatives, follow_all=a.flagged, params=params)
    lines = format_obs(records, unreadable) + (format_flagged(records, params.frames) + format_routes(records, params) + format_counterfactual(records) if a.flagged else []) + text
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
