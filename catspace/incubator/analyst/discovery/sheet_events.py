"""PLAN 78.42, THE SHEET'S EVENTS: listeners over the fact store, and the agenda of what a change wakes. Game-free: places are opaque strings, a position is read only through a `cell(place)` function
the caller gives (the codec's reader), and no game's object is named here.

The four events of October's agenda (rust/analyst_core/src/agenda.rs, section 40 of the plan: push / found / reason / impasse) are the four things this layer does:

  PUSH      a move was made: the places it changed are the event (`Listeners.notify`). Nothing is recomputed there: the facts subscribed to a changed place are named, with why.
  FOUND     a fact was filed: it is SUBSCRIBED (`Listeners.subscribe`) by what it read, and by role (below). A finding enters the listeners the moment it exists.
  REASON    a condition came true: a waiting record that waits on a condition is woken when the test on its places is true now (it is asked again, at once); one that names what breaks it is
            KILLED when that is true. A condition is a set of places and a test on the cells of those places (`Cond`): every kind (a place vacated, a line cleared, a duty lifted, a guard
            gone, ...) is the same machinery; a new kind is a new test, not new code.
  IMPASSE   what could not be settled (a route cut by its node cap, a wake that the budget ran out before) stays on the AGENDA, ordered by chance times stake, and is reported as stale and
            not re-read (`Agenda.left`); it is never served as current.

The roles of a subscription (the order is the strength of the wake):
  break     a waiting record's `breaks` condition came true: the record is dead.
  wake      a waiting record's `waits_on` condition came true: ask the route again now.
  route     a confirmed way: a place of its route or of the units that answer changed.
  refuter   a refuted way: a place of the unit or the exchange that refuted it changed.
  reads     any other place of the read set changed (the fact is stale: asked again in the order of the agenda).
"""
from __future__ import annotations

import heapq
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

ROLES = ("break", "wake", "route", "refuter", "reads")          # strongest first
_STRENGTH = {r: i for i, r in enumerate(ROLES)}


@dataclass(frozen=True)
class Cond:
    """A condition: the places it is about and a test over the cells of those places in the position NOW. `test(cell) -> bool`, `cell(place)` the reader of the position the event came to."""
    name: str
    places: tuple
    test: Callable = field(compare=False, default=lambda cell: True)
    words: str = ""


@dataclass
class Sub:
    fid: int
    role: str
    places: frozenset
    cond: "Cond | None" = None


@dataclass
class Wake:
    fid: int
    role: str                      # the strongest role that fired
    why: str = ""                  # the condition's name or the first changed place
    priority: float = 0.0


class Listeners:
    """The inverted index: place -> the subscriptions about it. `subscribe` files one, `drop` removes every subscription of a fact, `notify(changed, cell)` is the push event."""

    def __init__(self):
        self.by_place: dict = defaultdict(set)         # place -> {(fid, index in subs[fid])}
        self.subs: dict = {}                           # fid -> [Sub]
        self.fired = defaultdict(int)                  # role -> wakes returned

    def subscribe(self, fid: int, role: str, places, cond: "Cond | None" = None) -> Sub:
        if role not in _STRENGTH:
            raise ValueError(f"unknown role {role!r}")
        places = frozenset(places) | (frozenset(cond.places) if cond is not None else frozenset())
        lst = self.subs.setdefault(fid, [])
        lst.append(Sub(fid, role, places, cond))
        for p in places:
            self.by_place[p].add((fid, len(lst) - 1))
        return lst[-1]

    def drop(self, fid: int) -> None:
        for i, s in enumerate(self.subs.pop(fid, [])):
            for p in s.places:
                self.by_place[p].discard((fid, i))

    def notify(self, changed, cell) -> list:
        """The PUSH event: `changed` the places a move changed, `cell` the reader of the new position. -> one `Wake` per fact, with its strongest role. A subscription with a condition fires
        only when its test is true now (a condition whose places changed and whose test is false does not wake; the read set's own subscription still names the fact as stale); the others fire
        when a place of theirs changed."""
        hit: dict = {}
        changed = set(changed)
        todo = set()
        for p in changed:
            todo |= self.by_place.get(p, set())
        for fid, i in todo:
            s = self.subs[fid][i]
            if s.cond is not None:
                if not s.cond.test(cell):
                    continue
                why = s.cond.name
            else:
                why = next(iter(sorted(s.places & changed)))
            cur = hit.get(fid)
            if cur is None or _STRENGTH[s.role] < _STRENGTH[cur.role]:
                hit[fid] = Wake(fid, s.role, why)
        for w in hit.values():
            self.fired[w.role] += 1
        return sorted(hit.values(), key=lambda w: (_STRENGTH[w.role], w.fid))


class Agenda:
    """What the events woke, ordered by priority (the caller's: chance times stake when it has a table of chances, else stake; ties by the strength of the role, then the older fact first). A fact
    woken twice is on the agenda once, with the strongest role and the highest priority."""

    def __init__(self):
        self._heap: list = []
        self._best: dict = {}                          # fid -> (priority, Wake) of the live entry
        self._n = 0

    def push(self, wake: Wake, priority: float) -> None:
        wake.priority = priority
        cur = self._best.get(wake.fid)
        if cur is not None:
            old = cur[1]
            if _STRENGTH[old.role] < _STRENGTH[wake.role]:
                wake = Wake(wake.fid, old.role, old.why, priority)
            priority = max(priority, cur[0])
            wake.priority = priority
        self._best[wake.fid] = (priority, wake)
        self._n += 1
        heapq.heappush(self._heap, (-priority, _STRENGTH[wake.role], wake.fid, self._n, wake))

    def discard(self, fid: int) -> None:
        self._best.pop(fid, None)

    def pop(self) -> "Wake | None":
        while self._heap:
            _p, _s, fid, _n, wake = heapq.heappop(self._heap)
            cur = self._best.get(fid)
            if cur is not None and cur[1] is wake:
                del self._best[fid]
                return wake
        return None

    def __len__(self) -> int:
        return len(self._best)

    def left(self) -> list:
        """What is still on the agenda, highest priority first: these are STALE AND NOT RE-READ."""
        return [w for _p, w in sorted(self._best.values(), key=lambda x: (-x[0], _STRENGTH[x[1].role], x[1].fid))]

    def drain(self, ask: Callable, budget: "float | None", clock=time.perf_counter) -> list:
        """Ask the agenda's entries in order while the budget (seconds of the asks' own time) is not used up; one ask begun is finished. `ask(wake)` does the work. -> the wakes asked. A budget of
        None asks them all; 0 asks none."""
        done, spent = [], 0.0
        while len(self):
            if budget is not None and spent >= budget:
                break
            w = self.pop()
            t0 = clock()
            ask(w)
            spent += clock() - t0
            done.append(w)
        return done


# ---------------------------------------------------------------------------------------------------------------------------------------------- conditions as places plus a test

def occupant(cell_value) -> "tuple | None":
    """What stands on a place in a cell of the codec's reader: (side, kind) or None when empty."""
    return None if cell_value[2] is None else (cell_value[1], cell_value[2])


def all_empty(places) -> Callable:
    return lambda cell: all(cell(p)[2] is None for p in places)


def changed_from(places, filed: dict) -> Callable:
    """True when the cell of any of `places` is not what it was when the record was filed (`filed`: place -> the cell then)."""
    return lambda cell: any(cell(p) != filed.get(p) for p in places)


def occupant_changed(places, filed: dict) -> Callable:
    return lambda cell: any(occupant(cell(p)) != occupant(filed[p]) for p in places if p in filed)


def answer_made(origin, dest, side) -> Callable:
    """The answer a record names has been played for real: the place it leaves is empty and a unit of `side` stands on the place it goes to."""
    return lambda cell: cell(origin)[2] is None and cell(dest)[1] == side and cell(dest)[2] is not None
