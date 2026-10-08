"""SEARCH INSIDE A FRAME, pruned by lemmas, steered by subgoals (Kaveh 2026-10-08: "use lemmas to speed up our search ...
use plans and sub goals to guide our search in the frame of some frame; one of the frames is the exchange frame ...
there might be other frames too" -- the structure for mixing human-like planning with search).

A FRAME is a small game: states, the moves of the side to move, what playing a move does, and EXITS -- the side to
move may always stop, and the frame says what stopping is worth. Values are the side to move's (negamax), so one search
serves every frame:

    value(state) = max(exit(state), max over moves m of  gain(state, m) - value(next(state, m)))

A LEMMA is a proved bound: given a state (or a state and a move) it returns (lo, hi) on the value, or None. The search
asks the lemmas BEFORE expanding: a bound that settles the window cuts the subtree without visiting it, and the cut is
counted by the lemma's name -- so every saving is attributable and every lemma can be audited (turn it off: values must
not change). A lemma carries its status: "proved: <Lean theorem>" or "conjecture".

A PRIOR (Kaveh 2026-10-08: "a classifier is basically a prior ... experiment with different structures of how we
decide where to search and when to stop") is the unproved half: `order(state, moves)` says where to search first, and
`estimate(state)` may answer a state WITHOUT searching it (when to stop) -- an estimate, never a proof, so every
result counts how many of its leaves were estimated (`estimated`; 0 = the value is exact). Priors are swappable objects
so different structures can be compared on the same frames: nodes spent against value error.

MEMORY, RESUMPTION, RETRACTION (Kaveh 2026-10-08: "keep track of the search process itself and how far we went ... come
back and pick up where we left off"; "what decisions did we base on this lemma ... when it is invalidated, retract all
those decisions ... if we reactivate, we don't want to redo"): the search keeps every result it derives, as a record
(lo, hi, SUPPORT) per state, the support being the set of assumptions the result rests on -- the lemmas whose bounds it
used, the prior's estimates it accepted, and `open:<state>` for a node the budget stopped before finishing. A record is
VALID while its support is inside the current CONTEXT (the active lemmas and priors; `open:` tokens never are). So:
  - a budget stops the search; `resume` (more budget) re-runs from the root and every valid record answers at once --
    the work continues exactly where it stopped;
  - `retract(name)` drops an assumption from the context: every record resting on it stops answering (kept, not
    deleted), and the search re-derives only those;
  - `reinstate(name)` puts it back: the old records answer again, nothing recomputed;
  - `decisions(name)` lists the states whose results rest on it.
This is the assumption-based truth maintenance of de Kleer (ATMS, 1986) in its smallest form: labels = supports,
contexts = the active set; Doyle's JTMS (1979) is the retract-only ancestor.

THE PLAN LAYER (subgoals) uses the same hook: it orders, and it chooses among lines of EQUAL value -- the search returns
every principal line within the tie, a subgoal picks (a pawn chain kept, a file opened or kept closed); it never
changes a proved value (AGENTS rule 7: subgoals enter through the prior).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Any, Callable, Iterable, Protocol


class Frame(Protocol):
    def moves(self, state) -> Iterable[Any]: ...
    def gain(self, state, move) -> float: ...              # what the mover banks by the move itself
    def next(self, state, move) -> Any: ...                # the state after, the OTHER side to move
    def exit(self, state) -> float: ...                    # what stopping is worth to the side to move
    def key(self, state) -> Any: ...                       # for the cache


@dataclass
class Lemma:
    """A proved bound. `state`: state -> (lo, hi) | None, on the value of the state for its side to move. `move`:
    (state, move) -> (lo, hi) | None, on gain - value(next) of that move."""
    name: str
    status: str                                            # "proved: <theorem>" or "conjecture"
    state: Callable | None = None
    move: Callable | None = None


@dataclass
class Prior:
    """Where to search first and when to stop -- unproved. Both optional; the default prior searches everything in the
    frame's own order."""
    name: str = "none"
    order: Callable | None = None                          # (state, moves) -> moves in the order to try
    estimate: Callable | None = None                       # state -> value to accept without searching, or None


@dataclass
class FrameSearch:
    frame: Frame
    lemmas: list = field(default_factory=list)
    prior: Prior = field(default_factory=Prior)
    budget: int | None = None                              # nodes this search may expand in total (None: no limit)
    nodes: int = 0                                         # nodes expanded, re-entries included
    fresh: int = 0                                         # states expanded for the first time (no record at all yet)
    estimated: int = 0
    stopped: int = 0                                       # nodes the budget left open
    cuts: Counter = field(default_factory=Counter)
    table: dict = field(default_factory=dict)              # state key -> [(lo, hi, support)]
    context: set = field(default_factory=set)

    def __post_init__(self):
        if not self.context:
            self.context = {l.name for l in self.lemmas} | ({f"prior:{self.prior.name}"} if self.prior.estimate else set())

    # -- the truth maintenance --------------------------------------------------------------------------------------
    def valid(self, support) -> bool:
        return support <= self.context

    def retract(self, name: str) -> list:
        """Drop an assumption: the results resting on it stop answering (kept). -> the states affected."""
        self.context.discard(name)
        return self.decisions(name)

    def reinstate(self, name: str) -> None:
        self.context.add(name)

    def decisions(self, name: str) -> list:
        return [k for k, recs in self.table.items() if any(name in sup for _, _, sup in recs)]

    def resume(self, more: int | None = None) -> None:
        """Give the search more budget (None: unlimited); the next root/value call continues from the records."""
        self.budget = None if more is None or self.budget is None else self.budget + more

    # -- the search ---------------------------------------------------------------------------------------------------
    def _store(self, key, lo, hi, sup):
        self.table.setdefault(key, []).append((lo, hi, sup))

    def search(self, state, alpha: float = float("-inf"), beta: float = float("inf")) -> tuple:
        """-> (value for the side to move, support)."""
        key = self.frame.key(state)
        for lo, hi, sup in self.table.get(key, ()):        # a valid record that settles the window answers at once
            if self.valid(sup) and (lo == hi or lo >= beta or hi <= alpha):
                return (lo if (lo == hi or lo >= beta) else hi), sup
        if self.budget is not None and self.nodes >= self.budget:
            self.stopped += 1                              # unfinished: the exit as a lower bound, marked open
            return self.frame.exit(state), frozenset({f"open:{key}"})
        self.nodes += 1
        if key not in self.table:
            self.fresh += 1
        tag = f"prior:{self.prior.name}"
        if self.prior.estimate is not None and tag in self.context and (e := self.prior.estimate(state)) is not None:
            self.estimated += 1                            # the prior stops the search here: an estimate, not a proof
            self._store(key, e, e, frozenset({tag}))
            return e, frozenset({tag})
        a0 = alpha; sup = frozenset()
        lo, hi, hi_by = float("-inf"), float("inf"), frozenset()
        for lem in self.lemmas:                            # the lemmas first: bounds on this state
            if lem.state is not None and lem.name in self.context and (b := lem.state(state)) is not None:
                if b[0] > lo:
                    lo = b[0]
                if b[1] < hi:
                    hi, hi_by = b[1], frozenset({lem.name})
                if lo >= beta or hi <= alpha or lo == hi:
                    self.cuts[lem.name] += 1
                    v = lo if (lo >= beta or lo == hi) else hi
                    s2 = frozenset({lem.name})
                    self._store(key, v if v >= beta or lo == hi else float("-inf"), v if v <= alpha or lo == hi else float("inf"), s2)
                    return v, s2
        best = self.frame.exit(state)
        if best < beta:
            alpha = max(alpha, best)
            ms = list(self.frame.moves(state))
            if self.prior.order is not None:
                ms = self.prior.order(state, ms)
            for m in ms:
                g = self.frame.gain(state, m)
                skip = None
                for lem in self.lemmas:                    # bounds on this move before searching it
                    if lem.move is not None and lem.name in self.context and (b := lem.move(state, m)) is not None and b[1] <= alpha:
                        skip = lem.name; break
                if skip:
                    self.cuts[skip] += 1; sup |= {skip}
                    continue
                cv, csup = self.search(self.frame.next(state, m), g - beta, g - alpha)
                v = g - cv; sup |= csup
                best = max(best, v); alpha = max(alpha, v)
                if alpha >= beta:
                    break
                if alpha >= hi:                            # nothing can beat the lemma's ceiling: stop here
                    sup |= hi_by
                    break
        rlo = best if best > a0 else float("-inf")         # fail-soft bounds of this result
        rhi = best if best < beta else float("inf")
        self._store(key, rlo, rhi, sup)
        return best, sup

    def value(self, state, alpha: float = float("-inf"), beta: float = float("inf")) -> float:
        return self.search(state, alpha, beta)[0]

    def root(self, state) -> dict:
        """{move: value of playing it} for every move at the root (no window: ties all visible to the guide);
        `self.support` is what the root's values rest on (an `open:` element: not finished)."""
        out, self.support = {}, frozenset()
        for m in self.frame.moves(state):
            v, sup = self.search(self.frame.next(state, m))
            out[m] = self.frame.gain(state, m) - v; self.support |= sup
        return out

    def finished(self) -> bool:
        return not any(x.startswith("open:") for x in getattr(self, "support", ()))

    def lines(self, state, prefix=()) -> list:
        """Every principal line (moves, with 'exit' where a side stops) reaching the state's value: for the guide."""
        best = self.value(state)
        out = []
        if self.frame.exit(state) == best:
            out.append(prefix + ("exit",))
        for m in self.frame.moves(state):
            g = self.frame.gain(state, m)
            if g - self.value(self.frame.next(state, m)) == best:
                out.extend(self.lines(self.frame.next(state, m), prefix + (m,)))
        return out
