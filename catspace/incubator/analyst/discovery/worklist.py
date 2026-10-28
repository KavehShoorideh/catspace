"""plan 74.2 / 74.2b THE WORKLIST (Kaveh 2026-10-27: "spend your resources when searching for a move to find the cheapest move that'll maintain your state ... and then you look
for a better move from there"). ONE passive set of questions for the whole run, every depth: the given-clause loop's passive set (Otter, E), the best item processed next.

GENERIC: an item is a payload the engine hands in (the question), never read here, with what the selection reads: its STAKE, its DEPTH, the FRAME it belongs to (the line in
hand at its position: the agenda that asks it), the POLICY'S ORDER when no policy scores it (the engine's own order key, lower first), the judge's PRICE (None: not priced),
whether its answer could KEEP the status (None: no status read), the forcedness of its replies and the policy's score when known.

THE STATES: OPEN (selectable in its frame), DEFERRED (it arose outside the frame of the line in hand: queued, not asked, until the frame opens it -- the goal is dead there),
TAKEN (asked or served), OBSERVED (an observation's item: below every question, 74.4).

SELECTION (74.2b), in two clauses, in this order:
  (1) while no move is PROVED to keep the status: among the open items whose answer could keep it (its stake reaches the need, or it avoids a loss), the lowest PRICE
      CLASS (an equal price; an item not priced yet is in no lower class than a priced one), and the policy's order inside that class;
  (2) once a keeping move is proved: the policy's order as today (the judge's question -- can this change the decision, are its replies forced -- is asked of the routes).
With no status and no price (the count budget: no clock, no rate) both clauses are the policy's order: the engine's order key. 74 adds no number.

The open set of a frame is a binary heap on the order key (`heapq`): the same array, pushes and pops the engine's heap had, so an iteration over it meets the items in the
same order.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

OPEN, DEFERRED, TAKEN, OBSERVED = "open", "deferred", "taken", "observed"

# ---------------------------------------------------------------------------------------------------------------- plan 75.2 (a): THE OUTCOME LADDER (plan 59)
# The goal is the outcome: don't lose, then win. The STATUS of a position is the balance of a valuation less the threat against it, read on three rungs -- losing
# (below level), level (0), winning (at least one unit of the valuation ahead) -- and the LINE an answer must keep: one unit ahead when winning ('stay ahead'),
# level otherwise ('reach level'). An answer KEEPS the status when the balance plus its value reaches the line; at 'not losing' with time left the line RISES to
# 'stay ahead'. The unit is the valuation's (the engine passes it: this module names no valuation).
LOSING, LEVEL, WINNING = "losing", "level", "winning"


def rung(s0: int, unit: int) -> str:
    """The rung of a status s0: winning (at least one unit ahead), losing (below level), level."""
    return WINNING if s0 >= unit else LOSING if s0 < 0 else LEVEL


def ladder(balance: int, threat: int, unit: int) -> dict:
    """The status read on the outcome ladder: {balance, threat, s0 (balance less the threat), word (the rung), line (what an answer must keep)}."""
    s0 = balance - threat
    return dict(balance=balance, threat=threat, s0=s0, word=rung(s0, unit), line=unit if s0 >= unit else 0)


def kept(status: dict, value: "int | None") -> bool:
    """Does an answer worth `value` keep the status (None: no answer)?"""
    return value is not None and status["balance"] + value >= status["line"]


def raises(status: dict, value: int, unit: int) -> bool:
    """Kept at 'not losing' but not 'winning': the line may rise to 'stay ahead' (the next rung)."""
    return status["line"] == 0 and status["balance"] + value < unit


def raised(status: dict, unit: int) -> dict:
    """The status with its line risen one rung: 'stay ahead'."""
    return dict(status, line=unit, raised=True)


def losing(status: dict) -> bool:
    return status["word"] == LOSING


def line_name(status: dict) -> str:
    """The line in words: 'stay ahead' (a rung above level) or 'reach level'."""
    return "stay ahead" if status["line"] else "reach level"


@dataclass(eq=False)
class Item:
    payload: object
    frame: int
    order: tuple                                       # the policy's order when no policy scores (lower first; unique: its last field is a sequence number)
    stake: int = 0
    depth: int = 0
    node: "int | None" = None                          # the reasoning graph's node
    price: "float | None" = None                       # the judge's price (None: not priced)
    keeps: "bool | None" = None                        # could its answer keep the status (None: no status read)
    forced: "bool | None" = None                       # are its replies forced (None: not known)
    score: "float | None" = None                       # the policy's score (None: not scored)
    state: str = OPEN
    data: dict = field(default_factory=dict)


class Worklist:
    def __init__(self):
        self._open: dict = {}                          # frame -> the heap of (order..., item)
        self._deferred: dict = {}                      # frame -> [item] (queued: arose outside the frame of the line in hand)
        self._observed: dict = {}                      # frame -> [item] (74.4: observations, below every question)
        self.next_frame = 1
        self.taken = 0
        self.by_clause = {1: 0, 2: 0}                  # selections made by each clause where it narrowed the choice
        self.closed: set = set()

    def frame(self) -> int:
        """A new frame (one per agenda: the questions asked at one position for one line)."""
        f = self.next_frame
        self.next_frame += 1
        self._open[f] = []
        return f

    # ------------------------------------------------------------------------------------------------------------ the states
    def push(self, item: Item) -> Item:
        item.state = OPEN
        item.data["entry"] = item.order + (item,)
        heapq.heappush(self._open.setdefault(item.frame, []), item.data["entry"])
        return item

    def defer(self, item: Item) -> Item:
        item.state = DEFERRED
        self._deferred.setdefault(item.frame, []).append(item)
        return item

    def observe(self, item: Item) -> Item:
        item.state = OBSERVED
        self._observed.setdefault(item.frame, []).append(item)
        return item

    def deferred(self, frame: int) -> list:
        return list(self._deferred.get(frame, ()))

    def observed(self, frame: int) -> list:
        return list(self._observed.get(frame, ()))

    def activate(self, frame: int) -> list:
        """The frame's queued items are opened (the caller re-keys and pushes each): the line waits on them. -> the items, in the order they were queued."""
        out = self._deferred.pop(frame, [])
        return out

    def take(self, item: Item, pop: bool = False) -> Item:
        """The item is asked (or served): out of the open set -- popped from the heap's top (`pop`: it is the top), else removed and the heap rebuilt."""
        h = self._open[item.frame]
        if pop:
            assert h[0][-1] is item
            heapq.heappop(h)
        else:
            h.remove(item.data["entry"])
            heapq.heapify(h)
        item.state = TAKEN
        self.taken += 1
        return item

    def close(self, frame: int) -> None:
        """The frame's agenda ended: what is still open or queued stays in the run's passive set (nothing asks it yet: the ponderer drains the fact store's GAP queue, not this set)."""
        self.closed.add(frame)

    # ------------------------------------------------------------------------------------------------------------ reads
    def items(self, frame: int) -> list:
        """The open items of a frame, in the heap's array order (as the engine's heap was iterated)."""
        return [e[-1] for e in self._open.get(frame, ())]

    def head(self, frame: int) -> "Item | None":
        """The first open item by the order key (the heap's top)."""
        h = self._open.get(frame)
        return h[0][-1] if h else None

    def has_open(self, frame: int) -> bool:
        return bool(self._open.get(frame))

    def first_by(self, frame: int, key) -> "Item | None":
        """The open item that `key` (of its order tuple) ranks first."""
        h = self._open.get(frame)
        return min(h, key=lambda e: key(e[:-1]))[-1] if h else None

    def queued(self) -> list:
        """The passive set across every frame: the open and the queued items (the run's questions not asked)."""
        out = []
        for f, h in self._open.items():
            out += [e[-1] for e in h]
        for f, xs in self._deferred.items():
            out += xs
        return out

    # ------------------------------------------------------------------------------------------------------------ 74.2b: the selection
    @staticmethod
    def need(status) -> int:
        """What an answer must bring to keep the status: the line it must stay on less the balance it stands on now (the outcome ladder: don't lose, then win)."""
        return status["line"] - status["balance"]

    def candidates(self, frame: int, status=None, best=None) -> tuple:
        """74.2b's clauses as a restriction of the frame's open items. `status`: the ladder read at this frame ({balance, line}; None: no status read), `best`: the value of the best
        PROVED answer so far (None: none). A move is proved to keep the status when `best` brings the need (balance + best >= line). -> (the candidates in the heap's array order,
        the clause that narrowed them: 1, or 0 -- clause (2), or clause (1) with nothing to narrow by, is the policy's order over every open item). An item could keep the status
        when its stake reaches the need or its answer avoids a loss (`data["avoids"]`: what it saves counts against the need)."""
        its = self.items(frame)
        if status is None or not its:
            return its, 0
        need = self.need(status)
        if best is not None and best >= need:
            return its, 0                                      # (2): a keeping move is proved
        for i in its:
            i.keeps = i.stake >= need or bool(i.data.get("avoids"))
        keep = [i for i in its if i.keeps]
        if not keep:
            return its, 0
        priced = [i for i in keep if i.price is not None]
        if priced:
            low = min(i.price for i in priced)
            keep = [i for i in priced if i.price == low]
        if len(keep) == len(its):
            return its, 0
        self.by_clause[1] += 1
        return keep, 1

    def select(self, frame: int, status=None, best=None, policy=None) -> tuple:
        """THE SELECTION: the candidates of the two clauses, then the policy (`policy`: candidates -> the chosen item, or None: no policy) or the order key. -> (item, the clause
        that narrowed the candidates: 1 or 0)."""
        cands, clause = self.candidates(frame, status, best)
        if not cands:
            return None, 0
        pick = policy(cands) if policy is not None else None
        if pick is None:
            pick = self.head(frame) if not clause else min(cands, key=lambda i: i.order)
        return pick, clause
