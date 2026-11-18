"""PLAN 78.91 (SYSTEM_SPEC 7 "subgoals enter through the prior", 5 the search's reading order, 8 the wake, 9 the score): THE HINT IN THE SEARCH'S ORDER, THE PIECES THE MEASURE AND THE PLAYER SHARE.
A sandbox: nothing is wired into any player of the wake; nothing is written to rules/learned or data/derived. Game-free: the game is named only through the pieces that exist (odds_measure's kind names and worth record,
human_heads_data, subgoal_goals' board reader, symbols_play's reader of 78.81, symbols_own_play's reader of 78.90's fit A, tiebreak_measure's match harness). No new method and no new Rust: the order is handed to the
search through the root order argument that plan 78.51 built (`root_orders`, `root_order_mode`).

THE ORDER. The search's static root order, before any reading, is: the moves that gain for the mover by the judge's gain (what a move takes, what it changes the kind of; best gain first, ties in the order the engine generates the
moves), then every other move in the order generated (rust/engine/tactics/src/ledger.rs `ordered_at` at the first deepening, no killer, no history). The judge is the Worth judge of the odds record the control plays with (no worth
is assumed here: the numbers are the record's). A QUIET move is a move with no gain. The hint's order (ii) keeps the gaining moves where the search has them and reorders only the quiet moves, by the reasons' score, highest first,
a tie of scores in the static order. The order of a short reading (iii) is the order the deepening leaves at the root: the move the reading chose first, then the other moves it held equal, then the static order.

INTERFACE (written before the bodies).
  static_order(gains) -> [index]               gains[i] the gain of legal move i. The static order above.
  hint_order(gains, scores) -> [index]         scores {index: score} for QUIET moves only (a quiet move without a score has none: it keeps its place among the unscored in the static order, after the scored). The gaining moves first as
                                               static_order has them, then the quiet moves, highest score first, equal scores (to TIE_DIGITS places) in the static order.
  short_order(gains, first, equals) -> [index] `first` the index of the move a short reading chose, `equals` the indices it held equal: first, the others of `equals` in index order, then the rest in the static order.
  place_of(order, index, among=None) -> int    1-based place of `index` in `order`; with `among` (a set of indices) the place among the moves of that set only.
  HintOrder()                                  per process (use hint_order_reader()): the readers of 78.81 and 78.90's fit A and the worth record.
      .read(moves) -> {"text", "legal" (names), "gains" (one per legal move), "quiet" (indices)}
                                               one replay of each legal move (subgoal_goals.Facts.board before and after: the units the mover took, the change of the mover's own units by kind).
      .scores(moves, read, which) -> {index: score}   which "own" (OwnReader, the fit of A) or "reasons" (ReasonsReader, 78.81's weights); the quiet moves only; fewer than two quiet moves: {}.
      .order(moves, which) -> (order, read, scores)   the hint's order (ii) for the position after `moves`.
  OrderMover(TM.Mover67)                       the match player: the SAME search call as every player of 78.67 (`solve_equals_batch`) with `root_orders=[hint order]` and `root_order_mode="ties"` (the order only breaks ties of
                                               the search's own order; the search's value, its depth and its first-move rule are the control's), the move the search returns is played (no later pick). Kinds `or_own` and `or_reasons`
                                              . `charge` (spec): the order's reading time (the running mean so far) is taken out of the search's clock, as 78.75 and 78.82 did. stats: "order_s" the seconds of the orders.
  play91(spec) -> game                         module level (the Pool spawns workers that import it). Called by symbols_order_measure `match`.
Called by: symbols_order_measure (stages orders, cost, nodes, match).
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import time
from collections import Counter

import cs_engine as CE

from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import subgoal_goals as SG
from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
from catspace.incubator.analyst.discovery import symbols_own_play as OP
from catspace.incubator.analyst.discovery import symbols_play as SP

TIE_DIGITS = SP.TIE_DIGITS
ORDER_KINDS = {"or_own": "own", "or_reasons": "reasons"}
_READER = None


# ------------------------------------------------------------------------------------------------------------------------------ the orders (pure)

def static_order(gains: list) -> list:
    gain = [i for i, g in enumerate(gains) if g != 0]
    gain.sort(key=lambda i: (-gains[i], i))
    return gain + [i for i, g in enumerate(gains) if g == 0]


def hint_order(gains: list, scores: dict) -> list:
    st = static_order(gains)
    quiet = [i for i in st if gains[i] == 0]
    scored = sorted((i for i in quiet if i in scores), key=lambda i: (-round(scores[i], TIE_DIGITS), i))
    unscored = [i for i in quiet if i not in scores]
    return [i for i in st if gains[i] != 0] + scored + unscored


def short_order(gains: list, first: int, equals: list) -> list:
    head = [first] + sorted(i for i in set(equals) if i != first)
    return head + [i for i in static_order(gains) if i not in head]


def place_of(order: list, index: int, among: "set | None" = None) -> int:
    seq = order if among is None else [i for i in order if i in among]
    return seq.index(index) + 1


# ------------------------------------------------------------------------------------------------------------------------------ the reader

class HintOrder:
    """See the module doc. One per process."""

    def __init__(self):
        base = OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")
        self.worth_engine = dict(base["worth"])
        names = OM.kernel_names()
        self.worth = {k: self.worth_engine.get(v, 0) for k, v in names.items()}
        self.facts = SG.Facts(HD.VARIANT)
        self.own = OP.own_reader()
        self.r81 = SP.reader82()

    def read(self, moves: list) -> dict:
        text, mover, legal = CE.state(HD.VARIANT, None, list(moves))[:3]
        before = self.facts.board(text)
        gains = []
        for m in legal:
            after = self.facts.board(CE.state(HD.VARIANT, text, [m])[0])
            gone, came = before - after, after - before
            taken = sum(self.worth.get(k, 0) for o, k, _p in gone if o != mover)
            lost, gained = Counter(k for o, k, _p in gone if o == mover), Counter(k for o, k, _p in came if o == mover)
            changed = sum(self.worth.get(k, 0) * (gained[k] - lost[k]) for k in set(lost) | set(gained))
            gains.append(int(taken + changed))
        return {"text": text, "legal": list(legal), "gains": gains, "quiet": [i for i, g in enumerate(gains) if g == 0]}

    def scores(self, moves: list, read: dict, which: str) -> dict:
        quiet = read["quiet"]
        if len(quiet) < 2:
            return {}
        names = [read["legal"][i] for i in quiet]
        if which == "own":
            detail = self.own.pick(list(moves), names)[1]
        else:
            detail = self.r81.pick(list(moves), names, "reasons")[1]
        return {i: float(detail["scores"][m]) for i, m in zip(quiet, names)}

    def order(self, moves: list, which: str) -> tuple:
        read = self.read(moves)
        sc = self.scores(moves, read, which)
        return hint_order(read["gains"], sc), read, sc


def hint_order_reader() -> HintOrder:
    global _READER
    if _READER is None:
        _READER = HintOrder()
    return _READER


# ------------------------------------------------------------------------------------------------------------------------------ the player

class OrderMover(TM.Mover67):
    """See the module doc."""

    def __init__(self, spec: dict, seconds: float, seed: str, trace: str):
        super().__init__(spec, seconds, seed, trace)
        self.which = ORDER_KINDS.get(self.kind, "own")
        self.stats["order_s"] = 0.0

    def search_clock(self) -> float:
        return TM.reading_clock(self.seconds, self.costs) if self.charge and self.kind in ORDER_KINDS else self.seconds

    def choose(self, text: "str | None", moves: list, legal: list) -> tuple:
        if self.kind not in ORDER_KINDS:
            return super().choose(text, moves, legal)
        t0 = time.perf_counter()
        order, _read, _sc = hint_order_reader().order(list(moves), self.which)
        t1 = time.perf_counter()
        row = PM.equal_set(CE.solve_equals_batch([(OM.VARIANT, text, list(moves))], self.search_clock(), worth=dict(self.worth), ours_first=False, valuation="odds record", threads=1,
                                                 ledger_share=TM.LEDGER_SHARE, root_orders=[order], root_order_mode="ties")[0])
        t2 = time.perf_counter()
        st = self.stats
        st["decisions"] += 1
        st["ties"] += len(row["equals"]) >= 2
        self.costs.append(t1 - t0)
        st["order_s"] += t1 - t0
        st["search"] += t2 - t1
        st["read"] += t1 - t0
        st["wall"] += t2 - t0
        st["log"].append((len(moves), len(row["equals"]), 0))
        return row["move"], row["depth"]


def play91(spec: dict) -> dict:
    OM.Mover = OrderMover
    TM.REG.clear()
    g = OM.play_game(spec)
    g["stats"] = [dict(m.stats) for m in TM.REG]
    return g
