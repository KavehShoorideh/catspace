"""NODE ACCOUNTING (plan §37, phase one): ONE counter for the work a decision costs, across every searcher.
Without it "fewer nodes" cannot be measured, and a lemma that costs more than it saves is invisible (2026-09-21:
the first lemma schema made the puzzle gauge SLOWER and only the clock showed it).

    with count() as c:
        score_moves(board, memo, engine)
    c["total"], c["ledger"], c["gs"], c["engine"], c["graph"]

The units are deliberately different kinds of work, reported separately and summed with weights the caller sets:
  ledger  -- a leaf priced by the exchange ledger (rank.leaf_value): the cheap unit
  gs      -- a node created by the goal search
  engine  -- a call that reached Stockfish (cache hits are free)
  graph   -- an interaction graph built (the expensive unit: ~1 s)
  lemma   -- a family-lemma detection pass (its own cost, so the saving can be netted against it)
  covered -- replies NOT searched because a lemma closed them (the saving, counted where it happens)
"""
from __future__ import annotations

import contextlib
import threading

KINDS = ("ledger", "gs", "engine", "graph", "lemma", "covered")
WEIGHTS = {"ledger": 1.0, "gs": 1.0, "engine": 50.0, "graph": 200.0, "lemma": 1.0, "covered": 0.0}   # "total" = work, in ledger-leaf units

_state = threading.local()


def _counters() -> dict | None:
    return getattr(_state, "counters", None)


def tick(kind: str, n: int = 1):
    c = _counters()
    if c is not None:
        c[kind] = c.get(kind, 0) + n


@contextlib.contextmanager
def count(into: dict | None = None):
    """Count the work inside the block. Nests: an inner block's counts also reach the outer one."""
    prev = _counters()
    c = {k: 0 for k in KINDS} if into is None else into
    _state.counters = c
    try:
        yield c
    finally:
        _state.counters = prev
        if prev is not None:
            for k in KINDS:
                prev[k] = prev.get(k, 0) + c.get(k, 0)
        c["total"] = total(c)


def total(c: dict) -> float:
    return float(sum(WEIGHTS[k] * c.get(k, 0) for k in KINDS))


def fmt(c: dict) -> str:
    return f"{total(c):.0f} work (ledger {c.get('ledger', 0)}, goal-search {c.get('gs', 0)}, engine {c.get('engine', 0)}, graphs {c.get('graph', 0)}, lemma passes {c.get('lemma', 0)}, replies saved {c.get('covered', 0)})"
