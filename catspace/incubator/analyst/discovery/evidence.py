"""PLAN 78.30 EVIDENCE ON DEMAND. A forced seed's best candidate (sleep.py) often has NO refutation but too little support: it fires on 1 to 7 cards at precision 1.0, and the gates (abstraction.Gates:
fires, games, the lower bound of the precision) say "not enough seen". Instead of dropping it, the sleep may go and LOOK: in a CORPUS of extra positions (every position of the cycle's wake logs, every
position the sleep's playouts visited) it finds those where the candidate's body binds and the rule names a move -- by the engine's matcher on cards, no rank source needed for that step -- asks the
rank source at up to a bounded number of them, spread over groups, values those positions' cards by the source's grades (judges.revalue, the sleep's own way), splits them by group into
development and held-out exactly as the sleep splits games, and hands them back as cards. The SAME gates are then applied by the sleep to the old cards plus these, unchanged. A refutation among them is a
counter-example, never hidden.

Game-free: positions, a rank source object, the variant string; no game object is named here.

  Evidence(corpus, variant=, spec=, vocab=, teacher=, is_dev=, per_candidate=, total=, cap=, known=)
     .gather(lemma) -> Gathered | None      (None when nowhere in the corpus the body binds and the rule names a move, or the bounds are spent)
  corpus: [{start, moves, group}]; `known`: keys (start, tuple(moves)) of the positions the sleep already has as cards (not asked again).
  BOUNDS (printed by the sleep): `per_candidate` positions asked per candidate, `total` new positions asked in all, `cap` corpus positions read into cards (a stride over the corpus beyond it).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import cs_engine as CE

from catspace.incubator.analyst.discovery import judges as JG


@dataclass
class Gathered:
    cards: object                   # the CardSet of the positions asked, valued by the rank source
    dev: list                       # their indices (in `cards`) on the development side
    held: list                      # ... on the held-out side
    corpus: int                     # corpus positions read
    binds: int                      # positions where the body binds and the rule names a move
    asked: int                      # positions used as evidence for this candidate
    new_asks: int                   # of them, asked of the source for the first time
    keys: list                      # (start, moves tuple) of the positions used


def key_of(p: dict) -> tuple:
    return (p.get("start"), tuple(p["moves"]))


class Evidence:
    def __init__(self, corpus: list, *, variant: str, spec: dict, vocab: str, teacher, is_dev, per_candidate: int = 60, total: int = 600, cap: int = 30000, known: "set | None" = None,
                 depth: int = 1, threads: int = 0):
        self.variant, self.spec, self.vocab, self.teacher, self.is_dev = variant, spec, vocab, teacher, is_dev
        self.per_candidate, self.total, self.cap, self.depth, self.threads = per_candidate, total, cap, depth, threads
        seen, rows = set(known or ()), []
        for p in corpus:
            k = key_of(p)
            if k not in seen:
                seen.add(k)
                rows.append(p)
        stride = max(1, -(-len(rows) // max(1, cap)))
        self.rows = rows[::stride]
        self.total_corpus = len(corpus)
        self._cards = None
        self.used: set = set()                 # positions already asked of the source (any candidate)
        self.new_asks = 0

    def cards(self):
        """The corpus as cards (depth `depth`; their values are not used), built once."""
        if self._cards is None:
            items = [(self.variant, p.get("start"), list(p["moves"]), f"e{i}", p["group"]) for i, p in enumerate(self.rows)]
            self._cards = CE.build_cards(items, self.spec, self.vocab, self.depth, 2_000_000, self.threads, {}, False)
            self._ids = [json.loads(self._cards.card(i))["id"] for i in range(len(self._cards))]
        return self._cards

    def gather(self, lemma: dict) -> "Gathered | None":
        cards = self.cards()
        ev = cards.eval(json.dumps(lemma), None, 64, None)
        hits = [ci for ci, _h, k in ev["cases"] if k > 0]
        if not hits:
            return None
        by: dict = {}
        for ci in hits:
            by.setdefault(json.loads(cards.card(ci))["group"], []).append(ci)
        # spread over groups; positions already asked cost nothing and come first within a group
        for g in by:
            by[g].sort(key=lambda ci: (key_of(self.rows[int(self._ids[ci][1:])]) not in self.used, ci))
        pick, new, queues = [], 0, list(by.values())
        while queues and len(pick) < self.per_candidate:
            nxt = []
            for q in queues:
                if len(pick) >= self.per_candidate:
                    break
                ci = q.pop(0)
                k = key_of(self.rows[int(self._ids[ci][1:])])
                if k not in self.used:
                    if self.new_asks + new >= self.total:
                        continue
                    new += 1
                pick.append(ci)
                if q:
                    nxt.append(q)
            queues = nxt
        if not pick:
            return None
        rows = [self.rows[int(self._ids[ci][1:])] for ci in pick]
        items = [(self.variant, p.get("start"), list(p["moves"]), f"x{i}", p["group"]) for i, p in enumerate(rows)]
        cs = CE.build_cards(items, self.spec, self.vocab, self.depth, 2_000_000, self.threads, {}, False)
        positions = {f"x{i}": (p.get("start"), list(p["moves"])) for i, p in enumerate(rows)}
        cs, _rep = JG.revalue(cs, positions, self.teacher, self.variant)
        groups = [json.loads(cs.card(i))["group"] for i in range(len(cs))]
        dev = [i for i, g in enumerate(groups) if self.is_dev(g)]
        held = [i for i, g in enumerate(groups) if not self.is_dev(g)]
        for p in rows:
            self.used.add(key_of(p))
        self.new_asks += new
        return Gathered(cs, dev, held, len(self.rows), len(hits), len(rows), new, [key_of(p) for p in rows])
