"""THE LEMMA GENERATOR, domain-agnostic (Kaveh 2026-09-22: "completely not hand-written, so I can use it in other
domains"; "I can't justify using my knowledge to focus on 3-piece or 4-piece stuff").

A DOMAIN supplies its rules and nothing else (see Domain): entity types, a vocabulary of relations as programs over
a state, the actions and the transition, a scalar, and a way to place entities. Everything here is generic:

  FRAME    a conjunction of relation literals over k typed slots (a small connected pattern)
  CLAIM    over a slot: "every action by this slot changes the scalar the same way" -- evaluated on every placement
           of the frame the domain can produce (exhaustive when it fits the cap, else a uniform sample, flagged)
  POLARITY the majority outcome is the claim; the minority are the EXCEPTIONS
  GUARD    the smallest conjunction of literals (state-level over the slots, action-level over the action) that
           separates the exceptions from the rest -- sound here because the placements are the complete partition
           of the frame, not a sample through an operator
  ADMIT    a lemma is kept only if its guard explains the exceptions (coverage and purity) and the frame has
           support; what the guard cannot explain is the RESIDUAL
  EXTEND   sizes grow by iterative deepening: a size-k frame with residual is extended by one slot joined through
           one relation -- frames of size k+1 are born only from failures at size k, never enumerated wholesale
  COLLAPSE symmetry (the domain's) and observational equivalence (two frames with the same label vector are one)

    python -m catspace.incubator.analyst.lemma_gen run --domain chess --max-size 3 --cap 3000
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

OUT = Path("data/derived/lemmas")


@dataclass
class Relation:
    name: str
    arity: int                     # slots it takes (state-level), or
    on_action: bool = False        # an action-level relation: fn(state, action, *slots)
    fn: Callable = None


@dataclass
class Domain:
    """What a domain must provide. No method here knows what the entities are."""
    name: str
    types: list                                                      # entity types the frames may use
    relations: list                                                  # Relation, state-level (arity 1..3) and action-level
    placements: Callable[[tuple, int, int], Iterable[Any]]           # (types, cap, seed) -> states holding exactly those entities (plus what the domain requires), with the slot entities identified
    slots_of: Callable[[Any], list]                                  # state -> the slot entities (ids) in frame order
    actions: Callable[[Any, Any], list]                              # (state, entity id) -> actions
    transition: Callable[[Any, Any], Any]                            # (state, action) -> state
    scalar: Callable[[Any, Any], float]                              # (state, entity id) -> the scalar from that entity's side
    reply_scalar: Callable[[Any, Any], float]                        # (state after our action, entity id) -> the scalar after the opponent's best immediate reply, from our side


@dataclass
class Frame:
    types: tuple                   # entity type per slot
    literals: tuple                # (relation name, slots, polarity)
    parent: "Frame | None" = None

    @property
    def size(self):
        return len(self.types)

    def text(self):
        lits = " & ".join(("" if pol else "not ") + f"{r}({','.join(str(s) for s in sl)})" for r, sl, pol in self.literals)
        return f"[{' '.join(self.types)}] " + (lits or "(no relation)")


@dataclass
class Lemma:
    frame: Frame
    slot: int                      # whose actions the claim is about
    claim: str                     # "acting loses" | "acting keeps"
    support: int                   # (placement, action) instances
    holds: int
    guard: list                    # literals over (state, action) that mark the exceptions
    covered: int                   # exceptions the guard covers
    purity: float                  # of instances matching the guard, the share that are exceptions
    residual: int                  # exceptions the guard does not cover
    complete: bool                 # the placements were exhaustive
    examples: dict = field(default_factory=dict)

    def text(self):
        g = " & ".join(("" if pol else "not ") + f"{r}({','.join(str(s) for s in sl)})" for r, sl, pol in self.guard)
        return (f"{self.frame.text()} :: slot {self.slot} {self.claim} in {100 * self.holds / max(1, self.support):.1f}% of {self.support}"
                + (f"; unless {g} (covers {self.covered}/{self.covered + self.residual} exceptions, purity {self.purity:.2f})" if self.guard else f"; {self.residual} exceptions unexplained")
                + ("" if self.complete else " [sampled]"))


class Generator:
    def __init__(self, dom: Domain, cap: int = 3000, seed: int = 0, log=print):
        self.dom = dom; self.cap = cap; self.seed = seed; self.log = log
        self.rel = {r.name: r for r in dom.relations}
        self.state_rels = [r for r in dom.relations if not r.on_action]
        self.action_rels = [r for r in dom.relations if r.on_action]
        self.library: list[Lemma] = []; self.residual_frames: list[tuple[Frame, int]] = []
        self.seen_signatures: dict = {}

    # ---------------------------------------------------------------- instances of a frame
    def instances(self, frame: Frame):
        """(state, slot ids, action, label) for every placement matching the frame's literals and every action of
        every slot; label per slot = the scalar change after the action and the opponent's best reply."""
        n = 0; complete = True
        for state, exhaustive in self.dom.placements(frame.types, self.cap, self.seed):
            complete = complete and exhaustive
            ids = self.dom.slots_of(state)
            if not all(self._lit(state, ids, r, sl, pol) for r, sl, pol in frame.literals):
                continue
            n += 1
            yield state, ids, complete

    def _lit(self, state, ids, rname, slots, pol, action=None) -> bool:
        r = self.rel[rname]
        args = [ids[s] for s in slots]
        v = bool(r.fn(state, action, *args)) if r.on_action else bool(r.fn(state, *args))
        return v == pol

    # ---------------------------------------------------------------- evaluate one frame, one slot
    def evaluate(self, frame: Frame, slot: int) -> Lemma | None:
        rows = []                                                    # (state, ids, action, delta)
        complete = True
        for state, ids, comp in self.instances(frame):
            complete = comp
            e = ids[slot]
            for a in self.dom.actions(state, e):
                before = self.dom.scalar(state, e)
                after = self.dom.transition(state, a)
                delta = self.dom.reply_scalar(after, e) - before
                rows.append((state, ids, a, delta))
        if len(rows) < 30:
            return None
        loses = [d < 0 for _, _, _, d in rows]
        n_lose = sum(loses)
        claim = "acting loses" if n_lose * 2 >= len(rows) else "acting keeps"
        exc = [not l if claim == "acting loses" else l for l in loses]      # exceptions to the majority claim
        holds = len(rows) - sum(exc)
        # the label vector's signature: observational equivalence across frames
        sig = (tuple(frame.types), slot, len(rows), holds, hash(tuple(exc)))
        if sig in self.seen_signatures:
            return None
        self.seen_signatures[sig] = frame
        guard, covered, purity, matched = self.find_guard(rows, exc, frame)
        residual = sum(exc) - covered
        examples = {}
        for (state, ids, a, d), is_exc in zip(rows, exc):
            key = "exception" if is_exc else "holds"
            if key not in examples:
                examples[key] = (self.dom_text(state), str(a), round(d, 2))
        return Lemma(frame, slot, claim, len(rows), holds, guard, covered, purity, residual, complete, examples)

    def dom_text(self, state):
        return getattr(state, "fen", lambda: str(state))()

    # ---------------------------------------------------------------- the guard: the smallest separating conjunction
    def find_guard(self, rows, exc, frame: Frame, max_lits: int = 3):
        n_exc = sum(exc)
        if n_exc == 0:
            return [], 0, 1.0, 0
        # candidate literals: every state-level relation over the slots, every action-level relation over the action (+ slots)
        k = frame.size
        cands = []
        for r in self.state_rels:
            for sl in itertools.permutations(range(k), r.arity):
                cands.append((r.name, sl))
        for r in self.action_rels:
            for sl in itertools.permutations(range(k), r.arity):
                cands.append((r.name, sl))
        # evaluate literal truth per row (cached vectors)
        vec = {}
        for rname, sl in cands:
            r = self.rel[rname]
            try:
                vec[(rname, sl)] = [bool(r.fn(state, a, *[ids[s] for s in sl])) if r.on_action else bool(r.fn(state, *[ids[s] for s in sl])) for state, ids, a, _ in rows]
            except Exception:
                continue
        chosen = []; covered_mask = [False] * len(rows)
        for _ in range(max_lits):
            best = None
            for key, v in vec.items():
                for pol in (True, False):
                    m = [(x == pol) and e and not c for x, e, c in zip(v, exc, covered_mask)]
                    gain = sum(m)
                    if gain == 0:
                        continue
                    matched = sum((x == pol) for x in v)
                    pur = sum((x == pol) and e for x, e in zip(v, exc)) / max(1, matched)
                    score = gain * pur
                    if best is None or score > best[0]:
                        best = (score, key, pol, pur, gain)
            if best is None or best[3] < 0.6:
                break
            _, key, pol, pur, gain = best
            chosen.append((key[0], key[1], pol))
            v = vec[key]
            covered_mask = [c or ((x == pol) and e) for c, x, e in zip(covered_mask, v, exc)]
        covered = sum(covered_mask)
        matched_any = [any((vec[(r_, sl)][i] == pol) for r_, sl, pol in chosen) for i in range(len(rows))] if chosen else [False] * len(rows)
        purity = sum(m and e for m, e in zip(matched_any, exc)) / max(1, sum(matched_any))
        return chosen, covered, purity, sum(matched_any)

    # ---------------------------------------------------------------- the schedule: sizes by iterative deepening, extension from residual
    def run(self, max_size: int = 3, min_support: int = 60, admit_cov: float = 0.9, admit_pur: float = 0.8):
        t0 = time.perf_counter()
        frontier = [Frame((t,), ()) for t in self.dom.types]                # size one: every entity type alone
        for size in range(1, max_size + 1):
            self.log(f"[lemma_gen] size {size}: {len(frontier)} frames to evaluate ({time.perf_counter() - t0:.0f}s)")
            next_frontier = []; admitted = 0; evaluated = 0
            for fr in frontier:
                for slot in range(fr.size):
                    lem = self.evaluate(fr, slot)
                    if lem is None:
                        continue
                    evaluated += 1
                    if lem.support < min_support:
                        continue
                    n_exc = lem.covered + lem.residual
                    clean = n_exc == 0 or (lem.covered / n_exc >= admit_cov and lem.purity >= admit_pur)
                    if clean and lem.holds / lem.support >= 0.5:
                        self.library.append(lem); admitted += 1
                        self.log(f"  ADMIT  {lem.text()}")
                    if lem.residual > 0 and size < max_size:
                        next_frontier.extend(self.extend(fr))
                        self.residual_frames.append((fr, lem.residual))
            self.log(f"[lemma_gen] size {size}: evaluated {evaluated}, admitted {admitted}; {len(next_frontier)} extensions seeded from residual")
            # collapse duplicate extensions
            uniq = {}
            for f in next_frontier:
                uniq.setdefault((f.types, f.literals), f)
            frontier = list(uniq.values())
        self.save()
        return self.library

    def extend(self, frame: Frame) -> list[Frame]:
        """One more slot, joined to an existing slot through one binary relation, either polarity of nothing else."""
        out = []
        k = frame.size
        for t in self.dom.types:
            for r in self.state_rels:
                if r.arity != 2:
                    continue
                for s in range(k):
                    for sl in ((s, k), (k, s)):
                        out.append(Frame(frame.types + (t,), frame.literals + ((r.name, sl, True),), parent=frame))
        return out

    def save(self):
        OUT.mkdir(parents=True, exist_ok=True)
        json.dump([dict(frame=l.frame.text(), slot=l.slot, claim=l.claim, support=l.support, holds=l.holds, guard=[list(map(str, g)) for g in l.guard],
                        covered=l.covered, purity=l.purity, residual=l.residual, complete=l.complete, examples=l.examples) for l in self.library],
                  open(OUT / f"library_{self.dom.name}.json", "w"), indent=1)
        json.dump([dict(frame=f.text(), residual=r) for f, r in self.residual_frames], open(OUT / f"residual_{self.dom.name}.json", "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"]); ap.add_argument("--domain", default="chess")
    ap.add_argument("--max-size", type=int, default=3); ap.add_argument("--cap", type=int, default=3000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.domain == "chess":
        from .chess_domain import domain
        dom = domain()
    else:
        raise SystemExit(f"unknown domain {a.domain}")
    g = Generator(dom, cap=a.cap, seed=a.seed)
    lib = g.run(max_size=a.max_size)
    print(f"[lemma_gen] library: {len(lib)} lemmas -> {OUT}")


if __name__ == "__main__":
    main()
