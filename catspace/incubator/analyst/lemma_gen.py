"""THE LEMMA GENERATOR, domain-agnostic (Kaveh 2026-09-22: "completely not hand-written, so I can use it in other
domains"; "I can't justify using my knowledge to focus on 3-piece or 4-piece stuff"; "how do we combine them
logically?").

A DOMAIN supplies its rules and nothing else (see Domain): entity types, a vocabulary of relations as programs over
a state, the actions and the transition, a scalar, and a way to place entities. Everything here is generic:

  FRAME    a conjunction of relation literals over k typed slots (a small connected pattern)
  CLAIM    over a slot: "every action by this slot changes the scalar the same way" -- evaluated on every placement
           of the frame the domain can produce (exhaustive when it fits the cap, else a uniform sample, flagged)
  POLARITY the majority outcome is the claim; the minority are the EXCEPTIONS
  GUARD    a DISJUNCTION of TERMS, each term a small conjunction of literals (state-level over the slots, action-level
           over the action), that separates the exceptions from the rest -- disjunctive normal form over the
           relation vocabulary; sound here because the placements are the complete partition of the frame
  SUBSUME  a frame inherits the lemmas of ALL its sub-frames (every embedding of an admitted smaller frame into it
           with the same actor); their guards are applied first, and only the exceptions none of them explains are
           this frame's business ("unless any other piece is attacked" is the union of the two-piece lemmas)
  ADMIT    a lemma is kept only if it has something new (exceptions its sub-frames do not explain) and its own guard
           explains them (coverage and purity); what the guard cannot explain is the RESIDUAL
  EXTEND   sizes grow by iterative deepening: a frame whose sub-frames' lemmas broke is extended by one slot joined
           through one relation -- frames of size k+1 are born only from failures at size k
  COLLAPSE observational equivalence (two frames with the same label vector are one)
  PROVE    the exhaustive pass re-checks each admitted lemma's STORED guard at a far larger cap
  GENERALIZE anti-unification over types: lemmas identical up to their slot types become one SCHEMA with a type set
           per slot; the unseen instantiations (the product of the sets minus the observed tuples) are checked by
           the prover, so a schema is stated exactly over what held

    python -m catspace.incubator.analyst.lemma_gen run --domain chess --max-size 3 --cap 1000 --workers 10
    python -m catspace.incubator.analyst.lemma_gen prove --cap 10000 --workers 10
    python -m catspace.incubator.analyst.lemma_gen generalize --cap 10000 --workers 10
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
    entities: Callable[[Any], list] = None                           # raw state -> [(entity id, type)] (the applier: real states, not placements)
    state_of: Callable[[Any, list], Any] = None                      # (raw state, slot ids) -> State, so relations can be evaluated on a real state
    actor_ok: Callable[[Any, Any], bool] = None                      # (raw state, entity id) -> may this entity act now (e.g. its side is to move)
    null_reply_scalar: Callable[[Any, Any], float] = None            # (state, entity id) -> the scalar after the opponent's best reply if the entity's side PASSES (None: no null action); the RELATIVE outcome baseline


# a literal is (relation name, slots, polarity); a TERM is a tuple of literals (conjunction); a GUARD is a list of terms (disjunction)
def lit_text(r, sl, pol):
    return ("" if pol else "not ") + f"{r}({','.join(str(s) for s in sl)})"


def term_text(term):
    return " & ".join(lit_text(r, sl, pol) for r, sl, pol in term)


def guard_text(guard):
    return " or ".join(term_text(t) if len(t) == 1 else "(" + term_text(t) + ")" for t in guard)


def lemma_id(types, literals, slot, claim, full_guard) -> str:
    """A stable id from the lemma's CONTENT (frame types and literals, the acting slot, the claim, the full guard):
    the same lemma has the same id in every run, file and log, so it can be cited ("L-3f9a1c2d")."""
    import hashlib
    key = json.dumps([list(types), [[r, list(sl), pol] for r, sl, pol in sorted(literals)], slot, claim,
                      sorted([[r, list(sl), pol] for r, sl, pol in t] for t in full_guard)], sort_keys=True)
    return "L-" + hashlib.sha1(key.encode()).hexdigest()[:8]


def remap_guard(guard, m):
    return [tuple((r, tuple(m[s] for s in sl), pol) for r, sl, pol in t) for t in guard]


@dataclass
class Frame:
    types: tuple                   # entity type per slot
    literals: tuple                # (relation name, slots, polarity)
    parent: "Frame | None" = None

    @property
    def size(self):
        return len(self.types)

    def text(self):
        lits = " & ".join(lit_text(r, sl, pol) for r, sl, pol in self.literals)
        return f"[{' '.join(self.types)}] " + (lits or "(no relation)")


@dataclass
class Lemma:
    frame: Frame
    slot: int                      # whose actions the claim is about
    claim: str                     # "acting loses" | "acting keeps"
    support: int                   # (placement, action) instances
    holds: int
    guard: list                    # this frame's OWN guard: terms marking the exceptions its sub-frames left
    covered: int                   # exceptions the own guard covers
    purity: float                  # of instances matching the own guard, the share that are exceptions
    residual: int                  # exceptions no guard covers
    complete: bool                 # the placements were exhaustive
    examples: dict = field(default_factory=dict)
    inherited: list = field(default_factory=list)      # the sub-frames' guard terms (remapped), applied first
    broke: int = 0                                     # exceptions the inherited guards did NOT cover: what this frame exists to explain
    reversed: bool = False                             # a sub-frame's claim for this actor is the opposite of the majority here
    parents: list = field(default_factory=list)        # texts of the sub-frame lemmas inherited from
    sig: tuple = ()
    closure: list = field(default_factory=list)        # literals true on every placement of the frame (implied facts)

    def full_guard(self):
        return list(self.inherited) + list(self.guard)

    @property
    def id(self) -> str:
        return lemma_id(self.frame.types, self.frame.literals, self.slot, self.claim, self.full_guard())

    def text(self):
        g = guard_text(self.guard)
        cl = (" [" + "; ".join(f"{tag} {lit_text(r, sl, pol)}" for r, sl, pol, tag in self.closure[:6]) + "]") if self.closure else ""
        return (f"{self.id} {self.frame.text()}{cl} :: slot {self.slot} {self.claim} in {100 * self.holds / max(1, self.support):.1f}% of {self.support}"
                + (f"; unless {g} (covers {self.covered}/{self.covered + self.residual} exceptions, purity {self.purity:.2f})" if self.guard else f"; {self.residual} exceptions unexplained")
                + (f"; inherited [{guard_text(self.inherited)}] {'REVERSED' if self.reversed else 'broke'} on {self.broke}" if self.parents else "") + ("" if self.complete else " [sampled]"))


def _popcount(x: int) -> int:
    return x.bit_count()


def pick_examples(g, rows, exc: int, rest: int, guard: list, cache: dict, all_mask: int) -> dict:
    """The stored examples: 'holds' = a row where the claim held; 'exception' = a row the lemma's OWN guard explains
    (an exception the inherited guards left, covered by this frame's guard -- what is NEW here; Kaveh 2026-09-23:
    "show the new lemmas, not the inherited ones, under exception"); 'inherited_exception' = an exception a parent's
    guard already covered, for contrast."""
    own = 0
    for term in guard:
        own |= g.term_mask(rows, term, cache, all_mask)
    term_masks = [(term_text(t), g.term_mask(rows, t, cache, all_mask)) for t in guard]
    best = {}                                                                  # category -> (|outcome|, row): the most vivid row of each kind (Kaveh: "the one where the queen has the most advantage")
    for i, (state, ids, a, d) in enumerate(rows):
        bit = 1 << i
        if not (exc & bit):
            cat = "holds"
        elif (rest & bit) and (own & bit):
            cat = "exception"
        elif not (rest & bit):
            cat = "inherited_exception"
        else:
            cat = "unexplained"
        if cat not in best or abs(d) > best[cat][0]:
            fired = [t for t, m in term_masks if m & bit]
            best[cat] = (abs(d), (g.dom_text(state), str(a), round(d, 2), " | ".join(fired)))
    ex = {k: v[1] for k, v in best.items() if k != "unexplained"}
    if "exception" not in ex:                                                 # no own-guard exception (guard-less or inherited-only): the most vivid exception of any kind
        for k in ("unexplained", "inherited_exception"):
            if k in best:
                ex["exception"] = best[k][1]; break
    return ex


class Generator:
    def __init__(self, dom: Domain, cap: int = 3000, seed: int = 0, log=print, workers: int = 1, max_conj: int = 2, term_purity: float = 0.8, relative: bool = False, max_frames: int = 0):
        self.dom = dom; self.cap = cap; self.seed = seed; self.log = log; self.workers = workers; self.max_conj = max_conj; self.term_purity = term_purity
        self.relative = relative                                             # outcome = reply scalar minus the NULL-action reply scalar (what the standing threats already take)
        self.max_frames = max_frames                                         # per size, evaluate only the most frequent frames (by instance count in the placements source)
        self.rel = {r.name: r for r in dom.relations}
        self.state_rels = [r for r in dom.relations if not r.on_action]
        self.action_rels = [r for r in dom.relations if r.on_action]
        self.library: list[Lemma] = []; self.residual_frames: list[tuple[Frame, int]] = []
        self.seen_signatures: dict = {}

    # ---------------------------------------------------------------- instances of a frame
    def instances(self, frame: Frame):
        """(state, slot ids, complete) for every placement matching the frame's literals."""
        complete = True
        for state, exhaustive in self.dom.placements(frame.types, self.cap, self.seed):
            complete = complete and exhaustive
            ids = self.dom.slots_of(state)
            if not all(self._lit(state, ids, r, sl, pol) for r, sl, pol in frame.literals):
                continue
            yield state, ids, complete

    def _lit(self, state, ids, rname, slots, pol, action=None) -> bool:
        r = self.rel[rname]
        args = [ids[s] for s in slots]
        v = bool(r.fn(state, action, *args)) if r.on_action else bool(r.fn(state, *args))
        return v == pol

    def rows_of(self, frame: Frame, slot: int):
        rows = []; complete = True
        for state, ids, comp in self.instances(frame):
            complete = comp
            e = ids[slot]
            base = None
            if self.relative and self.dom.null_reply_scalar is not None:
                base = self.dom.null_reply_scalar(state, e)
                if base is None:
                    continue                                                  # no null action here (in check): the relative outcome is undefined
            for a in self.dom.actions(state, e):
                before = self.dom.scalar(state, e)
                after = self.dom.transition(state, a)
                delta = self.dom.reply_scalar(after, e) - (base if base is not None else before)
                rows.append((state, ids, a, delta))
        return rows, complete

    # ---------------------------------------------------------------- literal masks (bitsets over the rows)
    def lit_mask(self, rows, rname, sl, cache: dict) -> int:
        key = (rname, sl)
        if key in cache:
            return cache[key]
        r = self.rel[rname]
        m = 0
        for i, (state, ids, a, _) in enumerate(rows):
            try:
                v = bool(r.fn(state, a, *[ids[s] for s in sl])) if r.on_action else bool(r.fn(state, *[ids[s] for s in sl]))
            except Exception:
                v = False
            if v:
                m |= 1 << i
        cache[key] = m
        return m

    def term_mask(self, rows, term, cache: dict, all_mask: int) -> int:
        m = all_mask
        for rname, sl, pol in term:
            lm = self.lit_mask(rows, rname, sl, cache)
            m &= lm if pol else (all_mask & ~lm)
        return m

    def candidate_literals(self, k: int):
        cands = []
        for r in self.state_rels:
            for sl in itertools.permutations(range(k), r.arity):
                cands.append((r.name, sl))
        for r in self.action_rels:
            for sl in itertools.permutations(range(k), r.arity):
                cands.append((r.name, sl))
        return cands

    # ---------------------------------------------------------------- the guard: greedy disjunction of small conjunctions
    def find_guard(self, rows, exc: int, frame: Frame, cache: dict, max_terms: int = 3, exc_all: int | None = None):
        """exc: bitmask of the exceptions to explain (what the inherited guards left); exc_all: every exception, for
        purity -- a matched row that is an exception the parents already explained is not a false positive.
        Returns (guard terms, covered, purity, matched)."""
        n = len(rows); all_mask = (1 << n) - 1
        exc_all = exc if exc_all is None else exc_all
        n_exc = _popcount(exc)
        if n_exc == 0:
            return [], 0, 1.0, 0
        half = n // 2
        lits = []
        for rname, sl in self.candidate_literals(frame.size):
            lm = self.lit_mask(rows, rname, sl, cache)
            for pol, m in ((True, lm), (False, all_mask & ~lm)):
                if m and m != all_mask:
                    lits.append(((rname, sl, pol), m))
        terms = [((l,), m) for l, m in lits if _popcount(m) <= half]        # a term true of most rows explains nothing
        if self.max_conj >= 2:
            for (l1, m1), (l2, m2) in itertools.combinations(lits, 2):
                if l1[0] == l2[0] and l1[1] == l2[1]:
                    continue                                                  # a literal and its negation
                m = m1 & m2
                if m and _popcount(m) <= half and m != m1 and m != m2:
                    terms.append(((l1, l2), m))
        chosen = []; covered = 0
        for _ in range(max_terms):
            best = None
            for term, m in terms:
                gain = _popcount(m & exc & ~covered)
                if gain == 0:
                    continue
                pur = _popcount(m & exc_all) / _popcount(m)
                if pur < self.term_purity:
                    continue                                                  # a term is an explanation only if it is mostly exceptions
                key = (gain, -len(term), pur)                                 # most new exceptions; Occam on ties; then purity
                if best is None or key > best[0]:
                    best = (key, term, m, pur)
            if best is None:
                break
            chosen.append(best[1]); covered |= best[2] & exc
        matched_any = 0
        for term in chosen:
            matched_any |= self.term_mask(rows, term, cache, all_mask)
        cov = _popcount(matched_any & exc)
        purity = _popcount(matched_any & exc_all) / max(1, _popcount(matched_any))
        return chosen, cov, purity, _popcount(matched_any)

    # ---------------------------------------------------------------- evaluate one frame, one slot, given the sub-frames' lemmas
    def evaluate(self, frame: Frame, slot: int, inherits: list | None = None) -> Lemma | None:
        """inherits: [(claim, guard terms remapped into this frame's slots, text, literals to verify)] from every embedded sub-frame
        with the same actor. The frame's claim is its own majority; the inherited guards of the same polarity are
        applied first, and only what they leave is explained here."""
        inherits = inherits or []
        rows, complete = self.rows_of(frame, slot)
        if len(rows) < 30:
            return None
        n = len(rows); all_mask = (1 << n) - 1
        loses = 0
        for i, (_, _, _, d) in enumerate(rows):
            if d < 0:
                loses |= 1 << i
        n_lose = _popcount(loses)
        claim = "acting loses" if n_lose * 2 >= n else "acting keeps"
        exc = (all_mask & ~loses) if claim == "acting loses" else loses
        holds = n - _popcount(exc)
        cache: dict = {}
        inherited = []; parents = []; reversed_ = False
        for pclaim, pguard, ptext, missing in inherits:
            if any(self.term_mask(rows, ((r, sl, pol),), cache, all_mask) != all_mask for r, sl, pol in missing):
                continue                                                      # the sub-frame does not embed here: one of its literals fails on some placement
            if pclaim != claim:
                reversed_ = True                                              # the sub-frame's claim for this actor is the opposite of what holds here
                parents.append(ptext + " [opposite claim]")
                continue
            inherited.extend(t for t in pguard if t not in inherited); parents.append(ptext)
        covered_inh = 0
        for term in inherited:
            covered_inh |= self.term_mask(rows, term, cache, all_mask) & exc
        broke = _popcount(exc & ~covered_inh)
        rest = exc & ~covered_inh
        sig = (tuple(frame.types), slot, n, holds, hash(exc))
        guard, covered, purity, matched = self.find_guard(rows, rest, frame, cache, exc_all=exc)
        residual = _popcount(rest) - covered
        examples = pick_examples(self, rows, exc, rest, guard, cache, all_mask)
        closure = self.frame_closure(rows, frame, cache, all_mask)
        lem = Lemma(frame, slot, claim, n, holds, guard, covered, purity, residual, complete, examples, inherited, broke, reversed_, parents)
        lem.closure = closure
        lem.sig = sig
        return lem

    def frame_closure(self, rows, frame: Frame, cache: dict, all_mask: int) -> list:
        """The state-level literals true on EVERY placement of the frame that are not among its stated literals: the
        facts the frame implies (Kaveh 2026-09-23: "the black queen attacks the white bishop, which isn't stated" --
        a bishop attacking a queen along a diagonal is attacked back). Generic: any relation, either polarity."""
        stated = set(frame.literals)
        facts = []
        for rname, sl in self.candidate_literals(frame.size):
            if self.rel[rname].on_action or rname in self.type_determined():
                continue                                                      # a relation fixed by the slot types alone says nothing about the frame
            m = self.lit_mask(rows, rname, sl, cache)
            if m == all_mask and (rname, sl, True) not in stated:            # positive facts only: "never between" at a small cap is an absence, not a fact
                facts.append((rname, sl, True))
        # MINIMAL, GIVEN THE TYPES (Kaveh 2026-09-23: "all five are the same thing"): implications between literals are
        # learned on the bare frame of these types (no literals) -- a bishop attacking a queen along a diagonal is
        # attacked back, for THESE types. A fact equivalent to a stated literal is listed as a restatement ("i.e.");
        # a fact merely implied by the stated literals or by a kept fact is dropped; the rest are genuinely new ("also").
        imp = self.implications_given_types(frame.types)
        def implies(a, b):
            return (a[0], a[1], b[0], b[1]) in imp
        stated_l = list(stated)
        restated = [f for f in facts if any(implies(sl_, f) and implies(f, sl_) for sl_ in stated_l)]
        rest = [f for f in facts if f not in restated]
        # every fact here follows from the stated literals (it holds on all rows); show the ones that SUMMARIZE the
        # others: maximal under implication among the consequences, and implying at least one other consequence
        # ("the black queen attacks the bishop", which carries "aligned" and "attacked" with it); leaves implied
        # straight from a stated literal ("the black queen is attacked") are the same thing said smaller
        downstream = {f: sum(1 for g in rest if g != f and implies(f, g)) for f in rest}
        kept = [f for f in rest if downstream[f] >= 1 and not any(g != f and implies(g, f) and not implies(f, g) for g in rest)]
        # among mutually equivalent kept facts keep one
        out = []
        for f in sorted(kept, key=lambda f: -downstream[f]):
            if not any(implies(f, g) and implies(g, f) for g in out):
                out.append(f)
        return [(r, sl, pol, "i.e.") for r, sl, pol in restated] + [(r, sl, pol, "also") for r, sl, pol in out]

    def implications_given_types(self, types: tuple) -> set:
        """(A_rel, A_slots, B_rel, B_slots): A implies B on every sampled placement of the BARE frame of these types
        (A true on at least 15 rows, never with B false). Cached per type tuple."""
        cache_ = getattr(self, "_imp_t", {})
        if types in cache_:
            return cache_[types]
        k = len(types)
        lits = [(r.name, sl) for r in self.state_rels if r.name not in self.type_determined() for sl in itertools.permutations(range(k), r.arity)]
        cap = self.cap; self.cap = 300
        try:
            rows = [(state, self.dom.slots_of(state)) for state, _ in self.dom.placements(types, 300, 7)]
        finally:
            self.cap = cap
        vals = {}
        for rname, sl in lits:
            r = self.rel[rname]
            v = []
            for state, ids in rows:
                try:
                    v.append(bool(r.fn(state, *[ids[s] for s in sl])))
                except Exception:
                    v.append(False)
            vals[(rname, sl)] = v
        imp = set()
        for a, va in vals.items():
            if sum(va) < 15:
                continue
            for b, vb in vals.items():
                if a != b and not any(x and not y for x, y in zip(va, vb)):
                    imp.add((a[0], a[1], b[0], b[1]))
        cache_[types] = imp; self._imp_t = cache_
        return imp

    def implications(self, k: int) -> set:
        """(A_rel, A_slots, B_rel, B_slots) such that A implies B on every sampled placement of k random typed slots
        (A true on at least 25 rows, never with B false). Learned from the domain's placements, cached per size."""
        cache_ = getattr(self, "_imp", {})
        if k in cache_:
            return cache_[k]
        import random as _r
        rng = _r.Random(k)
        lits = [(r.name, sl) for r in self.state_rels if r.name not in self.type_determined() for sl in itertools.permutations(range(k), r.arity)]
        rows = []
        cap = self.cap; self.cap = 40
        try:
            for _ in range(24):
                types = tuple(rng.choice(self.dom.types) for _ in range(k))
                for state, _ in self.dom.placements(types, 40, rng.randrange(10 ** 6)):
                    rows.append((state, self.dom.slots_of(state)))
        finally:
            self.cap = cap
        vals = {}
        for rname, sl in lits:
            r = self.rel[rname]
            v = []
            for state, ids in rows:
                try:
                    v.append(bool(r.fn(state, *[ids[s] for s in sl])))
                except Exception:
                    v.append(False)
            vals[(rname, sl)] = v
        imp = set()
        for a, va in vals.items():
            if sum(va) < 25:
                continue
            for b, vb in vals.items():
                if a == b:
                    continue
                if not any(x and not y for x, y in zip(va, vb)):
                    imp.add((a[0], a[1], b[0], b[1]))
        cache_[k] = imp; self._imp = cache_
        return imp

    def type_determined(self) -> set:
        """Relations whose value depends on the slot types alone (in chess: same side, worth, slides), found generically:
        sampled placements of a few type pairs, a relation is type-determined if its value never varies within a type
        pair. Computed once per generator."""
        if getattr(self, "_type_det", None) is not None:
            return self._type_det
        det = set()
        rels = [r for r in self.dom.relations if not r.on_action and r.arity <= 2]
        cap = self.cap; self.cap = 60
        try:
            samples = {}
            for t1 in self.dom.types:
                for t2 in self.dom.types:                                     # every ordered type pair (same side and opposite): a relation constant within each pair is type-determined
                    for state, _ in self.dom.placements((t1, t2), 30, 0):
                        ids = self.dom.slots_of(state)
                        for r in rels:
                            args = [ids[0], ids[1]] if r.arity == 2 else [ids[0]]
                            try:
                                v = bool(r.fn(state, *args))
                            except Exception:
                                continue
                            samples.setdefault((r.name, t1, t2), set()).add(v)
            for r in rels:
                vals = [v for (n, _, _), v in samples.items() if n == r.name]
                if vals and all(len(v) == 1 for v in vals):
                    det.add(r.name)
        finally:
            self.cap = cap
        self._type_det = det
        return det

    def dom_text(self, state):
        return getattr(state, "fen", lambda: str(state))()

    # ---------------------------------------------------------------- subsumption: every admitted lemma that embeds into the frame with the same actor
    def applicable(self, frame: Frame, slot: int, by_actor_type: dict):
        out = []
        for lem in by_actor_type.get(frame.types[slot], []):
            lf = lem.frame
            if lf.size >= frame.size:
                continue
            for m in itertools.permutations(range(frame.size), lf.size):
                if m[lem.slot] != slot or any(frame.types[m[i]] != lf.types[i] for i in range(lf.size)):
                    continue
                # a sub-frame literal absent from this frame's literal set may still be IMPLIED by it (e.g. by the slot
                # types); such literals are checked on the frame's placements in evaluate(), and the embedding holds
                # only if they are true of every row
                missing = tuple((r, tuple(m[s] for s in sl), pol) for r, sl, pol in lf.literals if (r, tuple(m[s] for s in sl), pol) not in frame.literals)
                out.append((lem.claim, remap_guard(lem.full_guard(), m), lem.frame.text() + f" slot {lem.slot} as {m}", missing))
        return out

    # ---------------------------------------------------------------- the schedule
    def run(self, max_size: int = 3, min_support: int = 60, admit_cov: float = 0.9, admit_pur: float = 0.8):
        t0 = time.perf_counter()
        frontier = [Frame((t,), ()) for t in self.dom.types]
        by_actor: dict = defaultdict(list)
        for size in range(1, max_size + 1):
            self.log(f"[lemma_gen] size {size}: {len(frontier)} frames to evaluate ({time.perf_counter() - t0:.0f}s)")
            next_frontier = []; admitted = 0; evaluated = 0; broke_frames = 0
            for fi, (fr, results) in enumerate(self._evaluate_frontier(frontier, by_actor)):
                if fi and fi % 50 == 0:
                    self.log(f"[lemma_gen] size {size}: {fi}/{len(frontier)} frames, admitted {admitted} so far ({time.perf_counter() - t0:.0f}s)")
                extend_this = False
                for slot, inh, lem in results:
                    if lem is None:
                        continue
                    if lem.sig in self.seen_signatures:
                        continue
                    self.seen_signatures[lem.sig] = fr
                    evaluated += 1
                    if lem.support < min_support:
                        continue
                    if inh and lem.broke == 0:
                        continue                                              # the sub-frames' lemmas hold here unchanged: nothing new
                    n_exc = lem.covered + lem.residual
                    clean = n_exc == 0 or (lem.covered / n_exc >= admit_cov and lem.purity >= admit_pur)
                    if clean and (lem.holds / lem.support >= 0.5 or inh):
                        self.library.append(lem); by_actor[fr.types[slot]].append(lem); admitted += 1
                        self.log(f"  ADMIT  {lem.text()}")
                    else:
                        self.residual_frames.append((fr, lem.residual))
                    if not inh or lem.broke > 0:
                        extend_this = True
                if size < max_size and extend_this:
                    broke_frames += 1
                    next_frontier.extend(self.extend(fr))
            self.log(f"[lemma_gen] size {size}: evaluated {evaluated}, admitted {admitted}; {broke_frames} frames extended -> {len(next_frontier)} candidates at size {size + 1}")
            uniq = {}
            for f in next_frontier:
                uniq.setdefault((f.types, f.literals), f)
            frontier = list(uniq.values())
            if self.max_frames and len(frontier) > self.max_frames:
                # prioritize what we run into: count each frame's instances in the placements source and keep the most frequent
                t1 = time.perf_counter()
                counts = self._count_frontier(frontier)
                frontier = [f for f, _ in sorted(zip(frontier, counts), key=lambda fc: -fc[1])[: self.max_frames]]
                self.log(f"[lemma_gen] size {size + 1}: frontier bounded to the {self.max_frames} most frequent frames (counts {counts and max(counts)}..{sorted(counts)[-self.max_frames] if len(counts) >= self.max_frames else 0}; {time.perf_counter() - t1:.0f}s)")
        self.save()
        return self.library

    def count_instances(self, frame: Frame, scan_limit: int = 2000) -> int:
        """Instances in a bounded scan of the placements source (a source that honours `scan_limit`: the games source
        scans its first 2,000 positions), up to ten times the cap: frequency ranks frames beyond the evaluation cap."""
        cap = self.cap; self.cap = cap * 10
        had = getattr(self.dom.placements, "scan_limit", None)
        try:
            try:
                self.dom.placements.scan_limit = scan_limit
            except AttributeError:
                pass
            return sum(1 for _ in self.instances(frame))
        finally:
            self.cap = cap
            try:
                self.dom.placements.scan_limit = had
            except AttributeError:
                pass

    def _count_frontier(self, frontier):
        if self.workers <= 1 or len(frontier) < 4:
            return [self.count_instances(f) for f in frontier]
        import multiprocessing as mp
        global _WORKER_GEN
        _WORKER_GEN = self
        with mp.get_context("fork").Pool(self.workers) as pool:
            return list(pool.imap(_count_job, frontier, chunksize=8))

    def _evaluate_frontier(self, frontier, by_actor: dict):
        """Yield (frame, [(slot, inherits, lemma)]) in frontier order; frames are independent given the library of the
        earlier sizes, so they run in a fork pool when workers > 1 (the caller collapses and admits, in order)."""
        jobs = [(fr, [(slot, self.applicable(fr, slot, by_actor)) for slot in range(fr.size)]) for fr in frontier]
        if self.workers <= 1 or len(jobs) < 4:
            for fr, slots in jobs:
                yield fr, [(slot, inh, self.evaluate(fr, slot, inh)) for slot, inh in slots]
            return
        import multiprocessing as mp
        global _WORKER_GEN
        _WORKER_GEN = self
        ctx = mp.get_context("fork")
        with ctx.Pool(self.workers) as pool:
            for (fr, slots), res in zip(jobs, pool.imap(_eval_job, jobs, chunksize=1)):
                yield fr, [(slot, inh, lem) for (slot, inh), lem in zip(slots, res)]

    def extend(self, frame: Frame) -> list[Frame]:
        """One more slot, joined to an existing slot through one binary relation."""
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
        json.dump([lemma_to_json(l) for l in self.library], open(OUT / f"library_{self.dom.name}.json", "w"), indent=1)
        json.dump([dict(frame=f.text(), residual=r) for f, r in self.residual_frames], open(OUT / f"residual_{self.dom.name}.json", "w"), indent=1)


def lemma_to_json(l: Lemma) -> dict:
    return dict(id=l.id, closure=[[r, list(sl), pol, tag] for r, sl, pol, tag in l.closure], frame=l.frame.text(), types=list(l.frame.types), literals=[[r, list(sl), pol] for r, sl, pol in l.frame.literals], slot=l.slot, claim=l.claim,
                support=l.support, holds=l.holds, guard=[[[r, list(sl), pol] for r, sl, pol in t] for t in l.guard],
                inherited=[[[r, list(sl), pol] for r, sl, pol in t] for t in l.inherited], parents=l.parents, reversed=l.reversed, broke=l.broke,
                covered=l.covered, purity=l.purity, residual=l.residual, complete=l.complete, examples=l.examples)


def guard_from_json(g) -> list:
    """Terms; also reads the older flat format (a list of literals = single-literal terms) and the oldest string slots."""
    out = []
    for t in g or []:
        if t and isinstance(t[0], str):                                       # old: a literal [r, sl, pol]
            t = [t]
        term = []
        for r, sl, pol in t:
            sl = tuple(int(x) for x in (sl if isinstance(sl, list) else str(sl).strip("()").split(",")) if str(x).strip() != "")
            term.append((r, sl, (pol == "True") if isinstance(pol, str) else bool(pol)))
        out.append(tuple(term))
    return out


_WORKER_GEN = None


def _count_job(frame):
    return _WORKER_GEN.count_instances(frame)


def _eval_job(job):
    fr, slots = job
    g = _WORKER_GEN
    g.log = lambda *a, **k: None
    return [g.evaluate(fr, slot, inh) for slot, inh in slots]


def parse_frame(text: str) -> Frame:
    """'[wQ wR bK] same_side(0,1) & attacks(2,1)' -> Frame (for libraries saved before the structure was stored)."""
    head, _, rest = text.partition("] ")
    types = tuple(head.strip("[").split())
    lits = []
    if rest.strip() and rest.strip() != "(no relation)":
        for part in rest.split(" & "):
            pol = not part.startswith("not ")
            part = part[4:] if not pol else part
            name, _, args = part.partition("(")
            lits.append((name, tuple(int(x) for x in args.rstrip(")").split(",") if x != ""), pol))
    return Frame(types, tuple(lits))


# ---------------------------------------------------------------- the exhaustive pass, and generalization, share one check
_RELATIVE = False                                                            # the outcome definition the prove/generalize passes use (set by main)


def _check(dom: Domain, fr: Frame, slot: int, claim: str, guard: list, cap: int, min_cov: float, min_pur: float) -> dict:
    g = Generator(dom, cap=cap, log=lambda *a, **k: None, relative=_RELATIVE)
    rows, complete = g.rows_of(fr, slot)
    if not rows:
        return dict(frame=fr.text(), slot=slot, claim=claim, status="no placements")
    n = len(rows); all_mask = (1 << n) - 1
    loses = 0
    for i, (_, _, _, d) in enumerate(rows):
        if d < 0:
            loses |= 1 << i
    exc = (all_mask & ~loses) if claim == "acting loses" else loses
    n_exc = _popcount(exc)
    cache: dict = {}
    matched = 0
    for term in guard:
        matched |= g.term_mask(rows, term, cache, all_mask)
    cov = _popcount(matched & exc) / n_exc if n_exc else 1.0
    pur = _popcount(matched & exc) / max(1, _popcount(matched)) if guard else 1.0
    # a lemma admitted with NO guard claimed the outcome held on every sampled row; at the larger cap a handful of
    # exceptions appear (2 of 5,000): it still holds as a near-total claim if the exception rate stays under 2%
    near_total = not guard and n_exc <= 0.02 * n
    status = "PROVED" if (n_exc == 0 or near_total or (cov >= min_cov and pur >= min_pur)) else "conjecture"
    return dict(id=lemma_id(fr.types, fr.literals, slot, claim, guard), frame=fr.text(), slot=slot, claim=claim, support=n, exceptions=n_exc, coverage=cov, purity=pur, complete=complete, status=status, guard=guard_text(guard), near_total=near_total)


_WORKER_DOM = None


def _prove_job(args):
    entry, cap, min_cov, min_pur = args
    fr = Frame(tuple(entry["types"]), tuple((r, tuple(sl), pol) for r, sl, pol in entry["literals"])) if "types" in entry else parse_frame(entry["frame"])
    guard = guard_from_json(entry.get("inherited")) + guard_from_json(entry.get("guard"))
    return _check(_WORKER_DOM, fr, entry["slot"], entry["claim"], guard, cap, min_cov, min_pur)


def _pool_map(fn, jobs, workers: int, log, label: str):
    t0 = time.perf_counter()
    out = []
    if workers > 1 and len(jobs) >= 4:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            for k, o in enumerate(pool.imap(fn, jobs, chunksize=1)):
                out.append(o)
                if k and k % 100 == 0:
                    log(f"[lemma_gen/{label}] {k}/{len(jobs)} ({time.perf_counter() - t0:.0f}s)")
    else:
        out = [fn(j) for j in jobs]
    return out


def prove(dom: Domain, library_path: Path, cap: int, min_cov: float = 0.95, min_pur: float = 0.8, limit: int = 0, log=print, workers: int = 1):
    """THE EXHAUSTIVE PASS: re-run each admitted lemma with a far larger cap (exhaustive where the frame fits) and
    check the STORED full guard (inherited + own) -- not a re-fit -- for coverage and purity."""
    global _WORKER_DOM
    _WORKER_DOM = dom
    lib = json.load(open(library_path))
    entries = lib[: limit or len(lib)]
    t0 = time.perf_counter()
    out = _pool_map(_prove_job, [(e, cap, min_cov, min_pur) for e in entries], workers, log, "prove")
    for o in out:
        if "support" in o:
            log(f"  {o['status']:<10} {o.get('id', '')} {o['frame']} :: slot {o['slot']} {o['claim']} {100 * (o['support'] - o['exceptions']) / o['support']:.1f}% of {o['support']}; guard {o['guard']}: covers {100 * o['coverage']:.1f}% of {o['exceptions']} exceptions, purity {o['purity']:.2f}" + (" (complete)" if o["complete"] else f" (cap {cap})"))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"proved_{dom.name}.json", "w"), indent=1)
    c = Counter(o["status"] for o in out)
    log(f"[lemma_gen/prove] {dict(c)} ({time.perf_counter() - t0:.0f}s)")
    return out


# ---------------------------------------------------------------- generalization: anti-unification over the slot types
def schema_key(entry: dict, over_literals: bool = False):
    """Everything about a lemma except its slot types: literals, actor, claim, the full guard (as a set of terms).
    over_literals: the frame literals' RELATION NAMES become variables too (their slots and polarity stay), so the
    aligned / attacks / same_side siblings of a frame fall into one schema."""
    guard = guard_from_json(entry.get("inherited")) + guard_from_json(entry.get("guard"))
    lits = sorted((r, tuple(sl), pol) for r, sl, pol in entry["literals"])
    shape = tuple((None if over_literals else r, sl, pol) for r, sl, pol in lits)
    return (len(entry["types"]), shape, entry["slot"], entry["claim"], tuple(sorted(set(guard))))


def schema_tuple(entry: dict, over_literals: bool) -> tuple:
    """The cell a lemma occupies: its slot types, plus (over_literals) its literals' relation names in sorted-literal order."""
    lits = sorted((r, tuple(sl), pol) for r, sl, pol in entry["literals"])
    return tuple(entry["types"]) + (tuple(r for r, _, _ in lits) if over_literals else ())


def box_cover(held: set, k: int) -> list:
    """The largest product sub-boxes S_0 x ... x S_(k-1) inside a set of held type tuples, chosen greedily (most
    uncovered tuples, then size; ties by order), until every held tuple lies in some box. Deterministic; no domain
    knowledge -- if the boxes line up with an attribute (colour, value), that attribute is discovered, not assumed."""
    boxes = []; uncovered = set(held)
    per_slot = [sorted({h[i] for h in held}) for i in range(k)]
    while uncovered:
        best = None
        for seed in sorted(uncovered):
            box = [{seed[i]} for i in range(k)]
            improved = True
            while improved:
                improved = False
                for i in range(k):
                    for t in per_slot[i]:
                        if t in box[i]:
                            continue
                        cells = itertools.product(*[box[j] if j != i else {t} for j in range(k)])
                        if all(c in held for c in cells):
                            box[i].add(t); improved = True
            cells = set(itertools.product(*box))
            key = (len(cells & uncovered), len(cells))
            if best is None or key > best[0]:
                best = (key, box)
        boxes.append([sorted(b) for b in best[1]]); uncovered -= set(itertools.product(*best[1]))
    return boxes


def _gen_job(args):
    types, literals, slot, claim, guard, cap, min_cov, min_pur = args
    fr = Frame(tuple(types), tuple((r, tuple(sl), pol) for r, sl, pol in literals))
    return _check(_WORKER_DOM, fr, slot, claim, guard, cap, min_cov, min_pur)


def generalize(dom: Domain, library_path: Path, proved_path: Path | None, cap: int, min_cov: float = 0.95, min_pur: float = 0.8, log=print, workers: int = 1, only_proved: bool = True,
               over_literals: bool = False, max_unseen: int = 0):
    """Group lemmas by schema_key; per slot the set of observed types; the unseen instantiations (product minus
    observed) are checked at the prover's cap. A SCHEMA states the lemma over exactly the type tuples that held."""
    global _WORKER_DOM
    _WORKER_DOM = dom
    lib = json.load(open(library_path))
    status = {}
    if proved_path and Path(proved_path).exists():
        for e, p in zip(lib, json.load(open(proved_path))):
            status[(e["frame"], e["slot"])] = p.get("status")
    groups: dict = defaultdict(list)
    for e in lib:
        if only_proved and status and status.get((e["frame"], e["slot"])) != "PROVED":
            continue
        groups[schema_key(e, over_literals)].append(e)
    cache = {}
    tag = "_lit" if over_literals else ""
    prior = OUT / f"schemas{tag}_{dom.name}.json"
    if prior.exists():                                                        # cells already checked by an earlier pass are not re-proved
        for ps in json.load(open(prior)):
            pk = (ps["size"], tuple(sorted((r, tuple(sl), pol) for r, sl, pol in ps["literals"])), ps["slot"], ps["claim"], ps["guard"])
            if ps.get("over_literals"):
                pk = (ps["size"], tuple((None, tuple(sl), pol) for r, sl, pol in ps["shape"]), ps["slot"], ps["claim"], ps["guard"])
            for u in ps.get("unseen_proved", []) + ps.get("unseen_failed", []):
                cache[(pk, tuple(u[0]))] = u
    schemas = []; jobs = []; job_owner = []
    for key, members in groups.items():
        size, shape, slot, claim, guard = key
        observed = sorted({schema_tuple(m, over_literals) for m in members})
        width = len(observed[0])
        per_slot = [sorted({t[i] for t in observed}) for i in range(width)]
        seen = set(observed)
        unseen = [tuple(p) for p in itertools.product(*per_slot) if tuple(p) not in seen]
        if max_unseen and len(unseen) > max_unseen:
            unseen = unseen[:max_unseen]                                       # bounded: the cells nearest in product order
        s = dict(size=size, width=width, over_literals=over_literals, shape=[[r, list(sl), pol] for r, sl, pol in shape], slot=slot, claim=claim, guard=guard_text(list(guard)),
                 guard_terms=[[[r, list(sl), pol] for r, sl, pol in t] for t in guard], per_slot=per_slot, observed=[list(o) for o in observed],
                 unseen=[list(u) for u in unseen], unseen_proved=[], unseen_failed=[])
        s["literals"] = s["shape"]
        schemas.append(s)
        pk = (size, shape, slot, claim, guard_text(list(guard)))
        def lits_of(cell):
            if not over_literals:
                return [(r, sl, pol) for r, sl, pol in shape]
            return [(cell[size + j], sl, pol) for j, (_, sl, pol) in enumerate(shape)]
        for u in unseen:
            hit = cache.get((pk, tuple(u)))
            if hit is not None:
                (s["unseen_proved"] if hit[1] == "PROVED" else s["unseen_failed"]).append(hit); continue
            jobs.append((u[:size], lits_of(u), slot, claim, list(guard), cap, min_cov, min_pur)); job_owner.append(len(schemas) - 1)
    log(f"[lemma_gen/generalize] {len(lib)} lemmas ({'proved only' if only_proved and status else 'all'}) -> {len(groups)} schemas; {len(jobs)} unseen instantiations to check ({len(cache)} cached)")
    results = _pool_map(_gen_job, jobs, workers, log, "generalize")
    for owner, job, o in zip(job_owner, jobs, results):
        cell = list(job[0]) + ([r for r, _, _ in job[1]] if over_literals else [])
        (schemas[owner]["unseen_proved"] if o.get("status") == "PROVED" else schemas[owner]["unseen_failed"]).append([cell, o.get("status"), round(o.get("coverage", 0) or 0, 2), round(o.get("purity", 0) or 0, 2)])
    for s in schemas:
        held = {tuple(o) for o in s["observed"]} | {tuple(u[0]) for u in s["unseen_proved"]}
        w = s["width"]
        s["boxes"] = box_cover(held, w)                                        # the schema IS this union of product boxes: exactly what held
        s["held"] = len(held); s["cells"] = sum(1 for _ in itertools.product(*[sorted({t[i] for t in held}) for i in range(w)]))
        s["per_slot_held"] = [sorted({t[i] for t in held}) for i in range(w)]
        def box_text(box):
            head = " ".join("X%d∈{%s}" % (i, ",".join(box[i])) for i in range(s["size"]))
            if over_literals:
                lits = " & ".join(("" if pol else "not ") + "{%s}(%s)" % (",".join(box[s["size"] + j]), ",".join(map(str, sl))) for j, (_, sl, pol) in enumerate(s["shape"]))
                return f"[{head}] {lits or '(no relation)'}"
            return f"[{head}]"
        s["text"] = (" ∪ ".join(box_text(box) for box in s["boxes"]) + " "
                     + ("" if over_literals else (" & ".join(lit_text(r, sl, pol) for r, sl, pol in s["literals"]) or "(no relation)"))
                     + f" :: slot {s['slot']} {s['claim']} unless {s['guard'] or '(nothing)'}"
                     + f"  [{len(s['observed'])} observed, {len(s['unseen_proved'])} unseen proved, {len(s['unseen_failed'])} unseen failed; {len(s['boxes'])} boxes over {s['held']} held of {s['cells']} in the union box]")
    schemas.sort(key=lambda s: (-s["held"], len(s["boxes"])))
    for s in schemas:
        log("  " + s["text"])
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(schemas, open(OUT / f"schemas{tag}_{dom.name}.json", "w"), indent=1)
    n_inst = sum(len(s["observed"]) for s in schemas)
    nb = sum(len(s["boxes"]) for s in schemas)
    log(f"[lemma_gen/generalize] {n_inst} lemmas -> {len(schemas)} schemas stated as {nb} boxes; unseen instantiations proved {sum(len(s['unseen_proved']) for s in schemas)}, failed {sum(len(s['unseen_failed']) for s in schemas)}; schemas that are ONE box: {sum(1 for s in schemas if len(s['boxes']) == 1)}")
    return schemas


def _examples_job(args):
    entry, cap = args
    fr = Frame(tuple(entry["types"]), tuple((r, tuple(sl), pol) for r, sl, pol in entry["literals"]))
    g = Generator(_WORKER_DOM, cap=cap, log=lambda *a, **k: None, relative=_RELATIVE)
    rows, _ = g.rows_of(fr, entry["slot"])
    if not rows:
        return entry.get("examples", {})
    n = len(rows); all_mask = (1 << n) - 1
    loses = 0
    for i, (_, _, _, d) in enumerate(rows):
        if d < 0:
            loses |= 1 << i
    exc = (all_mask & ~loses) if entry["claim"] == "acting loses" else loses
    cache: dict = {}
    inh = 0
    for term in guard_from_json(entry.get("inherited")):
        inh |= g.term_mask(rows, term, cache, all_mask)
    rest = exc & ~inh
    ex = pick_examples(g, rows, exc, rest, guard_from_json(entry.get("guard")), cache, all_mask)
    closure = g.frame_closure(rows, fr, cache, all_mask)
    return dict(examples=ex, closure=[[r, list(sl), pol, tag] for r, sl, pol, tag in closure])


def refresh_examples(dom: Domain, library_path: Path, cap: int, workers: int, log=print):
    global _WORKER_DOM
    _WORKER_DOM = dom
    lib = json.load(open(library_path))
    out = _pool_map(_examples_job, [(e, cap) for e in lib], workers, log, "examples")
    changed = 0
    for e, r in zip(lib, out):
        ex = r["examples"] if isinstance(r, dict) and "examples" in r else r
        if ex != e.get("examples"):
            changed += 1
        e["examples"] = ex
        if isinstance(r, dict) and "closure" in r:
            e["closure"] = r["closure"]
    json.dump(lib, open(library_path, "w"), indent=1)
    log(f"[lemma_gen/examples] {len(lib)} lemmas, examples recomputed ({changed} changed) -> {library_path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "prove", "generalize", "examples"]); ap.add_argument("--domain", default="chess")
    ap.add_argument("--library", default=str(OUT / "library_chess.json")); ap.add_argument("--proved", default=str(OUT / "proved_chess.json")); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-size", type=int, default=3); ap.add_argument("--cap", type=int, default=3000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--max-conj", type=int, default=2); ap.add_argument("--all", action="store_true", help="generalize: over all lemmas, not only the proved")
    ap.add_argument("--over-literals", action="store_true", help="generalize: the frame literals' relation names are variables too"); ap.add_argument("--max-unseen", type=int, default=0)
    ap.add_argument("--source", choices=["sampled", "games"], default="sampled", help="placements: sampled frames (only the frame's pieces) or real game positions")
    ap.add_argument("--positions", type=int, default=20000, help="games source: how many positions from the knowledge corpus"); ap.add_argument("--lo", type=int, default=10); ap.add_argument("--hi", type=int, default=120)
    ap.add_argument("--relative", action="store_true", help="outcome relative to the null action (what the standing threats already take)")
    ap.add_argument("--max-frames", type=int, default=0, help="per size, evaluate only the most frequent frames")
    ap.add_argument("--oracle", choices=["material", "resolved", "tablebase"], default="material", help="the outcome: 1-ply material after the best capture/check reply, or the tablebase's WDL (sampled frames of <= 5 men)")
    ap.add_argument("--worth", choices=["canonical", "blend"], default="canonical", help="the worth relations: the value table, or the tables' removal value blended toward it by distance to the five-men edge"); ap.add_argument("--worth-tau", type=float, default=4.0)
    a = ap.parse_args(argv)
    if a.domain == "chess":
        from .chess_domain import domain
        fens = None
        if a.source == "games":
            import random as _r
            from .dataset import load as load_ds
            meta = load_ds("data/derived/knowledge", meta_only=True)["meta"]
            rows = [m["fen"] for m in meta if a.lo <= m["ply"] <= a.hi]
            _r.Random(a.seed).shuffle(rows); fens = rows[: a.positions]
            print(f"[lemma_gen] games source: {len(fens)} positions (plies {a.lo}-{a.hi})")
        dom = domain(fens, oracle=a.oracle, worth=a.worth, worth_tau=a.worth_tau)
    else:
        raise SystemExit(f"unknown domain {a.domain}")
    global _RELATIVE
    _RELATIVE = a.relative
    if a.cmd == "prove":
        prove(dom, Path(a.library), a.cap, limit=a.limit, workers=a.workers); return
    if a.cmd == "examples":
        refresh_examples(dom, Path(a.library), a.cap, a.workers); return
    if a.cmd == "generalize":
        generalize(dom, Path(a.library), Path(a.proved), a.cap, workers=a.workers, only_proved=not a.all, over_literals=a.over_literals, max_unseen=a.max_unseen); return
    g = Generator(dom, cap=a.cap, seed=a.seed, workers=a.workers, max_conj=a.max_conj, relative=a.relative, max_frames=a.max_frames)
    lib = g.run(max_size=a.max_size)
    print(f"[lemma_gen] library: {len(lib)} lemmas -> {OUT}")


if __name__ == "__main__":
    main()
