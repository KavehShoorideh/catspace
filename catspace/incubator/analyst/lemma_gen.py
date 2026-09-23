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


# a literal is (relation name, slots, polarity); a TERM is a tuple of literals (conjunction); a GUARD is a list of terms (disjunction)
def lit_text(r, sl, pol):
    return ("" if pol else "not ") + f"{r}({','.join(str(s) for s in sl)})"


def term_text(term):
    return " & ".join(lit_text(r, sl, pol) for r, sl, pol in term)


def guard_text(guard):
    return " or ".join(term_text(t) if len(t) == 1 else "(" + term_text(t) + ")" for t in guard)


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

    def full_guard(self):
        return list(self.inherited) + list(self.guard)

    def text(self):
        g = guard_text(self.guard)
        return (f"{self.frame.text()} :: slot {self.slot} {self.claim} in {100 * self.holds / max(1, self.support):.1f}% of {self.support}"
                + (f"; unless {g} (covers {self.covered}/{self.covered + self.residual} exceptions, purity {self.purity:.2f})" if self.guard else f"; {self.residual} exceptions unexplained")
                + (f"; inherited [{guard_text(self.inherited)}] {'REVERSED' if self.reversed else 'broke'} on {self.broke}" if self.parents else "") + ("" if self.complete else " [sampled]"))


def _popcount(x: int) -> int:
    return x.bit_count()


class Generator:
    def __init__(self, dom: Domain, cap: int = 3000, seed: int = 0, log=print, workers: int = 1, max_conj: int = 2):
        self.dom = dom; self.cap = cap; self.seed = seed; self.log = log; self.workers = workers; self.max_conj = max_conj
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
            for a in self.dom.actions(state, e):
                before = self.dom.scalar(state, e)
                after = self.dom.transition(state, a)
                delta = self.dom.reply_scalar(after, e) - before
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
    def find_guard(self, rows, exc: int, frame: Frame, cache: dict, max_terms: int = 3):
        """exc: bitmask of the exceptions to explain. Returns (guard terms, covered, purity, matched)."""
        n = len(rows); all_mask = (1 << n) - 1
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
                pur = _popcount(m & exc) / _popcount(m)
                score = gain * pur
                if best is None or score > best[0] or (score == best[0] and len(term) < len(best[1])):     # Occam: a longer term must strictly beat a shorter one
                    best = (score, term, m, pur)
            if best is None or best[3] < 0.6:
                break
            chosen.append(best[1]); covered |= best[2] & exc
        matched_any = 0
        for term in chosen:
            matched_any |= self.term_mask(rows, term, cache, all_mask)
        cov = _popcount(matched_any & exc)
        purity = cov / max(1, _popcount(matched_any))
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
        guard, covered, purity, matched = self.find_guard(rows, rest, frame, cache)
        residual = _popcount(rest) - covered
        examples = {}
        for i, (state, ids, a, d) in enumerate(rows):
            key = "exception" if (exc >> i) & 1 else "holds"
            if key not in examples:
                examples[key] = (self.dom_text(state), str(a), round(d, 2))
        lem = Lemma(frame, slot, claim, n, holds, guard, covered, purity, residual, complete, examples, inherited, broke, reversed_, parents)
        lem.sig = sig
        return lem

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
                out.append((lem.claim, remap_guard(lem.full_guard(), m), lem.frame.text() + f" slot {lem.slot}", missing))
                break
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
        self.save()
        return self.library

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
    return dict(frame=l.frame.text(), types=list(l.frame.types), literals=[[r, list(sl), pol] for r, sl, pol in l.frame.literals], slot=l.slot, claim=l.claim,
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
def _check(dom: Domain, fr: Frame, slot: int, claim: str, guard: list, cap: int, min_cov: float, min_pur: float) -> dict:
    g = Generator(dom, cap=cap, log=lambda *a, **k: None)
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
    status = "PROVED" if (n_exc == 0 or (cov >= min_cov and pur >= min_pur)) else "conjecture"
    return dict(frame=fr.text(), slot=slot, claim=claim, support=n, exceptions=n_exc, coverage=cov, purity=pur, complete=complete, status=status, guard=guard_text(guard))


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
            log(f"  {o['status']:<10} {o['frame']} :: slot {o['slot']} {o['claim']} {100 * (o['support'] - o['exceptions']) / o['support']:.1f}% of {o['support']}; guard {o['guard']}: covers {100 * o['coverage']:.1f}% of {o['exceptions']} exceptions, purity {o['purity']:.2f}" + (" (complete)" if o["complete"] else f" (cap {cap})"))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"proved_{dom.name}.json", "w"), indent=1)
    c = Counter(o["status"] for o in out)
    log(f"[lemma_gen/prove] {dict(c)} ({time.perf_counter() - t0:.0f}s)")
    return out


# ---------------------------------------------------------------- generalization: anti-unification over the slot types
def schema_key(entry: dict):
    """Everything about a lemma except its slot types: literals, actor, claim, the full guard (as a set of terms)."""
    guard = guard_from_json(entry.get("inherited")) + guard_from_json(entry.get("guard"))
    return (len(entry["types"]), tuple(sorted((r, tuple(sl), pol) for r, sl, pol in entry["literals"])), entry["slot"], entry["claim"], tuple(sorted(set(guard))))


def _gen_job(args):
    types, literals, slot, claim, guard, cap, min_cov, min_pur = args
    fr = Frame(tuple(types), tuple((r, tuple(sl), pol) for r, sl, pol in literals))
    return _check(_WORKER_DOM, fr, slot, claim, guard, cap, min_cov, min_pur)


def generalize(dom: Domain, library_path: Path, proved_path: Path | None, cap: int, min_cov: float = 0.95, min_pur: float = 0.8, log=print, workers: int = 1, only_proved: bool = True):
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
        groups[schema_key(e)].append(e)
    schemas = []; jobs = []; job_owner = []
    for key, members in groups.items():
        size, literals, slot, claim, guard = key
        observed = sorted({tuple(m["types"]) for m in members})
        per_slot = [sorted({t[i] for t in observed}) for i in range(size)]
        seen = set(observed)
        unseen = [tuple(p) for p in itertools.product(*per_slot) if tuple(p) not in seen]
        s = dict(size=size, literals=[[r, list(sl), pol] for r, sl, pol in literals], slot=slot, claim=claim, guard=guard_text(list(guard)),
                 guard_terms=[[[r, list(sl), pol] for r, sl, pol in t] for t in guard], per_slot=per_slot, observed=[list(o) for o in observed],
                 unseen=[list(u) for u in unseen], unseen_proved=[], unseen_failed=[])
        schemas.append(s)
        for u in unseen:
            jobs.append((u, [(r, sl, pol) for r, sl, pol in literals], slot, claim, list(guard), cap, min_cov, min_pur)); job_owner.append(len(schemas) - 1)
    log(f"[lemma_gen/generalize] {len(lib)} lemmas ({'proved only' if only_proved and status else 'all'}) -> {len(groups)} schemas; {len(jobs)} unseen instantiations to check")
    results = _pool_map(_gen_job, jobs, workers, log, "generalize")
    for owner, job, o in zip(job_owner, jobs, results):
        (schemas[owner]["unseen_proved"] if o.get("status") == "PROVED" else schemas[owner]["unseen_failed"]).append([list(job[0]), o.get("status"), round(o.get("coverage", 0) or 0, 2), round(o.get("purity", 0) or 0, 2)])
    schemas.sort(key=lambda s: -(len(s["observed"]) + len(s["unseen_proved"])))
    for s in schemas:
        held = [tuple(o) for o in s["observed"]] + [tuple(u[0]) for u in s["unseen_proved"]]
        s["per_slot_held"] = [sorted({t[i] for t in held}) for i in range(s["size"])]
        s["text"] = ("[" + " ".join("X%d∈{%s}" % (i, ",".join(ts)) for i, ts in enumerate(s["per_slot_held"])) + "] "
                     + (" & ".join(lit_text(r, sl, pol) for r, sl, pol in s["literals"]) or "(no relation)")
                     + f" :: slot {s['slot']} {s['claim']} unless {s['guard'] or '(nothing)'}"
                     + f"  [{len(s['observed'])} observed, {len(s['unseen_proved'])} unseen proved, {len(s['unseen_failed'])} unseen failed]")
        log("  " + s["text"])
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(schemas, open(OUT / f"schemas_{dom.name}.json", "w"), indent=1)
    n_inst = sum(len(s["observed"]) for s in schemas)
    log(f"[lemma_gen/generalize] {n_inst} lemmas -> {len(schemas)} schemas; unseen instantiations proved {sum(len(s['unseen_proved']) for s in schemas)}, failed {sum(len(s['unseen_failed']) for s in schemas)}")
    return schemas


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "prove", "generalize"]); ap.add_argument("--domain", default="chess")
    ap.add_argument("--library", default=str(OUT / "library_chess.json")); ap.add_argument("--proved", default=str(OUT / "proved_chess.json")); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-size", type=int, default=3); ap.add_argument("--cap", type=int, default=3000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--max-conj", type=int, default=2); ap.add_argument("--all", action="store_true", help="generalize: over all lemmas, not only the proved")
    a = ap.parse_args(argv)
    if a.domain == "chess":
        from .chess_domain import domain
        dom = domain()
    else:
        raise SystemExit(f"unknown domain {a.domain}")
    if a.cmd == "prove":
        prove(dom, Path(a.library), a.cap, limit=a.limit, workers=a.workers); return
    if a.cmd == "generalize":
        generalize(dom, Path(a.library), Path(a.proved), a.cap, workers=a.workers, only_proved=not a.all); return
    g = Generator(dom, cap=a.cap, seed=a.seed, workers=a.workers, max_conj=a.max_conj)
    lib = g.run(max_size=a.max_size)
    print(f"[lemma_gen] library: {len(lib)} lemmas -> {OUT}")


if __name__ == "__main__":
    main()
