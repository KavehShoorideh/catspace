"""PLAN 76.4c (b) MINING OBJECTS (docs/ANALYST_PLAN.md 76.4; Kaveh 2026-10-28: a DERIVED OBJECT is a defined relation over the base relations -- "line is not in the
rules of chess: the rook's movement generates it" -- and "whatever we define as an object will therefore get a node in the graph"). GENERIC: this module names no
game object. The GENERATOR proposes `define`d relations of the one term language (terms.py) over the spec's relations, cheapest first (priced by size), and SCORES
each by USE only (plan 76's borrowing (5), interestingness from use -- never a hand-written score), the plan's two properties:

  (i)  does the term appear in a ZERO-FAILURE MEASURE of a class (the technique miner re-run with the derived relations admitted as atoms: read by the plan's
       script for the relations (ii) accepts -- a whole-class miner run per library prices it last);
  (ii) is it CITED by a stated proof obligation or by the regression's derivations -- each a TEST (`Test`):
       THE DERIVATIONS (`harvest`): the regressions of the store's GOAL entries on the guard positions, read by the regression's Python prover (regression.py).
       Every TRUE literal of their derivations with two location arguments (a, b) is a READING, with the locations its derivation's LEAVES stand on besides a
       and b (the fluents and the negated atoms it rests on); a WITNESS of a defect whose action FLIPS a leaf of a true reading gives a FLIP: the reading after
       the action -- one test per relation and the relation of the leaf the action flipped (a unit placed on a negated leaf, a fluent lifted). A relation of three
       locations EXPLAINS a relation R's readings when, at every reading of R, the middle locations it gives (a, b) are exactly the reading's leaf locations (some
       reading has one); a relation of two locations explains R's flips (of one leaf relation) when it holds at (a, b) after every such flip of R and at no true
       reading of R -- what the action did, named.
       THE OBLIGATIONS: a set of locations a stated obligation's measure cites (its values given by the caller on the class's positions) is explained by a
       candidate equal to it on every one of them.
       A candidate never reads the relation it explains (no explanation by itself).
THE PRIOR (borrowing (5) as the miner's prior): a test's ALPHABET is the base its citing source rests on -- the relations of the cited readings' derivations'
LEAVES and the static tables their clauses read, or the relations the obligation's term reads through the shelf (ontology.term_relations) -- plus the accepted
relations; `without` reads the placement fluent itself
(its lifted units are the placement's literals). THE SEARCH (`Generator`, one per test): bottom-up by size over typed productions -- the alphabet's literals (a side
slot read relatively, us / them or anonymous; a unit slot a constant or anonymous; another slot held or anonymous), not, and, or, minus, closure, restrict, compose,
via, exists, reach, extreme, without -- only in the sorts the test's sort is built from; OBSERVATIONAL EQUIVALENCE on a sample of positions (two terms of one sort
with the same values there are one candidate: the cheaper); a term empty at every sample position names nothing (dropped). THE FUSE (named, with its reason):
no candidate is larger than the largest clause body the spec itself states (`body_fuse`: a derived relation no bigger than the rules' own largest definition).
LEVELS, SIMPLEST FIRST: the tests' searches run size by size together; a level ENDS at the first size at which some test accepts a candidate (the cheapest that
explains it; among equally cheap ones the MOST GENERAL -- the most rows on the test's states, the general boundary of the version space -- then the first in
print order); the accepted relations are DEFINED (a name of size one) and the next level searches again with them; the levels end when one accepts nothing.
LEARNING FROM FAILURES (plan 76's borrowing (4)): a candidate whose explanation fails at a reading because it MISSES a location prunes every candidate that
implies it (terms.implies: it misses the location too) from that test; one that fails because it holds an EXTRA location prunes every candidate it implies --
counted as pruned, never evaluated against that test.
"""
from __future__ import annotations

import itertools
import time
from collections import Counter
from dataclasses import dataclass, field

from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.incubator.analyst.discovery import terms as TL


def body_fuse(prog: GDL.Program) -> int:
    """THE FUSE: the most literals any clause of the spec has in its body (a derived relation is priced no higher than the rules' own largest definition)."""
    return max(len([l for l in c.body if l[0] in ("pos", "neg")]) for c in prog.clauses)


# ------------------------------------------------------------------------------------------------------------------------------------------ the readings

@dataclass
class Readings:
    """What the regression's derivations read: `states` [(state, side)]; `true` [(state index, relation, a, b, the leaf locations besides a and b)]; `flips`
    [(state index before, state index after, relation, a, b, the flipped leaf's relation)]; `reads` relation -> the BASE its readings' derivations rest on: the relations of their leaves (the
    fluents, the negated atoms) and the static tables their clauses read."""
    states: list = field(default_factory=list)
    true: list = field(default_factory=list)
    flips: list = field(default_factory=list)
    reads: dict = field(default_factory=dict)


def _locs_of(dom: TL.Domain, rel: str, args: tuple) -> list:
    """The ground locations of an atom, in its slots' order (an unbound one left out)."""
    out = []
    n = len(dom.comps)
    for s, i in dom.slots(rel):
        if s == TL.LOC:
            t = tuple(args[i:i + n])
            if all(v is not None and not GDL.is_var(v) for v in t) and t in dom.index:
                out.append(dom.index[t])
    return out


def harvest(R, dom: TL.Domain, cases: list, log=print) -> Readings:
    """THE DERIVATIONS' READINGS (see the docstring) over `cases` [(state, side, [goals])] -- `R` a PyRegressor (its prover; its binding's `apply`, the certified next
    state). A holding goal's rows and every true ground literal of a regression's clauses are walked down their derivations; each node of two location arguments
    is a reading with its leaves' locations."""
    out = Readings()
    t0 = time.monotonic()
    nreg = 0
    for x, side, goals in cases:
        si = len(out.states)
        out.states.append((x, side))
        for g in goals:
            reg = R.regress(x, g)
            nreg += 1
            st = reg.state if reg.state is not None else x
            P = R.prover(st)
            mine: dict = {}

            def walk(rel: str, args: tuple, depth: int = 0) -> set:
                """The relations the derivations of (rel, args) read (recorded per reading)."""
                if (rel, args) in mine or depth > 40 or R.spec.kind(rel) in ("static", "input"):
                    return set()
                lv = P.leaves(rel, args)
                mine[(rel, args)] = lv
                below = set()
                for _idx, lits in P.derivations(rel, args):
                    for kind, r2, av in lits:
                        if kind in ("pos", "neg") and R.spec.kind(r2) == "static":
                            below.add(r2)                       # the static tables its clauses read (the geometry)
                        if kind == "pos" and all(v is not None for v in av):
                            below |= walk(r2, av, depth + 1)
                ls = _locs_of(dom, rel, args)
                if len(ls) == 2:
                    leaf_locs = set()
                    for leaf in lv:
                        leaf_locs |= set(_locs_of(dom, leaf.atom[0], leaf.atom[1]))
                        below.add(leaf.atom[0])
                    out.true.append((si, rel, ls[0], ls[1], frozenset(leaf_locs - set(ls))))
                    out.reads.setdefault(rel, set()).update(below)
                return below

            for row in reg.rows:                                   # a holding head: each of its rows, walked down its derivations
                walk(g.head[0], tuple(row))
            for _ci, lits in reg.clauses:
                for _li, kind, atom, truth in lits:
                    if kind == "pos" and truth and all(v is not None and not GDL.is_var(v) for v in atom[1]):
                        walk(atom[0], tuple(atom[1]))
            for d in reg.defects:
                w = d.witness
                if w is None or w.action is None or not w.flips:
                    continue
                hit = [(rel, args, lv & w.flips) for (rel, args), lv in mine.items() if lv & w.flips and len(_locs_of(dom, rel, args)) == 2]
                if not hit:
                    continue
                x2 = R.B.apply(st, w.action)
                sj = len(out.states)
                out.states.append((x2, side))
                for rel, args, flipped in hit:
                    a, b = _locs_of(dom, rel, args)
                    for leaf_rel in sorted({leaf.atom[0] for leaf in flipped}):
                        out.flips.append((si, sj, rel, a, b, leaf_rel))
    log(f"OBS objects harvest: states {len(out.states)} ({len(cases)} positions, {len(out.states) - len(cases)} after a flipping witness), regressions {nreg}, "
        f"true readings of two locations {len(out.true)} (" + ", ".join(f"{r} {n}" for r, n in sorted(Counter(t[1] for t in out.true).items())) + f"), flips "
        f"{len(out.flips)} (" + ", ".join(f"{r} {n}" for r, n in sorted(Counter(t[2] for t in out.flips).items())) + f"); {time.monotonic() - t0:.1f} s")
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ the tests

@dataclass
class Test:
    """One citing source: `name`, `sort` (the sort a candidate must have), `alphabet` (the spec relations its source reads), `rel` (the relation explained: never
    read by a candidate), `kind` ('readings' | 'flips' | 'set') and its data."""
    name: str
    kind: str
    sort: tuple
    alphabet: frozenset
    rel: "str | None" = None
    data: object = None


def tests_of(dom: TL.Domain, readings: Readings, cited_sets: dict) -> list:
    """The tests: per relation with readings of which some has a leaf location, its readings (a relation of three locations: the sort (P.., loc, loc, loc), the P
    the relation's held slots); per relation with flips, its flips (two locations); per cited set of an obligation, the set (`cited_sets`: name -> (the spec
    relations its term reads, [(state, side)], [the set's locations at each])). (A relation of the readings with no held slot reads none.)"""
    out = []
    rels = sorted({r[1] for r in readings.true})
    for rel in rels:
        rs = [r for r in readings.true if r[1] == rel]
        if any(r[4] for r in rs):
            out.append(Test(f"readings {rel}", "readings", None, frozenset(readings.reads.get(rel, ())) - {rel}, rel, rs))
    for rel, leaf_rel in sorted({(f[2], f[5]) for f in readings.flips}):
        fs = [f for f in readings.flips if f[2] == rel and f[5] == leaf_rel]
        tr = [r for r in readings.true if r[1] == rel]
        out.append(Test(f"flips {rel} by {leaf_rel}", "flips", None, frozenset(readings.reads.get(rel, ())) - {rel}, rel, (fs, tr)))
    for name, (rels, states, vals) in sorted(cited_sets.items()):
        out.append(Test(f"obligation {name}", "set", TL.LOCS, frozenset(rels), None, (states, vals)))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ the generator

@dataclass
class Cand:
    term: TL.T
    sig: tuple
    size: int
    rels: frozenset


class Generator:
    """THE SEARCH of one test (see the docstring) on a sample [(state, side)] through a reader: its alphabet's literals, the accepted names, the productions in
    the sorts the test's sort is built from."""

    def __init__(self, dom: TL.Domain, reader, ctxs: list, defs: TL.Defs, alphabet: frozenset, target):
        self.dom, self.reader, self.ctx, self.defs = dom, reader, ctxs, defs
        self.alphabet = alphabet
        self.target = target                       # a sort, or 'rel3' / 'rel2' (a relation of three / two locations, any held slots)
        self.pools: dict = {}                      # sort -> size -> [Cand]
        self.known: set = set()                     # (sort, sig)
        self.tried = 0
        self.dropped = Counter()
        self.static = set(dom.prog.static)
        self._lits = None

    # ---- the alphabet
    def literals(self) -> list:
        if self._lits is not None:
            return self._lits
        dom = self.dom
        out = []
        for rel in sorted(self.alphabet | {self.dom.placement}):
            if rel not in dom.prog.nodes or rel in GDL.INPUTS or (dom.prog.needed((rel,)) & set(GDL.INPUTS)):
                continue
            try:
                sl = dom.slots(rel)
            except TL.TermError:
                continue
            nloc = sum(1 for s, _i in sl if s == TL.LOC)
            if not 1 <= nloc <= 2:
                continue
            choices = []
            for s, _i in sl:
                if s == TL.LOC:
                    choices.append([None, TL.ANON] if nloc == 2 else [None])
                elif s == TL.SIDE:
                    choices.append([TL.US, TL.THEM, TL.ANON])
                elif s == TL.UNIT:
                    choices.append(dom.domain(s) + [TL.ANON])
                else:
                    choices.append([None, TL.ANON])
            for pat in itertools.product(*choices):
                open_slots = [s for (s, _i), p in zip(sl, pat) if p is None]
                if TL.LOC not in open_slots or len(open_slots) > 3:
                    continue
                try:
                    t = TL.lit(dom, rel, *pat)
                except TL.TermError:
                    continue
                if rel == self.dom.placement and rel not in self.alphabet and t.sort != TL.LOCS:
                    continue                                   # (the placement only for `without`'s lifted units)
                out.append(t)
        self._lits = out
        return out

    def wanted(self, sort) -> bool:
        """The sorts the test's sort is built from: a set of locations always (restrict, reach, without read one); for a relation of three locations its two-location
        part; for two locations the three-location relations (exists projects their middle) and the plain (location, location) pairs."""
        if sort == TL.LOCS or sort == (TL.LOC, TL.LOC):
            return True
        if not isinstance(sort, tuple) or len(sort) < 2 or sort[-1] != TL.LOC or sort[-2] != TL.LOC:
            return False
        nloc = sum(1 for s in sort if s == TL.LOC)
        if self.target == TL.LOCS:
            return False
        return nloc in (2, 3) and len(sort) - nloc <= 1

    # ---- evaluation and observational equivalence
    def values(self, t: TL.T) -> "tuple | None":
        """The term on every sample state, the term itself not cached (its sub-terms, members of the pools, are)."""
        try:
            fn = TL._EVAL[t.op]
            if not (t.rels(self.dom, self.defs) - self.static):
                v = fn(self.ctx[0], t)
                return (v,) * len(self.ctx)
            return tuple(fn(c, t) for c in self.ctx)
        except (TL.TermError, KeyError, ValueError) as x:
            self.dropped[f"error {type(x).__name__}"] += 1
            return None

    def add(self, t: TL.T) -> "Cand | None":
        self.tried += 1
        if not self.wanted(t.sort):
            return None
        s = self.values(t)
        if s is None:
            return None
        if all(not v for v in s):
            self.dropped["empty on every sample position"] += 1
            return None
        key = (t.sort, hash(s))
        if key in self.known:
            self.dropped["observationally equal to a cheaper one"] += 1
            return None
        self.known.add(key)
        for c, v in zip(self.ctx, s):
            c._val[t] = v
        c = Cand(t, (), t.size, t.rels(self.dom, self.defs))
        self.pools.setdefault(t.sort, {}).setdefault(t.size, []).append(c)
        return c

    def pool(self, sort, size: int) -> list:
        return self.pools.get(sort, {}).get(size, [])

    def size1(self) -> list:
        out = []
        for t in self.literals() + [self.defs.ref(n) for n in sorted(self.defs.body)]:
            c = self.add(t)
            if c is not None:
                out.append(c)
        return out

    def family(self, sort) -> bool:
        """The sorts a candidate of the test may have (the rest are the AUXILIARY sorts its constructors read: a set of locations for a relation's test, the
        (location, location) steps for a set's)."""
        if self.target == TL.LOCS:
            return sort == TL.LOCS
        return isinstance(sort, tuple) and sum(1 for s in sort if s == TL.LOC) >= 2

    def grow(self, size: int) -> list:
        """Every new candidate of `size` (the productions over the pools of smaller sizes). An AUXILIARY sort's terms are grown two sizes behind: inside a
        candidate of `size` such a term is an argument beside at least one other (and the constructor's own price), so it is never larger than `size` - 2."""
        out = []
        if size - 2 >= 2:
            for t in self._terms(size - 2):
                if not self.family(t.sort):
                    self.add(t)
        for t in self._terms(size):
            if self.family(t.sort):
                c = self.add(t)
                if c is not None:
                    out.append(c)
        return out

    def _terms(self, size: int):
        placements = [l for l in self.literals() if l.sort == TL.LOCS and l.args[0] == self.dom.placement]
        for sort in sorted(self.pools, key=str):
            if sort == TL.LOCS:
                for a in self.pool(sort, size - 1):
                    if a.term.op != "not":
                        yield TL.mk("not", a.term)
            for i in range(1, size - 1):
                j = size - 1 - i
                for a in self.pool(sort, i):
                    for b in self.pool(sort, j):
                        if a is b:
                            continue
                        if i < j or (i == j and a.term.sexpr() < b.term.sexpr()):
                            yield TL.mk("and", a.term, b.term)
                            yield TL.mk("or", a.term, b.term)
                        yield TL.mk("minus", a.term, b.term)
            two = isinstance(sort, tuple) and len(sort) >= 2 and sort[-1] == TL.LOC and sort[-2] == TL.LOC
            if two:
                for a in self.pool(sort, size - 1):
                    if a.term.op != "closure":
                        yield TL.mk("closure", a.term)
                for i in range(1, size - 1):
                    j = size - 1 - i
                    for a in self.pool(sort, i):
                        for s in self.pool(TL.LOCS, j):
                            yield TL.mk("restrict", a.term, s.term)
                        for b in self.pool(sort, j):
                            yield TL.mk("compose", a.term, b.term)
                            yield TL.mk("via", a.term, b.term)
            if isinstance(sort, tuple) and len(sort) >= 2:
                for a in self.pool(sort, size - 1):
                    if a.term.op == "exists":
                        continue
                    for j in range(len(sort)):
                        rest = sort[:j] + sort[j + 1:]
                        if TL.LOC in rest:
                            yield TL.mk("exists", j, a.term)
            if sort == (TL.LOC, TL.LOC):
                for i in range(1, size - 1):
                    for r in self.pool(sort, i):
                        for x in self.pool(TL.LOCS, size - 1 - i):
                            yield TL.mk("reach", r.term, x.term, TL.INF_STEPS)
                            yield TL.mk("extreme", "min", x.term, r.term)
                            yield TL.mk("extreme", "max", x.term, r.term)
            if size >= 3:
                for a in self.pool(sort, size - 2):
                    if a.term.op == "without":
                        continue
                    for x in placements:
                        yield TL.mk("without", x, a.term)


# ------------------------------------------------------------------------------------------------------------------------------------------ the scorer

class Scorer:
    """THE TESTS (ii) read for a candidate -> its citations; the pruning of (4)."""

    def __init__(self, dom: TL.Domain, reader, readings: Readings, defs: TL.Defs):
        self.dom, self.reader, self.R, self.defs = dom, reader, readings, defs
        self.static = set(dom.prog.static)
        self.failed: dict = {}                    # test name -> [(term, mode)]: mode 'small' (missed a location) / 'large' (held an extra one)
        self.pruned = 0
        self.tested = 0
        self.reset()

    def reset(self) -> None:
        self.ctx = [TL.Ctx(self.dom, self.reader, x, side, self.defs) for x, side in self.R.states]
        self._set_ctx: dict = {}

    def _value(self, c: Cand, si: int):
        if not (c.rels - self.static):
            return self.ctx[0].ev(c.term)
        return self.ctx[si].ev(c.term)

    def _pruned(self, c: Cand, test: str) -> bool:
        for t, mode in self.failed.get(test, ()):
            if mode == "small" and TL.implies(c.term, t):
                return True
            if mode == "large" and TL.implies(t, c.term):
                return True
        return False

    def _fail(self, c: Cand, mode: str, test: str) -> bool:
        self.failed.setdefault(test, []).append((c.term, mode))
        return False

    def generality(self, c: Cand, test: Test) -> int:
        """How many rows the candidate holds over the test's states (its readings' states, its flips' states before and after, its set's states)."""
        if test.kind == "set":
            ctxs = self._set_ctx[test.name]
            return sum(len(ctx.ev(c.term)) for ctx in ctxs)
        idx = sorted({r[0] for r in test.data} if test.kind == "readings" else {i for f in test.data[0] for i in f[:2]} | {r[0] for r in test.data[1]})
        return sum(len(self._value(c, i)) for i in idx)

    def cites(self, c: Cand, test: Test) -> bool:
        """Does the test cite the candidate (see the docstring)?"""
        if test.rel is not None and test.rel in c.rels:
            return False
        sort = c.term.sort
        if not isinstance(sort, tuple):
            return False
        nloc = sum(1 for s in sort if s == TL.LOC)
        if test.kind == "readings" and not (nloc == 3 and sort[-3:] == (TL.LOC,) * 3):
            return False
        if test.kind == "flips" and not (nloc == 2 and sort[-2:] == (TL.LOC, TL.LOC)):
            return False
        if test.kind == "set" and sort != TL.LOCS:
            return False
        if self._pruned(c, test.name):
            self.pruned += 1
            return False
        self.tested += 1
        if test.kind == "readings":
            cache: dict = {}
            dyn = bool(c.rels - self.static)
            for si, _rel, a, b, mid in test.data:
                k = si if dyn else 0
                v = cache.get(k)
                if v is None:
                    v = cache[k] = {}
                    for row in self._value(c, si):
                        v.setdefault((row[-3], row[-2]), set()).add(row[-1])
                got = v.get((a, b), set())
                if got != mid:
                    return self._fail(c, "small" if mid - got else "large", test.name)
            return True
        if test.kind == "flips":
            fs, tr = test.data
            pairs = lambda si: {(r[-2], r[-1]) for r in self._value(c, si)}
            for si, sj, _rel, a, b, _leaf in fs:
                if (a, b) not in pairs(sj):
                    return self._fail(c, "small", test.name)
                if (a, b) in pairs(si):
                    return self._fail(c, "large", test.name)
            for si, _rel, a, b, _mid in tr:
                if (a, b) in pairs(si):
                    return self._fail(c, "large", test.name)
            return True
        states, vals = test.data
        ctxs = self._set_ctx.get(test.name)
        if ctxs is None:
            ctxs = self._set_ctx[test.name] = [TL.Ctx(self.dom, self.reader, x, side, self.defs) for x, side in states]
        for ctx, want in zip(ctxs, vals):
            got = {r[0] for r in ctx.ev(c.term)}
            if got != want:
                return self._fail(c, "small" if want - got else "large", test.name)
        return True


# ------------------------------------------------------------------------------------------------------------------------------------------ the levels

def _readings(t: Test) -> int:
    """How many readings a test holds (the uses a relation it cites gains): its readings, its flips, its set's states."""
    return len(t.data) if t.kind == "readings" else len(t.data[0])


def mine(dom: TL.Domain, reader, sample: list, readings: Readings, tests: list, fuse: int, log=print, max_levels: int = 6) -> dict:
    """THE GENERATOR'S LEVELS (see the docstring) -> {accepted: [(name, term, size, the tests citing it, level)], levels, tried, tested, pruned, dropped, defs}."""
    defs = TL.Defs()
    sc = Scorer(dom, reader, readings, defs)
    by_name = {t.name: t for t in tests}
    acc = []
    explained: set = set()
    levels = []
    tried = 0
    dropped = Counter()
    t0 = time.monotonic()
    for level in range(1, max_levels + 1):
        ctxs = [TL.Ctx(dom, reader, x, side, defs) for x, side in sample]
        sc.reset()
        open_tests = [t for t in tests if t.name not in explained]
        if not open_tests:
            break
        shared: dict = {}                                       # one search per (alphabet, sort): the tests that read the same base share its pools
        for t in open_tests:
            shared.setdefault((t.alphabet, t.sort), Generator(dom, reader, ctxs, defs, t.alphabet, t.sort))
        gens = {t.name: shared[(t.alphabet, t.sort)] for t in open_tests}
        hits: list = []
        stop = None
        for size in range(1, fuse + 1):
            new_of = {}
            for key, g in shared.items():
                new_of[key] = g.size1() if size == 1 else g.grow(size)
            for t in open_tests:
                for c in new_of[(t.alphabet, t.sort)]:
                    if c.term.op in ("lit", "ref"):
                        continue                                # a literal names a relation of the spec, a name an accepted one: no object to mine
                    if sc.cites(c, t):
                        hits.append((t, c))
            log(f"OBS objects level {level} size {size}: candidates " + ", ".join(f"{t.name} {sum(len(v.get(size, ())) for v in gens[t.name].pools.values())}"
                                                                   for t in open_tests) + f"; cited {len(hits)}; {time.monotonic() - t0:.1f} s")
            if hits:
                stop = size
                break
        for g in shared.values():
            tried += g.tried
            dropped.update(g.dropped)
        if not hits:
            levels.append(dict(level=level, size=None, accepted=[]))
            log(f"OBS objects level {level}: accepted 0 (to the fuse {fuse}); the levels end")
            break
        kept: dict = {}
        for t, c in hits:                                         # per test the cheapest, then THE MOST GENERAL (the most rows on the test's states: the version
            g = sc.generality(c, t)                               # space's general boundary -- the explanation that commits to least), then the first in print
            prev = kept.get(t.name)                               # order; one relation may explain several tests
            if prev is None or (c.size, -g, c.term.sexpr()) < (prev[0].size, -prev[1], prev[0].term.sexpr()):
                kept[t.name] = (c, g)
        kept = {k: v[0] for k, v in kept.items()}
        names = {}
        for tname, c in sorted(kept.items()):
            key = c.term.sexpr()
            if key not in names:
                names[key] = (f"mined_{len(acc) + len(names) + 1}", c, [])
            names[key][2].append(tname)
            explained.add(tname)
        for key, (name, c, tl) in sorted(names.items(), key=lambda kv: kv[1][0]):
            defs.define(name, c.term)
            acc.append(dict(name=name, term=c.term, size=c.size, tests=sorted(tl), level=level, uses={x: _readings(by_name[x]) for x in tl}))
        sc.reset()
        for t in tests:                                           # USES ACCRUE: an accepted relation that explains a test still open is cited by it (no new name)
            if t.name in explained:
                continue
            for a in acc:
                ref = defs.ref(a["name"])
                if sc.cites(Cand(ref, (), 1, ref.rels(dom, defs)), t):
                    a["tests"] = sorted(set(a["tests"]) | {t.name})
                    a["uses"][t.name] = _readings(t)
                    explained.add(t.name)
                    log(f"OBS objects level {level}: {a['name']} explains {t.name} too")
                    break
        levels.append(dict(level=level, size=stop, accepted=[a["name"] for a in acc if a["level"] == level]))
        log(f"OBS objects level {level}: accepted {sum(a['level'] == level for a in acc)} at size {stop}" + "".join(
            f"; {a['name']} := {a['term'].sexpr()} (size {a['size']}, cited by {', '.join(a['tests'])})" for a in acc if a["level"] == level)
            + f"; {time.monotonic() - t0:.1f} s")
    return dict(accepted=acc, levels=levels, tried=tried, tested=sc.tested, pruned=sc.pruned, dropped=dict(dropped), defs=defs, explained=sorted(explained),
                open=sorted(t.name for t in tests if t.name not in explained), secs=round(time.monotonic() - t0, 1))
