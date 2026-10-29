"""PLAN 76.4c (a) ONE TERM LANGUAGE WITH TYPED COMBINATORS (docs/ANALYST_PLAN.md 76.4; plan 75.3: "the miner's atoms as terms with typed combinators: sets of
locations, reach, counts, per-axis distances, min over units, who moves; the generator proposes terms priced by size"). GENERIC: this module names no game object
-- no relation, no constant and no word of a game appears in its code. Every relation a term reads is a relation of the RULE SPEC, named by the ONTOLOGY SHELF
(ontology.py: one name, its bindings) and read on a state through a READER (the game's binding: `GDLReader` here reads any GDL description by its interpreter, the
oracle; a game's fast reader is its binding's, certified equal to it on the guard positions). The language's own sorts are found in the spec by ROLE, never by name:

  location   the shelf's OBJECT kind whose sort is a PRODUCT of the spec's sorts (its components, each ordered by the spec's table of successors over it)
  side       the spec's roles (the GDL keyword `role`)
  unit       the sort of the PLACEMENT fluent's last slot (the fluent of the state that puts a (side, unit) on a location)
  locations  a set of locations: a relation of one location slot          units   a relation of (location, unit): units placed
  count      a whole number                                               bool    a truth value
  and every other sort of the spec as an opaque, finite slot sort (its declared members).

A relation's SORT is the tuple of its slot sorts: the spec's signature with each run of the location's components read as one location slot.

THE CONSTRUCTORS (the s-expression each prints as; `?` a free slot):
  (R a1 .. an)            a LITERAL of the spec's relation R: each slot free (`?`), a constant of its sort, or a side term (`us`, `them`, `mover`); its sort the free slots
  us / them / mover       the side the term is read for, its opponent (the spec's table of opponents), the side to move (the spec's fluent of whose turn it is)
  (not A) (and A B) (or A B) (minus A B)    set operations (`not` within the product of A's slot domains)
  (exists j A)            slot j projected away (there is some value there)
  (closure A)             A of sort (P.., location, location): its transitive closure in the last two slots, the leading slots held
  (reach R X) (reach R X k)    the locations reached from the set X by at most k steps of R (location, location), X included (no k: any number)
  (restrict A S)          the rows of A whose last location is in the set S
  (compose A B)           the rows (P.., a, b) with some c: A (P.., a, c) and B (P.., c, b)
  (without X A)           A read on the state with the units on the locations X lifted (the placement fluent's rows there removed)
  (count A) (wcount j V A)     how many rows; the sum over A's rows of the weight V gives the row's slot j (V a VALUATION's worths, by the slot sort's members)
  (dist R A B)            the fewest steps of R (location, location) from a location of A to one of B (0 when they share one; 0 when either is empty)
  (axisdist c A B)        the fewest successor steps along the location's component c between a location of A and one of B (0 when either is empty)
  (axisends c)            the locations whose component c is an end of its successor chain
  (extreme min|max S R)   the locations of S with the fewest / the most R-successors
  all                     every location           (area S)   the product of S's spans along the components
  (first A)               A of sort (location, X): the location of A's first row, X by its sort's declared order, then the location's order
  (sum a b) (product a b) (sub a b)   numbers (sub: a minus b, signed)
  (ind b) (eq a b)        1 when b holds, else 0; two side terms equal
  (value A)               A of one row of an ordered sort: its member's place in the sort's declared order
  (lex a b ..)            the lexicographic tuple
  NAME                    a DEFINED relation (`define name := term`: `Defs`), priced 1 where it is used

Each term PRINTS as its s-expression (`sexpr`, and `parse` reads it back), has its READS (the spec relations it reads, `rels`; through the shelf, the SUPPLIED base
kinds those relations read, `base_reads`), and is PRICED BY SIZE (`size`: every constructor and every literal one; a side term or a constant inside a literal none;
a defined name one). `implies(a, b)` is the term language's IMPLICATION (a partial order computed on the syntax: every row of `a` is a row of `b` on every state),
read by the generator's pruning (plan 76 borrowing (4): a candidate violated at a position prunes every candidate it implies there).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from catspace.incubator.analyst.discovery import gdl as GDL

LOC, SIDE, UNIT, COUNT, BOOL = "location", "side", "unit", "count", "bool"
LOCS = (LOC,)
INF_STEPS = None                       # (reach R X) with no bound: any number of steps


class TermError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------------------------------------------------ the domain

class Domain:
    """The language's sorts read off a spec and the ontology shelf (see the module's docstring): the location (its components, their successor tables, every location
    in order -- the first component fastest), the side sort, the placement fluent, the unit sort, whose turn, the opponents, every other sort's members; each
    relation's sort (its signature with the location's components grouped)."""

    def __init__(self, prog: GDL.Program, shelf: dict):
        self.prog = prog
        prods = [e for e in shelf.values() if e["kind"] == "OBJECT" and isinstance(e["sort"], list)]
        if len(prods) != 1:
            raise TermError(f"the shelf must hold exactly one OBJECT kind whose sort is a product (the location): it holds {len(prods)}")
        self.location_kind = prods[0]["name"]
        self.comps = tuple(prods[0]["sort"])
        self.members = {s: [m for m in ms if isinstance(m, str)] for s, ms in prog.sorts.items()}
        self.succ = {}
        self.chain = {}
        for c in self.comps:
            tabs = [r for r in sorted(prog.signatures) if tuple(prog.signatures[r]) == (c, c) and r in prog.static and all(not cl.body for cl in prog.rules_of.get(r, ()))]
            if len(tabs) != 1:
                raise TermError(f"the component {c} must have exactly one successor table in the spec (a table of facts of signature ({c}, {c})): {tabs}")
            self.succ[c] = tabs[0]
            pairs = dict(prog.static[tabs[0]].facts)
            first = [m for m in self.members[c] if m not in set(pairs.values())]
            order = [first[0]]
            while order[-1] in pairs:
                order.append(pairs[order[-1]])
            if sorted(order) != sorted(self.members[c]):
                raise TermError(f"the successor table {tabs[0]} is not one chain over the sort {c}")
            self.chain[c] = order
        self.locs = [tuple(reversed(t)) for t in itertools.product(*[self.chain[c] for c in reversed(self.comps)])]
        self.index = {t: i for i, t in enumerate(self.locs)}
        self.pos = [tuple(self.chain[c].index(t[j]) for j, c in enumerate(self.comps)) for t in self.locs]
        roles = list(prog.roles())
        sides = {s for s, ms in self.members.items() if set(roles) and set(ms) == set(roles)}
        if len(sides) != 1:
            raise TermError(f"the spec's roles {roles} must be the members of exactly one sort")
        self.side_sort = sides.pop()
        self.roles = roles
        n = len(self.comps)
        fl = [r for r in sorted(prog.signatures) if self._is_fluent(r) and tuple(prog.signatures[r][:n]) == self.comps and len(prog.signatures[r]) == n + 2
              and prog.signatures[r][n] == self.side_sort]
        if len(fl) != 1:
            raise TermError(f"the spec must have exactly one placement fluent (location, side, unit): {fl}")
        self.placement = GDL.FLAT[0] + "." + fl[0]
        self.unit_sort = prog.signatures[fl[0]][n + 1]
        turn = [r for r in sorted(prog.signatures) if self._is_fluent(r) and tuple(prog.signatures[r]) == (self.side_sort,)]
        if len(turn) != 1:
            raise TermError(f"the spec must have exactly one fluent of one side (whose turn): {turn}")
        self.turn = GDL.FLAT[0] + "." + turn[0]
        opp = [r for r in sorted(prog.signatures) if tuple(prog.signatures[r]) == (self.side_sort, self.side_sort) and r in prog.static]
        if len(opp) != 1:
            raise TermError(f"the spec must have exactly one static relation of two sides (the opponents): {opp}")
        self.other = dict(prog.static[opp[0]].facts)
        self.other_rel = opp[0]
        self._slots: dict = {}

    def _is_fluent(self, rel: str) -> bool:
        return any(c.head[0] == GDL.FLAT[1] + "." + rel for c in self.prog.clauses) or any(r == GDL.FLAT[0] + "." + rel for r in self.prog.nodes)

    def sig(self, rel: str) -> tuple:
        """The spec's signature of a relation (its `true.` form reads the fluent's)."""
        s = self.prog.signature(rel)
        if s is None:
            raise TermError(f"{rel} is no relation of the spec")
        return tuple(s)

    def slots(self, rel: str) -> tuple:
        """THE RELATION'S SLOTS: ((sort, first argument index), ...) -- a run of the location's components one slot of sort `location`."""
        got = self._slots.get(rel)
        if got is None:
            sig, out, i, n = self.sig(rel), [], 0, len(self.comps)
            while i < len(sig):
                if tuple(sig[i:i + n]) == self.comps:
                    out.append((LOC, i))
                    i += n
                else:
                    out.append((self.lang_sort(sig[i]), i))
                    i += 1
            got = self._slots[rel] = tuple(out)
        return got

    def lang_sort(self, spec_sort: str) -> str:
        return SIDE if spec_sort == self.side_sort else UNIT if spec_sort == self.unit_sort else spec_sort

    def spec_sort(self, lang_sort: str) -> str:
        return self.side_sort if lang_sort == SIDE else self.unit_sort if lang_sort == UNIT else lang_sort

    def domain(self, sort: str) -> list:
        if sort == LOC:
            return list(range(len(self.locs)))
        return list(self.members[self.spec_sort(sort)])

    def group(self, rel: str, row: tuple) -> tuple:
        """A spec row of `rel` in the language's form: each location one index."""
        n = len(self.comps)
        out = []
        for s, i in self.slots(rel):
            out.append(self.index[tuple(row[i:i + n])] if s == LOC else row[i])
        return tuple(out)

    def ungroup(self, loc: int) -> tuple:
        return self.locs[loc]


# ------------------------------------------------------------------------------------------------------------------------------------------ readers

class GDLReader:
    """THE ORACLE READER: any GDL description read by its interpreter (gdl.py) on a state given as the spec's facts (a frozenset of (relation, args)); `lift`
    removes the placement fluent's facts at the given locations."""

    def __init__(self, prog: GDL.Program, inputs=()):
        self.prog = prog
        self.inputs = tuple(inputs)
        self._db: dict = {}

    def key(self, x):
        return x

    def rows(self, x, rel: str):
        db = self._db.get(x)
        if db is None:
            if len(self._db) > 64:
                self._db.clear()
            db = self._db[x] = self.prog.run(x, inputs=self.inputs)
        return db.facts(rel)

    def lift(self, x, locs, dom: Domain):
        n = len(dom.comps)
        drop = {dom.ungroup(i) for i in locs}
        return frozenset(f for f in x if not (f[0] == dom.placement and tuple(f[1][:n]) in drop))


# ------------------------------------------------------------------------------------------------------------------------------------------ the terms

STATE = GDL.FLAT[0] + "."              # the keyword of the state's fluents (`true.<fluent>`): a literal of a fluent prints as the fluent


def _shown(rel: str) -> str:
    return rel[len(STATE):] if rel.startswith(STATE) else rel


@dataclass(frozen=True)
class T:
    """One term: `op` the constructor, `args` its sub-terms and parameters (a relation name, a slot, a bound, a valuation's name and worths), `sort` its sort."""
    op: str
    args: tuple
    sort: object = field(compare=False, hash=False, default=None)

    # ---- printing
    def sexpr(self) -> str:
        o, a = self.op, self.args
        if o == "lit":
            return "(" + " ".join([_shown(a[0])] + [x.sexpr() if isinstance(x, T) else ("?" if x is None else str(x)) for x in a[1]]) + ")"
        if o in ("us", "them", "mover", "all"):
            return o
        if o == "const":
            return str(a[1])
        if o == "ref":
            return a[0]
        if o == "reach" and a[2] is not INF_STEPS:
            return f"(reach {a[0].sexpr()} {a[1].sexpr()} {a[2]})"
        if o == "reach":
            return f"(reach {a[0].sexpr()} {a[1].sexpr()})"
        if o in ("exists",):
            return f"(exists {a[0]} {a[1].sexpr()})"
        if o == "wcount":
            return f"(wcount {a[0]} {a[1]} {a[3].sexpr()})"
        if o in ("axisdist",):
            return f"(axisdist {a[0]} {a[1].sexpr()} {a[2].sexpr()})"
        if o == "axisends":
            return f"(axisends {a[0]})"
        if o == "extreme":
            return f"(extreme {a[0]} {a[1].sexpr()} {a[2].sexpr()})"
        return "(" + " ".join([o] + [x.sexpr() if isinstance(x, T) else str(x) for x in a]) + ")"

    __str__ = sexpr

    # ---- pricing and reads
    def kids(self) -> tuple:
        if self.op == "lit":
            return tuple(x for x in self.args[1] if isinstance(x, T))
        return tuple(x for x in self.args if isinstance(x, T))

    @property
    def size(self) -> int:
        """THE PRICE: every constructor and every literal one; a side term or a constant in a literal's slot none; a defined name one."""
        if self.op in ("us", "them", "mover", "const"):
            return 0
        if self.op in ("lit", "ref"):
            return 1
        return 1 + sum(k.size for k in self.kids())

    def rels(self, dom: "Domain | None" = None, defs: "Defs | None" = None) -> frozenset:
        """THE READS: the spec relations the term reads (a defined name's through its body; `without` the placement fluent; `mover` whose turn; `them` the
        opponents; `axisdist` / `axisends` the component's successor table; `all` and `not` the location's object)."""
        out = set()
        stack = [self]
        while stack:
            t = stack.pop()
            if t.op == "lit":
                out.add(t.args[0])
            elif t.op == "ref":
                if defs is None or t.args[0] not in defs.body:
                    raise TermError(f"the defined name {t.args[0]} has no body here")
                stack.append(defs.body[t.args[0]])
            elif dom is not None:
                if t.op == "without":
                    out.add(dom.placement)
                elif t.op == "mover":
                    out.add(dom.turn)
                elif t.op == "them":
                    out.add(dom.other_rel)
                elif t.op in ("axisdist", "axisends"):
                    out.add(dom.succ[t.args[0]])
            stack.extend(t.kids())
        return frozenset(out)


# the constructors' result sorts
def _rel_sort(dom: Domain, rel: str, pattern) -> tuple:
    sl = dom.slots(rel)
    if len(pattern) != len(sl):
        raise TermError(f"{rel} has {len(sl)} slots, the literal gives {len(pattern)}")
    out = []
    for (s, _i), p in zip(sl, pattern):
        if p is None:
            out.append(s)
        elif isinstance(p, T):
            if p.sort != s:
                raise TermError(f"{rel}: a {p.sort} term in a {s} slot")
        elif s == LOC or p not in dom.domain(s):
            raise TermError(f"{rel}: {p!r} is no member of the slot sort {s}")
    return tuple(out) if out else BOOL


def lit(dom: Domain, rel: str, *pattern) -> T:
    """A literal: each slot None (free), a constant of its sort or a side term."""
    pat = tuple(const(dom, s, p) if (p is not None and not isinstance(p, T)) else p for (s, _i), p in zip(dom.slots(rel), pattern))
    return T("lit", (rel, pat), _rel_sort(dom, rel, pat))


def const(dom: Domain, sort: str, v) -> T:
    if v not in dom.domain(sort):
        raise TermError(f"{v!r} is no member of {sort}")
    return T("const", (sort, v), sort)


US, THEM, MOVER, ALL = T("us", (), SIDE), T("them", (), SIDE), T("mover", (), SIDE), T("all", (), LOCS)


def _rel(t: T, what: str) -> tuple:
    if not isinstance(t.sort, tuple):
        raise TermError(f"{what}: {t.sexpr()} is no relation")
    return t.sort


def _bin_loc(t: T, what: str) -> tuple:
    s = _rel(t, what)
    if len(s) < 2 or s[-1] != LOC or s[-2] != LOC:
        raise TermError(f"{what}: {t.sexpr()} is not of sort (.., location, location)")
    return s


def mk(op: str, *args) -> T:
    """A term of a constructor, its sort checked (TermError on a sort mismatch)."""
    if op in ("and", "or", "minus"):
        a, b = args
        if _rel(a, op) != _rel(b, op):
            raise TermError(f"{op}: {a.sexpr()} and {b.sexpr()} are of two sorts")
        return T(op, args, a.sort)
    if op == "not":
        return T(op, args, _rel(args[0], op))
    if op == "exists":
        j, a = args
        s = _rel(a, op)
        if not 0 <= j < len(s) or len(s) < 2:
            raise TermError(f"exists: slot {j} of {a.sexpr()}")
        return T(op, args, s[:j] + s[j + 1:])
    if op == "closure":
        return T(op, args, _bin_loc(args[0], op))
    if op == "reach":
        r, x, k = args
        if _bin_loc(r, op) != (LOC, LOC) or _rel(x, op) != LOCS:
            raise TermError(f"reach: ({r.sexpr()} {x.sexpr()}) is not (location, location) over a set of locations")
        return T(op, args, LOCS)
    if op == "restrict":
        a, s = args
        _bin_loc(a, op)
        if _rel(s, op) != LOCS:
            raise TermError(f"restrict: {s.sexpr()} is no set of locations")
        return T(op, args, a.sort)
    if op == "compose":
        a, b = args
        if _bin_loc(a, op) != _bin_loc(b, op):
            raise TermError(f"compose: {a.sexpr()} and {b.sexpr()} are of two sorts")
        return T(op, args, a.sort)
    if op == "without":
        x, a = args
        if _rel(x, op) != LOCS:
            raise TermError(f"without: {x.sexpr()} is no set of locations")
        return T(op, args, a.sort)
    if op == "count":
        _rel(args[0], op)
        return T(op, args, COUNT)
    if op == "wcount":
        j, name, worth, a = args
        s = _rel(a, op)
        if not 0 <= j < len(s) or s[j] == LOC:
            raise TermError(f"wcount: slot {j} of {a.sexpr()} carries no weighed sort")
        return T(op, (j, name, tuple(sorted(dict(worth).items())), a), COUNT)
    if op == "dist":
        r, a, b = args
        if _bin_loc(r, op) != (LOC, LOC) or _rel(a, op) != LOCS or _rel(b, op) != LOCS:
            raise TermError("dist: (location, location) between two sets of locations")
        return T(op, args, COUNT)
    if op == "axisdist":
        c, a, b = args
        if _rel(a, op) != LOCS or _rel(b, op) != LOCS:
            raise TermError("axisdist: between two sets of locations")
        return T(op, args, COUNT)
    if op == "axisends":
        return T(op, args, LOCS)
    if op == "extreme":
        w, s, r = args
        if w not in ("min", "max") or _rel(s, op) != LOCS or _bin_loc(r, op) != (LOC, LOC):
            raise TermError("extreme: min | max over a set of locations by a (location, location) relation")
        return T(op, args, LOCS)
    if op == "area":
        if _rel(args[0], op) != LOCS:
            raise TermError("area: of a set of locations")
        return T(op, args, COUNT)
    if op == "first":
        s = _rel(args[0], op)
        if len(s) != 2 or s[0] != LOC or s[1] == LOC:
            raise TermError("first: of a relation (location, X)")
        return T(op, args, LOCS)
    if op in ("sum", "product", "sub"):
        a, b = args
        if a.sort != COUNT or b.sort != COUNT:
            raise TermError(f"{op}: of two counts")
        return T(op, args, COUNT)
    if op == "ind":
        if args[0].sort != BOOL:
            raise TermError("ind: of a truth value")
        return T(op, args, COUNT)
    if op == "eq":
        a, b = args
        if a.sort != b.sort or a.sort not in (SIDE, UNIT):
            raise TermError("eq: of two side terms")
        return T(op, args, BOOL)
    if op == "nonempty":
        _rel(args[0], op)
        return T(op, args, BOOL)
    if op == "notb":
        if args[0].sort != BOOL:
            raise TermError("notb: of a truth value")
        return T(op, args, BOOL)
    if op == "value":
        s = _rel(args[0], op)
        if len(s) != 1 or s[0] == LOC:
            raise TermError("value: of a relation of one ordered slot")
        return T(op, args, COUNT)
    if op == "lex":
        if any(a.sort != COUNT for a in args):
            raise TermError("lex: of counts")
        return T(op, args, ("lex",) + (COUNT,) * len(args))
    raise TermError(f"no constructor {op}")


CONSTRUCTORS = ("lit", "us", "them", "mover", "not", "and", "or", "minus", "exists", "closure", "reach", "restrict", "compose", "without", "count", "wcount",
                "dist", "axisdist", "axisends", "extreme", "all", "area", "first", "sum", "product", "sub", "ind", "eq", "nonempty", "notb", "value", "lex", "ref")
SORTS = (LOC, SIDE, UNIT, "locations", "units", COUNT, BOOL)


class Defs:
    """THE DEFINED RELATIONS (`define name := term`): name -> its body; a defined name is a term of its body's sort, priced 1 where it is used."""

    def __init__(self):
        self.body: dict = {}

    def define(self, name: str, body: T) -> T:
        if name in self.body and self.body[name] != body:
            raise TermError(f"{name} is defined already as {self.body[name].sexpr()}")
        self.body[name] = body
        return self.ref(name)

    def ref(self, name: str) -> T:
        return T("ref", (name,), self.body[name].sort)


# ------------------------------------------------------------------------------------------------------------------------------------------ parsing

def parse(text: str, dom: Domain, defs: "Defs | None" = None, valuations: "dict | None" = None) -> T:
    """The term an s-expression prints (`T.sexpr`), read back; `valuations`: name -> worths for `wcount`."""
    toks = GDL.tokenize(text)
    pos = [0]

    def read():
        t = toks[pos[0]]
        pos[0] += 1
        if t == "(":
            out = []
            while toks[pos[0]] != ")":
                out.append(read())
            pos[0] += 1
            return out
        return t
    return _build_term(read(), dom, defs, valuations or {})


def _build_term(x, dom: Domain, defs, vals) -> T:
    if isinstance(x, str):
        if x == "us":
            return US
        if x == "them":
            return THEM
        if x == "mover":
            return MOVER
        if x == "all":
            return ALL
        if defs is not None and x in defs.body:
            return defs.ref(x)
        raise TermError(f"unknown symbol {x}")
    head = x[0]
    b = lambda y: _build_term(y, dom, defs, vals)
    if head not in dom.prog.nodes and STATE + head in dom.prog.nodes:
        head = STATE + head                                 # a fluent of the state, printed without the keyword
    if head in dom.prog.nodes and head not in CONSTRUCTORS:
        pat = []
        for (s, _i), y in zip(dom.slots(head), x[1:]):
            if y == "?":
                pat.append(None)
            elif y in ("us", "them", "mover"):
                pat.append(b(y))
            else:
                pat.append(y)
        return lit(dom, head, *pat)
    if head == "reach":
        return mk("reach", b(x[1]), b(x[2]), int(x[3]) if len(x) > 3 else INF_STEPS)
    if head == "exists":
        return mk("exists", int(x[1]), b(x[2]))
    if head == "wcount":
        return mk("wcount", int(x[1]), x[2], vals[x[2]], b(x[3]))
    if head == "axisdist":
        return mk("axisdist", x[1], b(x[2]), b(x[3]))
    if head == "axisends":
        return mk("axisends", x[1])
    if head == "extreme":
        return mk("extreme", x[1], b(x[2]), b(x[3]))
    return mk(head, *[b(y) for y in x[1:]])


# ------------------------------------------------------------------------------------------------------------------------------------------ evaluation

class Ctx:
    """One state read for one side: the reader's rows (cached per relation), the term values (cached per term); `without` reads a child context on the lifted state."""

    def __init__(self, dom: Domain, reader, x, us: str, defs: "Defs | None" = None):
        self.dom, self.reader, self.x, self.us = dom, reader, x, us
        self.defs = defs
        self._rows: dict = {}
        self._val: dict = {}
        self._kids: dict = {}

    def rows(self, rel: str) -> frozenset:
        got = self._rows.get(rel)
        if got is None:
            if rel in self.dom.prog.static:
                src = self.dom.prog.static[rel].facts
            else:
                src = self.reader.rows(self.x, rel)
            got = self._rows[rel] = frozenset(self.dom.group(rel, tuple(r)) for r in src)
        return got

    def side(self, t: T) -> str:
        if t.op == "us":
            return self.us
        if t.op == "them":
            return self.dom.other[self.us]
        if t.op == "mover":
            rows = self.rows(self.dom.turn)
            return min(rows)[0]
        if t.op == "const":
            return t.args[1]
        raise TermError(f"{t.sexpr()} is no side term")

    def on_lifted(self, locs: frozenset) -> "Ctx":
        k = self._kids.get(locs)
        if k is None:
            k = self._kids[locs] = Ctx(self.dom, self.reader, self.reader.lift(self.x, sorted(locs), self.dom), self.us, self.defs)
        return k

    def ev(self, t: T):
        got = self._val.get(t)
        if got is None and t not in self._val:
            got = self._val[t] = _EVAL[t.op](self, t)
        return got


def _domain_product(ctx: Ctx, sort: tuple):
    return itertools.product(*[ctx.dom.domain(s) for s in sort])


def _lit(ctx: Ctx, t: T):
    rel, pat = t.args
    rows = ctx.rows(rel)
    want = [(j, ctx.side(p) if isinstance(p, T) else p) for j, p in enumerate(pat) if p is not None]
    free = [j for j, p in enumerate(pat) if p is None]
    got = frozenset(tuple(r[j] for j in free) for r in rows if all(r[j] == v for j, v in want))
    return bool(got) if t.sort == BOOL else got


def _adj(rel: frozenset, n: int) -> dict:
    """(P.., a, b) rows -> {P: [the set of b per a]}."""
    out: dict = {}
    for r in rel:
        p = r[:-2]
        m = out.get(p)
        if m is None:
            m = out[p] = [set() for _ in range(n)]
        m[r[-2]].add(r[-1])
    return out


def _closure(ctx: Ctx, t: T):
    a = ctx.ev(t.args[0])
    n = len(ctx.dom.locs)
    out = set()
    for p, m in _adj(a, n).items():
        for s in range(n):
            if not m[s]:
                continue
            reached, todo = set(), list(m[s])
            while todo:
                y = todo.pop()
                if y in reached:
                    continue
                reached.add(y)
                todo.extend(m[y])
            out.update(p + (s, y) for y in reached)
    return frozenset(out)


def _reach(ctx: Ctx, t: T):
    r, x, k = t.args
    m = _adj(ctx.ev(r), len(ctx.dom.locs)).get((), None)
    reached = {v[0] for v in ctx.ev(x)}
    ring = set(reached)
    steps = 0
    while ring and (k is INF_STEPS or steps < k) and m is not None:
        nxt = set()
        for y in ring:
            nxt |= m[y]
        ring = nxt - reached
        reached |= ring
        steps += 1
    return frozenset((v,) for v in reached)


def _not(ctx: Ctx, t: T):
    a = ctx.ev(t.args[0])
    return frozenset(r for r in _domain_product(ctx, t.sort) if r not in a)


def _exists(ctx: Ctx, t: T):
    j, a = t.args
    return frozenset(r[:j] + r[j + 1:] for r in ctx.ev(a))


def _restrict(ctx: Ctx, t: T):
    a, s = t.args
    keep = {v[0] for v in ctx.ev(s)}
    return frozenset(r for r in ctx.ev(a) if r[-1] in keep)


def _compose(ctx: Ctx, t: T):
    a, b = (ctx.ev(x) for x in t.args)
    by: dict = {}
    for r in b:
        by.setdefault(r[:-2] + (r[-2],), []).append(r[-1])
    out = set()
    for r in a:
        for y in by.get(r[:-2] + (r[-1],), ()):
            out.add(r[:-2] + (r[-2], y))
    return frozenset(out)


def _without(ctx: Ctx, t: T):
    x, a = t.args
    locs = frozenset(v[0] for v in ctx.ev(x))
    return ctx.on_lifted(locs).ev(a)


def _dist(ctx: Ctx, t: T):
    r, a, b = t.args
    A, B = {v[0] for v in ctx.ev(a)}, {v[0] for v in ctx.ev(b)}
    if not A or not B:
        return 0
    m = _adj(ctx.ev(r), len(ctx.dom.locs)).get((), [set() for _ in ctx.dom.locs])
    reached, ring, d = set(A), set(A), 0
    while not ring & B:
        nxt = set()
        for y in ring:
            nxt |= m[y]
        ring = nxt - reached
        if not ring:
            return len(ctx.dom.locs)              # unreachable: more steps than there are locations
        reached |= ring
        d += 1
    return d


def _axisdist(ctx: Ctx, t: T):
    c, a, b = t.args
    j = ctx.dom.comps.index(c)
    A, B = [ctx.dom.pos[v[0]][j] for v in ctx.ev(a)], [ctx.dom.pos[v[0]][j] for v in ctx.ev(b)]
    if not A or not B:
        return 0
    return min(abs(x - y) for x in A for y in B)


def _axisends(ctx: Ctx, t: T):
    j = ctx.dom.comps.index(t.args[0])
    last = len(ctx.dom.chain[t.args[0]]) - 1
    return frozenset((i,) for i, p in enumerate(ctx.dom.pos) if p[j] in (0, last))


def _extreme(ctx: Ctx, t: T):
    w, s, r = t.args
    S = [v[0] for v in ctx.ev(s)]
    if not S:
        return frozenset()
    deg: dict = {}
    for row in ctx.ev(r):
        deg[row[0]] = deg.get(row[0], 0) + 1
    vals = {x: deg.get(x, 0) for x in S}
    top = min(vals.values()) if w == "min" else max(vals.values())
    return frozenset((x,) for x, v in vals.items() if v == top)


def _area(ctx: Ctx, t: T):
    S = [ctx.dom.pos[v[0]] for v in ctx.ev(t.args[0])]
    if not S:
        return 0
    out = 1
    for j in range(len(ctx.dom.comps)):
        xs = [p[j] for p in S]
        out *= max(xs) - min(xs) + 1
    return out


def _first(ctx: Ctx, t: T):
    a = ctx.ev(t.args[0])
    if not a:
        return frozenset()
    order = ctx.dom.domain(t.args[0].sort[1])
    top = min(a, key=lambda r: (order.index(r[1]), r[0]))
    return frozenset({(top[0],)})


def _wcount(ctx: Ctx, t: T):
    j, _name, worth, a = t.args
    w = dict(worth)
    return sum(w.get(r[j], 0) for r in ctx.ev(a))


def _value(ctx: Ctx, t: T):
    a = ctx.ev(t.args[0])
    if len(a) != 1:
        raise TermError(f"value: {t.args[0].sexpr()} holds {len(a)} rows")
    order = ctx.dom.domain(t.args[0].sort[0])
    return order.index(min(a)[0])


_EVAL = {
    "lit": _lit,
    "ref": lambda c, t: c.ev(c.defs.body[t.args[0]]),
    "all": lambda c, t: frozenset((i,) for i in range(len(c.dom.locs))),
    "not": _not,
    "and": lambda c, t: c.ev(t.args[0]) & c.ev(t.args[1]),
    "or": lambda c, t: c.ev(t.args[0]) | c.ev(t.args[1]),
    "minus": lambda c, t: c.ev(t.args[0]) - c.ev(t.args[1]),
    "exists": _exists,
    "closure": _closure,
    "reach": _reach,
    "restrict": _restrict,
    "compose": _compose,
    "without": _without,
    "count": lambda c, t: len(c.ev(t.args[0])),
    "wcount": _wcount,
    "dist": _dist,
    "axisdist": _axisdist,
    "axisends": _axisends,
    "extreme": _extreme,
    "area": _area,
    "first": _first,
    "sum": lambda c, t: c.ev(t.args[0]) + c.ev(t.args[1]),
    "product": lambda c, t: c.ev(t.args[0]) * c.ev(t.args[1]),
    "sub": lambda c, t: c.ev(t.args[0]) - c.ev(t.args[1]),
    "ind": lambda c, t: int(bool(c.ev(t.args[0]))),
    "eq": lambda c, t: c.side(t.args[0]) == c.side(t.args[1]),
    "nonempty": lambda c, t: bool(c.ev(t.args[0])),
    "notb": lambda c, t: not c.ev(t.args[0]),
    "value": _value,
    "lex": lambda c, t: tuple(c.ev(a) for a in t.args),
}


def evaluate(t: T, dom: Domain, reader, x, us: str, defs: "Defs | None" = None):
    """The value of a term on the state `x` read for the side `us`: a relation as a frozenset of rows (a location its index in `dom.locs`), a count, a truth
    value or a tuple."""
    return Ctx(dom, reader, x, us, defs).ev(t)


# ------------------------------------------------------------------------------------------------------------------------------------------ implication

def implies(a: T, b: T) -> bool:
    """THE IMPLICATION the language computes on its syntax: True when every row of `a` is a row of `b` on every state (a sound, incomplete order): a term implies
    itself; (and A B) and (minus A B) and (restrict A S) imply A; A implies (or A B); A implies (closure A); (reach R X k) implies (reach R X k') for k <= k' (no k:
    the largest); and the order is closed under (and X _), (or X _), (closure X), (restrict X S), (compose X Y) on the left and (minus _ X) on the right."""
    if a == b:
        return True
    if not (isinstance(a.sort, tuple) and a.sort == b.sort):
        return False
    if a.op in ("and", "minus", "restrict") and implies(a.args[0], b):
        return True
    if a.op == "and" and implies(a.args[1], b):
        return True
    if b.op == "or" and (implies(a, b.args[0]) or implies(a, b.args[1])):
        return True
    if b.op == "closure" and implies(a, b.args[0]):
        return True
    if a.op == b.op == "reach" and a.args[0] == b.args[0] and a.args[1] == b.args[1]:
        ka, kb = a.args[2], b.args[2]
        return kb is INF_STEPS or (ka is not INF_STEPS and ka <= kb)
    if a.op == b.op and a.op in ("and", "or", "compose") and implies(a.args[0], b.args[0]) and implies(a.args[1], b.args[1]):
        return True
    if a.op == b.op == "restrict" and implies(a.args[0], b.args[0]) and implies(a.args[1], b.args[1]):
        return True
    if a.op == b.op == "closure" and implies(a.args[0], b.args[0]):
        return True
    if a.op == b.op == "minus" and implies(a.args[0], b.args[0]) and implies(b.args[1], a.args[1]):
        return True
    return False


def base_reads(t: T, dom: Domain, shelf: dict, defs: "Defs | None" = None) -> list:
    """The SUPPLIED base kinds of the shelf the term reads: each spec relation it reads mapped to its shelf entry (the entry binding that relation or fluent), the
    entry itself when it is SUPPLIED, else the entry's own `reads`."""
    by_rel = {}
    for e in shelf.values():
        g = e.get("gdl") or {}
        if "slot" in g or "constant" in g or "args" in g:
            continue
        if "relation" in g:
            by_rel[g["relation"]] = e
        elif "fluent" in g:
            by_rel[STATE + g["fluent"]] = e
    out = set()
    for r in t.rels(dom, defs):
        e = by_rel.get(r)
        if e is None:
            raise TermError(f"the spec relation {r} has no shelf entry")
        out |= {e["name"]} if e["origin"] == "SUPPLIED" else set(e["reads"])
    return sorted(out)
