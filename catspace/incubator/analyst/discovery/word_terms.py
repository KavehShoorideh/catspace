"""PLAN 78.35 (1) A WORD IS A TERM THAT COMPILES TO RULES. A small term language, written as s-expressions, over the relations of a description text. A term is compiled to rules of the
description's own language (a `signature` line and `(<= head body ..)` rules, with helper relations `<name>__1`, `<name>__2` ... where a sub-term needs its own rule), so the evaluator that
reads every other relation reads a word too: nothing of a word is evaluated in Python on any decision path. Generic: no object of any game is named here; October's term language of
plan 76.4c (terms.py) is the model for the meanings, and only the constructors a stratified rule language can say are kept.

THE CONSTRUCTORS
  (R a1 .. an)       a literal of the description's relation R. A slot is a variable `?x`, the anonymous `_` (some value, projected away) or a constant. A relation the description's state
                     holds (a FLUENT: it occurs in `(true (R ..))`, `(next (R ..))` or `(init (R ..))`) is read as `(true (R ..))`; any other relation is read as itself. Sides, units and the
                     like are ordinary variables or constants: there are no side terms here. A relation of no slot is written `(R)`.
  (and A B ..)       the join on shared variable names.
  (or A B ..)        the union; every branch has the same free variables (and the same sorts).
  (not A)            only INSIDE an `and`, where every variable of A is bound by a positive item of that `and` (safe negation).
  (distinct a b)     only inside an `and`, both sides bound there (a variable or a constant).
  (exists (?x ..) A) projection: the listed variables are not free in the result.
  (compose A B [m])  A has columns (P.., x, z), B has (P.., z, y), each of x, z, y a LOCATION of m slots: the result is (P.., x, y). The COLUMNS of a term are its free variables in order of first
                     occurrence. The result's free variables are A's prefix, A's x columns and B's y columns (they must be distinct names: rename a clash).
  (via A B [m])      the same with the middle location kept: (P.., x, y, z), z named as in A.
  (reach A k [m])    A has columns (P.., x, y): the pairs joined by AT LEAST ONE AND AT MOST k steps of A (1 <= k <= 8, UNROLLED into k helper relations). x itself is NOT included, so (x, x) is a
                     row only where a cycle of at most k steps returns to x. The result has A's free variables.
  (reach0 A k [m])   the same with x itself included: also the rows (x, x) for every location x that A mentions (in its first or last location column).
  (closure A [m])    the transitive closure of A (one or more steps) as a recursive rule; the result has A's free variables.
THE LOCATION'S WIDTH `m` is a PARAMETER, default 1: a location may be a PRODUCT of several slots (m = 2 for a location of two coordinates). It is not inferred from the signatures: a term written
for a one-slot location and a product location differ only by the trailing integer. The last 2m columns of A must be two locations of the same sorts (checked by `sort_of`).

PRICE `size(term)`: every constructor and every literal one (a word's name used in a literal is one literal; a constant or a variable inside a slot none).

  parse(text) -> term                         show(term) -> text (parse(show(t)) == t)
  free_vars(term) -> [?v ..]                  size(term) -> int
  read_description(text) -> Description       .signatures {relation: (sort, ..)}, .sorts {sort: [member ..]}, .fluents {relation}
  sort_of(term, signatures, sorts=None) -> (sort, ..) of the free variables; raises TermError on an unknown relation, a wrong arity, a sort clash, a constant outside its sort (when `sorts` is
                     given), unsafe negation, a malformed constructor
  compile(name, term, signatures, *, head_vars=None, fluents=(), sorts=None) -> rules text: `(signature name sort ..)`, the rules of the helpers, the rules of `name`. ADDITIVE ONLY. `head_vars`
                     picks and orders the head's variables from the free ones (the others are projected away)
  expand(term, registry) -> the term with every registered word's name replaced by its own term (down to relations the registry does not hold): the expansion in base words
  canonical(term) -> the term with its variables renamed ?v0 ?v1 .. by first occurrence (equality of words modulo the names of variables)
"""
from __future__ import annotations

import itertools
import re

from catspace.incubator.analyst.discovery import gdl as GDL

ARITY_MAX = 8                      # the evaluator's widest relation (rust/engine/gdl: ARITY_MAX)
K_MAX = 8                          # the most steps `reach` unrolls
KEYWORDS = ("and", "or", "not", "exists", "distinct", "compose", "via", "reach", "reach0", "closure")
_LEGAL_NAME = re.compile(r"[a-z][a-z0-9_]*$")
_LEGAL_VAR = re.compile(r"\?[A-Za-z0-9][A-Za-z0-9_]*$")


class TermError(ValueError):
    pass


# the terms: tuples ('lit', rel, args) ('and', items) ('or', items) ('not', t) ('exists', vars, t) ('distinct', a, b) ('compose', A, B, m) ('via', A, B, m) ('reach', A, k, m, zero) ('closure', A, m)

def lit(rel: str, *args: str) -> tuple:
    return ("lit", rel, tuple(args))


def is_var(a) -> bool:
    return isinstance(a, str) and a.startswith("?")


# ------------------------------------------------------------------------------------------------------------------------------------------ parse and show

def _int(x, what: str) -> int:
    if not isinstance(x, str) or not x.isdigit():
        raise TermError(f"{what} must be a whole number, not {x!r}")
    return int(x)


def _arg(x) -> str:
    if not isinstance(x, str):
        raise TermError(f"a slot is a variable, `_` or a constant, not {x!r}")
    if x.startswith("?") and not _LEGAL_VAR.match(x):
        raise TermError(f"{x!r} is not a legal variable (?name; names starting with ?_ are the compiler's)")
    return x


def _from(x) -> tuple:
    if isinstance(x, str):
        if x in KEYWORDS or x.startswith("?"):
            raise TermError(f"{x!r} is not a term")
        return ("lit", x, ())
    if not x:
        raise TermError("() is not a term")
    h, rest = x[0], x[1:]
    if not isinstance(h, str):
        raise TermError(f"a term starts with a name, not {h!r}")
    if h in ("and", "or"):
        if not rest:
            raise TermError(f"({h}) needs at least one item")
        return (h, tuple(_from(y) for y in rest))
    if h == "not":
        if len(rest) != 1:
            raise TermError("(not A) takes one term")
        return ("not", _from(rest[0]))
    if h == "exists":
        if len(rest) != 2 or not isinstance(rest[0], list) or not rest[0]:
            raise TermError("(exists (?x ..) A)")
        vs = tuple(_arg(v) for v in rest[0])
        if not all(is_var(v) for v in vs) or len(set(vs)) != len(vs):
            raise TermError(f"exists binds distinct variables, not {rest[0]}")
        return ("exists", vs, _from(rest[1]))
    if h == "distinct":
        if len(rest) != 2:
            raise TermError("(distinct a b) takes two slots")
        a, b = _arg(rest[0]), _arg(rest[1])
        if "_" in (a, b):
            raise TermError("(distinct ..) cannot take the anonymous slot")
        return ("distinct", a, b)
    if h in ("compose", "via"):
        if len(rest) not in (2, 3):
            raise TermError(f"({h} A B [m])")
        return (h, _from(rest[0]), _from(rest[1]), _int(rest[2], "m") if len(rest) == 3 else 1)
    if h in ("reach", "reach0"):
        if len(rest) not in (2, 3):
            raise TermError(f"({h} A k [m])")
        k = _int(rest[1], "k")
        if not 1 <= k <= K_MAX:
            raise TermError(f"k is at most {K_MAX} steps and at least 1, not {k}")
        return ("reach", _from(rest[0]), k, _int(rest[2], "m") if len(rest) == 3 else 1, h == "reach0")
    if h == "closure":
        if len(rest) not in (1, 2):
            raise TermError("(closure A [m])")
        return ("closure", _from(rest[0]), _int(rest[1], "m") if len(rest) == 2 else 1)
    return ("lit", h, tuple(_arg(y) for y in rest))


def parse(text: str) -> tuple:
    forms = GDL.parse(text)
    if len(forms) != 1:
        raise TermError(f"a term is one s-expression, not {len(forms)}: {text!r}")
    t = _from(forms[0])
    for m in _locations(t):
        if m < 1:
            raise TermError("a location has at least one slot")
    return t


def _locations(t):
    if t[0] in ("compose", "via"):
        yield t[3]
    elif t[0] == "reach":
        yield t[3]
    elif t[0] == "closure":
        yield t[2]
    for c in _children(t):
        yield from _locations(c)


def _children(t) -> tuple:
    k = t[0]
    if k in ("and", "or"):
        return t[1]
    if k in ("not",):
        return (t[1],)
    if k == "exists":
        return (t[2],)
    if k in ("compose", "via"):
        return (t[1], t[2])
    if k in ("reach", "closure"):
        return (t[1],)
    return ()


def show(t: tuple) -> str:
    k = t[0]
    if k == "lit":
        return "(" + " ".join((t[1],) + t[2]) + ")"
    if k in ("and", "or"):
        return "(" + k + " " + " ".join(show(y) for y in t[1]) + ")"
    if k == "not":
        return f"(not {show(t[1])})"
    if k == "exists":
        return f"(exists ({' '.join(t[1])}) {show(t[2])})"
    if k == "distinct":
        return f"(distinct {t[1]} {t[2]})"
    if k in ("compose", "via"):
        return f"({k} {show(t[1])} {show(t[2])}" + (f" {t[3]}" if t[3] != 1 else "") + ")"
    if k == "reach":
        return f"({'reach0' if t[4] else 'reach'} {show(t[1])} {t[2]}" + (f" {t[3]}" if t[3] != 1 else "") + ")"
    return f"(closure {show(t[1])}" + (f" {t[2]}" if t[2] != 1 else "") + ")"


def size(t: tuple) -> int:
    return 1 + sum(size(c) for c in _children(t))


def _uniq(xs) -> list:
    return list(dict.fromkeys(xs))


def _split(cols: list, m: int, what: str) -> tuple:
    p = len(cols) - 2 * m
    if p < 0:
        raise TermError(f"{what}: an operand with columns {cols} has no two locations of {m} slot(s)")
    return cols[:p], cols[p:p + m], cols[p + m:]


def free_vars(t: tuple) -> list:
    k = t[0]
    if k == "lit":
        return _uniq(a for a in t[2] if is_var(a))
    if k in ("and", "or"):
        return _uniq(v for y in (t[1][:1] if k == "or" else t[1]) for v in free_vars(y))
    if k == "not":
        return free_vars(t[1])
    if k == "exists":
        return [v for v in free_vars(t[2]) if v not in t[1]]
    if k == "distinct":
        return _uniq(a for a in t[1:] if is_var(a))
    if k in ("compose", "via"):
        m = t[3]
        pa, xa, za = _split(free_vars(t[1]), m, k)
        pb, zb, yb = _split(free_vars(t[2]), m, k)
        if len(pa) != len(pb):
            raise TermError(f"({k} ..): the operands' prefixes differ in length ({len(pa)} and {len(pb)})")
        out = pa + xa + yb + (za if k == "via" else [])
        if len(set(out)) != len(out):
            raise TermError(f"({k} ..): the result's variables {out} are not distinct: rename the operands' variables")
        return out
    cols = free_vars(t[1])
    _split(cols, t[3] if k == "reach" else t[2], k)
    return cols


# ------------------------------------------------------------------------------------------------------------------------------------------ the description

class Description:
    """What a term needs of a description text: its signatures, its sorts, its fluents."""

    def __init__(self, signatures: dict, sorts: dict, fluents: set):
        self.signatures, self.sorts, self.fluents = signatures, sorts, fluents


def _fluents(x, out: set) -> None:
    if isinstance(x, list) and x:
        if x[0] in ("true", "next", "init") and len(x) == 2:
            f = x[1]
            out.add(f if isinstance(f, str) else f[0])
        for y in x:
            _fluents(y, out)


def read_description(text: str) -> Description:
    forms = GDL.parse(text)
    sorts, sigs, rest = GDL.split_declarations(forms)
    fl: set = set()
    for x in rest:
        _fluents(x, fl)
    return Description(sigs, sorts, fl)


# ------------------------------------------------------------------------------------------------------------------------------------------ sorts and safety

def _member(c: str, sort: str, sorts: "dict | None") -> bool:
    return not sorts or sort not in sorts or any(isinstance(m, str) and m == c for m in sorts[sort]) or not all(isinstance(m, str) for m in sorts[sort])


def _merge(into: dict, v: str, s: str, where: str) -> None:
    if into.setdefault(v, s) != s:
        raise TermError(f"{where}: variable {v} is of sort {into[v]} and of sort {s}")


def _sorts(t: tuple, sigs: dict, sorts: "dict | None") -> dict:
    """{free variable: sort} in the order of free_vars; raises on every malformation."""
    k = t[0]
    if k == "lit":
        sig = sigs.get(t[1])
        if sig is None:
            raise TermError(f"the relation {t[1]} has no signature in the description")
        if len(sig) != len(t[2]):
            raise TermError(f"({t[1]} ..) takes {len(sig)} slot(s), got {len(t[2])}")
        d: dict = {}
        for a, s in zip(t[2], sig):
            if is_var(a):
                _merge(d, a, s, show(t))
            elif a != "_" and not _member(a, s, sorts):
                raise TermError(f"{show(t)}: {a} is no member of the sort {s}")
        return d
    if k in ("not", "distinct"):
        raise TermError(f"{show(t)}: (not ..) and (distinct ..) are only allowed inside an `and`, next to positive items that bind their variables")
    if k == "and":
        pos = [y for y in t[1] if y[0] not in ("not", "distinct")]
        d = {}
        for y in pos:
            for v, s in _sorts(y, sigs, sorts).items():
                _merge(d, v, s, show(t))
        for y in t[1]:
            if y[0] == "not":
                if y[1][0] in ("not", "distinct"):
                    raise TermError(f"{show(y)}: a negation of a negation is not supported")
                inner = _sorts(y[1], sigs, sorts)
                for v, s in inner.items():
                    if v not in d:
                        raise TermError(f"unsafe negation: {v} of {show(y)} is not bound by a positive item of {show(t)}")
                    _merge(d, v, s, show(t))
            elif y[0] == "distinct":
                ss = []
                for a in y[1:]:
                    if is_var(a):
                        if a not in d:
                            raise TermError(f"unsafe (distinct ..): {a} is not bound by a positive item of {show(t)}")
                        ss.append(d[a])
                for a in y[1:]:
                    if not is_var(a) and ss and not _member(a, ss[0], sorts):
                        raise TermError(f"{show(y)}: {a} is no member of the sort {ss[0]}")
                if len(ss) == 2 and ss[0] != ss[1]:
                    raise TermError(f"{show(y)}: the two sides are of sorts {ss[0]} and {ss[1]}")
        return {v: d[v] for v in free_vars(t)}
    if k == "or":
        ds = [_sorts(y, sigs, sorts) for y in t[1]]
        for y, d in zip(t[1], ds):
            if set(d) != set(ds[0]):
                raise TermError(f"{show(t)}: the branches have different free variables ({sorted(ds[0])} and {sorted(d)} in {show(y)})")
            for v, s in d.items():
                if ds[0][v] != s:
                    raise TermError(f"{show(t)}: variable {v} is of sort {ds[0][v]} and of sort {s} in different branches")
        return dict(ds[0])
    if k == "exists":
        d = _sorts(t[2], sigs, sorts)
        for v in t[1]:
            if v not in d:
                raise TermError(f"{show(t)}: {v} does not occur free in the body")
        return {v: s for v, s in d.items() if v not in t[1]}
    if k in ("compose", "via"):
        m = t[3]
        da, db = _sorts(t[1], sigs, sorts), _sorts(t[2], sigs, sorts)
        ca, cb = list(da), list(db)
        pa, xa, za = _split(ca, m, k)
        pb, zb, yb = _split(cb, m, k)
        sa, sb = [da[v] for v in ca], [db[v] for v in cb]
        p = len(pa)
        if p != len(pb) or sa[:p] != sb[:p]:
            raise TermError(f"{show(t)}: the operands' prefix columns have different sorts {sa[:p]} and {sb[:p]}")
        if sa[p:p + m] != sa[p + m:] or sb[p:p + m] != sb[p + m:] or sa[p + m:] != sb[p:p + m]:
            raise TermError(f"{show(t)}: the two locations of {m} slot(s) must have the same sorts in both operands (A: {sa[p:]}, B: {sb[p:]})")
        out = pa + xa + yb + (za if k == "via" else [])
        sort_of_name = {**{v: da[v] for v in ca}, **{v: db[v] for v in yb}}
        return {v: sort_of_name[v] for v in free_vars(t)}
    m = t[3] if k == "reach" else t[2]
    d = _sorts(t[1], sigs, sorts)
    cols = list(d)
    _, x, y = _split(cols, m, k)
    if [d[v] for v in x] != [d[v] for v in y]:
        raise TermError(f"{show(t)}: the two locations of {m} slot(s) must have the same sorts ({[d[v] for v in x]} and {[d[v] for v in y]})")
    return d


def sort_of(t: tuple, signatures: dict, sorts: "dict | None" = None) -> tuple:
    """The sorts of the term's free variables, in the order of `free_vars`."""
    return tuple(_sorts(t, signatures, sorts).values())


# ------------------------------------------------------------------------------------------------------------------------------------------ canonical form, expansion

def _rename(t: tuple, mp: dict, fresh) -> tuple:
    """Rename variables by `mp` (a variable missing from it is given fresh(v) the first time it is met, binders included)."""
    def a(x):
        if is_var(x):
            if x not in mp:
                mp[x] = fresh(x)
            return mp[x]
        return x
    k = t[0]
    if k == "lit":
        return ("lit", t[1], tuple(a(x) for x in t[2]))
    if k in ("and", "or"):
        return (k, tuple(_rename(y, mp, fresh) for y in t[1]))
    if k == "not":
        return ("not", _rename(t[1], mp, fresh))
    if k == "exists":
        vs = tuple(a(v) for v in t[1])
        return ("exists", vs, _rename(t[2], mp, fresh))
    if k == "distinct":
        return ("distinct", a(t[1]), a(t[2]))
    if k in ("compose", "via"):
        return (k, _rename(t[1], mp, fresh), _rename(t[2], mp, fresh), t[3])
    if k == "reach":
        return ("reach", _rename(t[1], mp, fresh), t[2], t[3], t[4])
    return ("closure", _rename(t[1], mp, fresh), t[2])


def canonical(t: tuple) -> tuple:
    """The term with its variables renamed ?v0 ?v1 .. in order of first occurrence (the free variables of an operand keep their relative order, so the columns are unchanged)."""
    n = itertools.count()
    return _rename(t, {}, lambda _v: f"?v{next(n)}")


_FRESH = itertools.count()


def _has_structure(t: tuple) -> bool:
    return t[0] in ("compose", "via", "reach", "closure") or any(_has_structure(c) for c in _children(t))


def _occurrences(t: tuple, v: str) -> int:
    if t[0] == "lit":
        return sum(1 for a in t[2] if a == v)
    if t[0] == "distinct":
        return sum(1 for a in t[1:] if a == v)
    return sum(_occurrences(c, v) for c in _children(t)) + (sum(1 for b in t[1] if b == v) if t[0] == "exists" else 0)


def _subst(t: tuple, mp: dict) -> tuple:
    """Simultaneous substitution of the free variables by slots; a bound variable is renamed apart."""
    k = t[0]
    g = lambda x: mp.get(x, x)       # noqa: E731
    if k == "lit":
        return ("lit", t[1], tuple(g(x) for x in t[2]))
    if k in ("and", "or"):
        return (k, tuple(_subst(y, mp) for y in t[1]))
    if k == "not":
        return ("not", _subst(t[1], mp))
    if k == "exists":
        inner = {**mp, **{v: f"?w{next(_FRESH)}" for v in t[1]}}
        return ("exists", tuple(inner[v] for v in t[1]), _subst(t[2], inner))
    if k == "distinct":
        a, b = g(t[1]), g(t[2])
        if "_" in (a, b):
            raise TermError("a word's variable used in (distinct ..) cannot be given the anonymous slot")
        return ("distinct", a, b)
    if k in ("compose", "via"):
        return (k, _subst(t[1], mp), _subst(t[2], mp), t[3])
    if k == "reach":
        return ("reach", _subst(t[1], mp), t[2], t[3], t[4])
    return ("closure", _subst(t[1], mp), t[2])


def _as_term(x) -> tuple:
    if isinstance(x, tuple):
        return x
    if isinstance(x, dict):
        x = x["term"]
    return parse(x)


def expand(t: tuple, registry: dict, _stack: tuple = ()) -> tuple:
    """Every literal whose relation is a registered word (`registry`: {name: term | term text | entry with a "term"}) replaced by the word's own term, its free variables (in order) the literal's
    slots in order, recursively down to relations the registry does not hold. A word used with a constant, `_` or a repeated variable in a slot is expanded only when its term has no
    compose / via / reach / closure (their columns would change); otherwise TermError."""
    k = t[0]
    if k == "lit" and t[1] in registry:
        if t[1] in _stack:
            raise TermError(f"the word {t[1]} is defined in terms of itself")
        w = expand(_as_term(registry[t[1]]), registry, _stack + (t[1],))
        cols = free_vars(w)
        if len(cols) != len(t[2]):
            raise TermError(f"the word {t[1]} has {len(cols)} free variable(s), used with {len(t[2])} slot(s)")
        plain = all(is_var(a) for a in t[2]) and len(set(t[2])) == len(t[2])
        if not plain:
            if _has_structure(w):
                raise TermError(f"the word {t[1]} is built with compose / via / reach / closure: use it with distinct variables to expand it")
            for v, a in zip(cols, t[2]):
                if a == "_" and _occurrences(w, v) > 1:
                    raise TermError(f"the word {t[1]} uses {v} more than once: its slot cannot be `_`")
        return _subst(w, dict(zip(cols, t[2])))
    if k == "lit" or k == "distinct":
        return t
    if k in ("and", "or"):
        return (k, tuple(expand(y, registry, _stack) for y in t[1]))
    if k == "not":
        return ("not", expand(t[1], registry, _stack))
    if k == "exists":
        return ("exists", t[1], expand(t[2], registry, _stack))
    if k in ("compose", "via"):
        return (k, expand(t[1], registry, _stack), expand(t[2], registry, _stack), t[3])
    if k == "reach":
        return ("reach", expand(t[1], registry, _stack), t[2], t[3], t[4])
    return ("closure", expand(t[1], registry, _stack), t[2])


# ------------------------------------------------------------------------------------------------------------------------------------------ compile

class _Compiler:
    def __init__(self, name: str, sigs: dict, fluents, sorts):
        self.name, self.sigs, self.fluents, self.sorts = name, sigs, set(fluents), sorts
        self.rules: list = []                 # (helper name, signature, [rule text ..]) in creation order
        self.memo: dict = {}
        self.n = 0
        self.fresh_n = itertools.count(1)

    def fresh(self) -> str:
        return f"?_{next(self.fresh_n)}"

    def atom(self, rel: str, args) -> str:
        a = f"({rel} {' '.join(args)})" if args else rel
        return f"(true {a})" if rel in self.fluents else a

    def new_name(self) -> str:
        self.n += 1
        h = f"{self.name}__{self.n}"
        if h in self.sigs:
            raise TermError(f"the helper name {h} is already a relation of the description")
        return h

    def sig_of(self, t: tuple) -> tuple:
        s = sort_of(t, self.sigs, self.sorts)
        if len(s) > ARITY_MAX:
            raise TermError(f"{show(t)} has {len(s)} free variables: the evaluator's widest relation has {ARITY_MAX}")
        return s

    def helper(self, t: tuple) -> str:
        """A relation holding the term's rows (columns = its free variables in order); -> its name."""
        key = show(t)
        if key in self.memo:
            return self.memo[key]
        h = self.new_name()
        self.memo[key] = h
        vs = free_vars(t)
        sig = self.sig_of(t)
        self.rules.append((h, sig, self.rules_for(h, t, vs, {v: v for v in vs})))
        return h

    def head(self, h: str, args) -> str:
        return f"({h} {' '.join(args)})" if args else h

    def rules_for(self, h: str, t: tuple, head_vars: list, env: dict) -> list:
        k = t[0]
        if k in ("reach", "closure") and head_vars == free_vars(t):
            return self.reach_rules(h, t) if k == "reach" else self.closure_rules(h, t)
        return [f"(<= {self.head(h, [env[v] for v in head_vars])} {' '.join(body)})" if body else f"{self.head(h, [env[v] for v in head_vars])}"
                for body in self.alts(t, env)]

    def alts(self, t: tuple, env: dict) -> list:
        if t[0] == "or":
            return [b for y in t[1] for b in self.alts(y, env)]
        if t[0] == "exists":
            return self.alts(t[2], {**env, **{v: self.fresh() for v in t[1]}})
        return [self.inline(t, env)]

    def step_body(self, a: tuple, args: list) -> list:
        """The body that says (args) is a row of the operand a (its columns in order)."""
        cols = free_vars(a)
        return self.inline(a, dict(zip(cols, args)))

    def inline(self, t: tuple, env: dict) -> list:
        k = t[0]
        if k == "lit":
            return [self.atom(t[1], [env[a] if is_var(a) else (self.fresh() if a == "_" else a) for a in t[2]])]
        if k == "and":
            out = []
            for y in t[1]:
                if y[0] not in ("not", "distinct"):
                    out += self.inline(y, env)
            for y in t[1]:
                if y[0] == "not":
                    inner = y[1]
                    if inner[0] == "lit" and "_" not in inner[2]:
                        out.append(f"(not {self.inline(inner, env)[0]})")
                    else:
                        h = self.helper(inner)
                        out.append(f"(not {self.head(h, [env[v] for v in free_vars(inner)])})")
            for y in t[1]:
                if y[0] == "distinct":
                    out.append(f"(distinct {' '.join(env[a] if is_var(a) else a for a in y[1:])})")
            return out
        if k == "or":
            h = self.helper(t)
            return [self.head(h, [env[v] for v in free_vars(t)])]
        if k == "exists":
            inner = {**env, **{v: self.fresh() for v in t[1]}}
            return self.inline(t[2], inner)
        if k in ("compose", "via"):
            m = t[3]
            res = free_vars(t)
            pa, xa, za = _split(free_vars(t[1]), m, k)
            p = len(pa)
            P, X = [env[v] for v in res[:p]], [env[v] for v in res[p:p + m]]
            Y = [env[v] for v in res[p + m:p + 2 * m]]
            Z = [env[v] for v in res[p + 2 * m:]] if k == "via" else [self.fresh() for _ in range(m)]
            return self.step_body(t[1], P + X + Z) + self.step_body(t[2], P + Z + Y)
        h = self.helper(t)                                                   # reach, closure
        return [self.head(h, [env[v] for v in free_vars(t)])]

    # reach and closure: recursive helpers over the columns of the operand (P.., x, y)
    def _vars(self, a: tuple, m: int) -> tuple:
        cols = free_vars(a)
        p, x, y = _split(cols, m, "location")
        return [f"?p{i}" for i in range(len(p))], [f"?x{i}" for i in range(m)], [f"?y{i}" for i in range(m)], [f"?w{i}" for i in range(m)]

    def reach_rules(self, h: str, t: tuple) -> list:
        _, a, k, m, zero = t
        P, X, Y, W = self._vars(a, m)
        sig = self.sig_of(a)
        step = lambda x, y: " ".join(self.step_body(a, P + x + y))       # noqa: E731
        names = [h if j == k else self.new_name() for j in range(1, k + 1)]
        prev, rules_of = None, {}
        if zero:
            prev = self.new_name()
            anon = lambda: [self.fresh() for _ in range(m)]              # noqa: E731
            self.rules.append((prev, sig, [f"(<= {self.head(prev, P + X + X)} {step(X, anon())})", f"(<= {self.head(prev, P + Y + Y)} {step(anon(), Y)})"]))
        for nm in names:
            rs = []
            if prev is None:
                rs.append(f"(<= {self.head(nm, P + X + Y)} {step(X, Y)})")
            else:
                if zero:
                    rs.append(f"(<= {self.head(nm, P + X + Y)} {self.head(prev, P + X + Y)})")
                else:
                    rs.append(f"(<= {self.head(nm, P + X + Y)} {step(X, Y)})")
                rs.append(f"(<= {self.head(nm, P + X + Y)} {self.head(prev, P + X + W)} {step(W, Y)})")
            rules_of[nm] = rs
            prev = nm
        for nm in names[:-1]:
            self.rules.append((nm, sig, rules_of[nm]))
        return rules_of[h]

    def closure_rules(self, h: str, t: tuple) -> list:
        _, a, m = t
        P, X, Y, W = self._vars(a, m)
        step = lambda x, y: " ".join(self.step_body(a, P + x + y))       # noqa: E731
        return [f"(<= {self.head(h, P + X + Y)} {step(X, Y)})", f"(<= {self.head(h, P + X + Y)} {self.head(h, P + X + W)} {step(W, Y)})"]


def compile(name: str, term: tuple, signatures: dict, *, head_vars: "list | None" = None, fluents=(), sorts: "dict | None" = None) -> str:   # noqa: A001
    """The rules text of the word `name` = `term`: `(signature name sort ..)`, then the helpers' signatures and rules, then the rules of `name` (see the module). ADDITIVE ONLY: it adds
    relations and changes none. `signatures` the description's {relation: (sort ..)}; `fluents` the relations read as `(true (R ..))`; `sorts` (optional) lets a constant be checked
    against its slot's sort."""
    if not _LEGAL_NAME.match(name):
        raise TermError(f"{name!r} is not a legal relation name ([a-z][a-z0-9_]*)")
    if name in signatures or name in KEYWORDS:
        raise TermError(f"{name} is already a relation (or a constructor): choose another name")
    sorts_by_var = _sorts(term, signatures, sorts)             # raises on every error
    vs = list(sorts_by_var)
    hv = vs if head_vars is None else list(head_vars)
    if any(v not in sorts_by_var for v in hv) or len(set(hv)) != len(hv):
        raise TermError(f"head_vars {hv} must be distinct free variables of the term {vs}")
    if len(hv) > ARITY_MAX:
        raise TermError(f"{name} would have {len(hv)} arguments: the evaluator's widest relation has {ARITY_MAX}")
    c = _Compiler(name, signatures, fluents, sorts)
    rules = c.rules_for(name, term, hv, {v: v for v in vs})
    sig_line = lambda n, sig: " ".join(["(signature", n, *sig]) + ")"        # noqa: E731
    out = [sig_line(name, [sorts_by_var[v] for v in hv])] + [sig_line(h, sig) for h, sig, _rs in c.rules]
    for _h, _sig, rs in c.rules:
        out += rs
    return "\n".join(out + rules) + "\n"
