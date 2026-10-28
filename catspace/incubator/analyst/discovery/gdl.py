"""plan 76.0 (b) A GDL INTERPRETER (Kaveh 2026-10-28: "the machinery has nothing to do with chess. Chess is a rule set which comes and becomes the base layer of every
other learned reasoning on top of it"). A plain Datalog fixpoint over the clauses of a Stanford GDL description (KIF syntax): correctness over speed. Nothing in this
module knows a game; `gdl_chess.py` binds chess's kernel states to the spec's facts.

THE LANGUAGE READ: facts `(rel a b)` and rules `(<= head lit ...)`; a literal is an atom, `(not atom)`, `(distinct t1 t2)` or `(or lit ...)` (expanded into one clause per
branch); variables are `?x`; terms are constants or compound `(f t ...)`. The GDL keywords are relations like any other: `role`, `init`, `true`, `does`, `legal`, `next`,
`terminal`, `goal`. `(true (f ...))`, `(next (f ...))` and `(init (f ...))` are stored flattened by the fluent's functor (relation `true.f`, `next.f`, `init.f`): the state is a
set of `true.f` facts.

THE EVALUATION: the relations' dependency graph is cut into strongly connected components; a negative edge inside one is an error (the description is not stratified);
the components are evaluated bottom-up in topological order, a recursive one by SEMI-NAIVE iteration (each round joins one body literal against the last round's new facts,
the others against everything known). Relations that read neither `true` nor `does` (the STATIC ones: the board's geometry, the tables) are evaluated once, when the
description is loaded. A body is joined left to right in an order chosen once per rule (the next positive literal with the most bound arguments; a negation or a
`distinct` as soon as its variables are bound -- GDL's safety rule guarantees they all are), each positive literal read through a hash index on its bound arguments.

PLAN 76.1 (b) THE NAMED EXTENSIONS this reader keeps (rules/chess.gdl's header states them): (1) SORTS -- the DECLARATIONS `(sort <name> <constant> ...)` (a constant
may be `(range lo hi)`, the numerals between; or a compound pattern `(f <sort> ...)`, a term sort such as the moves) and `(signature <relation> <sort> ...)` (a fluent's
signature serves its `true` / `next` / `init` forms) are not clauses: they are kept in `Program.sorts` / `Program.signatures`, and `typecheck` reads every clause
against them (each argument a member of its slot's sort, each variable of one sort, every relation typed); (5) INPUTS -- besides the state and `does`, a relation
of `INPUTS` (`record`: what the game's record says of the position, a rule that is not state-local) is supplied by the binding at every turn (`run(..., inputs=)`).

  python -m catspace.incubator.analyst.discovery.gdl rules/chess.gdl      # OBS lines: clauses, relations, strata, static facts, sorts, the type check
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict

Term = "str | tuple"                       # a constant / a variable ('?x') / a compound term (functor, arg, ...)

_TOKEN = re.compile(r"\(|\)|[^\s()]+")
FLAT = ("true", "next", "init")            # the keywords whose one argument is a fluent, stored flattened by its functor


class GDLError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------------------------------------------------ reading

def tokenize(text: str) -> list:
    text = "\n".join(line.split(";", 1)[0] for line in text.splitlines())
    return _TOKEN.findall(text)


def parse(text: str) -> list:
    """The description's s-expressions: nested lists of symbols."""
    toks = tokenize(text)
    out, stack = [], []
    for t in toks:
        if t == "(":
            stack.append([])
        elif t == ")":
            if not stack:
                raise GDLError("unbalanced ')'")
            x = stack.pop()
            (stack[-1] if stack else out).append(x)
        else:
            (stack[-1] if stack else out).append(t.lower())
    if stack:
        raise GDLError("unbalanced '('")
    return out


def is_var(t) -> bool:
    return isinstance(t, str) and t.startswith("?")


def to_term(x) -> Term:
    """An s-expression as a term: a symbol stays a string, a list becomes a tuple (functor, args...)."""
    if isinstance(x, str):
        return x
    if not x:
        raise GDLError("empty term ()")
    return tuple(to_term(y) for y in x)


def to_atom(x) -> tuple:
    """An s-expression as an atom (relation name, argument terms). `(true (f a))` -> ('true.f', ('a',))."""
    if isinstance(x, str):
        return (x, ())
    rel = x[0]
    if not isinstance(rel, str):
        raise GDLError(f"relation name is not a symbol: {x}")
    args = tuple(to_term(y) for y in x[1:])
    if rel in FLAT:
        if len(args) != 1:
            raise GDLError(f"({rel} ...) takes one fluent: {x}")
        f = args[0]
        if isinstance(f, str):
            if is_var(f):
                raise GDLError(f"a variable fluent ({rel} {f}) is not supported")
            return (f"{rel}.{f}", ())
        return (f"{rel}.{f[0]}", tuple(f[1:]))
    return (rel, args)


class Clause:
    """head (rel, args); body: a list of literals ('pos', rel, args) | ('neg', rel, args) | ('distinct', t1, t2)."""

    __slots__ = ("head", "body", "text")

    def __init__(self, head, body, text=""):
        self.head, self.body, self.text = head, body, text

    def __repr__(self):
        return self.text or f"Clause({self.head}, {self.body})"


def _literals(x) -> list:
    """One body literal -> its alternatives (each a list of plain literals): `or` expands into one alternative per branch."""
    if isinstance(x, list) and x and x[0] == "or":
        alts = []
        for y in x[1:]:
            alts += _literals(y)
        return alts
    if isinstance(x, list) and x and x[0] == "not":
        if len(x) != 2:
            raise GDLError(f"(not ...) takes one atom: {x}")
        inner = x[1]
        if isinstance(inner, list) and inner and inner[0] in ("or", "not", "distinct"):
            raise GDLError(f"(not {inner[0]} ...) is not supported: {x}")
        rel, args = to_atom(inner)
        return [[("neg", rel, args)]]
    if isinstance(x, list) and x and x[0] == "distinct":
        if len(x) != 3:
            raise GDLError(f"(distinct ...) takes two terms: {x}")
        return [[("distinct", to_term(x[1]), to_term(x[2]))]]
    rel, args = to_atom(x)
    return [[("pos", rel, args)]]


def _unparse(x) -> str:
    return x if isinstance(x, str) else "(" + " ".join(_unparse(y) for y in x) + ")"


DECLARATIONS = ("sort", "signature")       # plan 76.1 (b) (1): the forms that declare the sorts and the relations' signatures (not clauses)


def split_declarations(sexprs: list) -> tuple:
    """-> (sorts {name: [member ...]}, signatures {relation: (sort, ...)}, the other s-expressions). A sort's member is a constant, `(range lo hi)` (expanded to
    its numerals) or a compound pattern (a tuple: functor, argument sorts)."""
    sorts: dict = {}
    sigs: dict = {}
    rest = []
    for x in sexprs:
        if isinstance(x, list) and x and x[0] in DECLARATIONS:
            if len(x) < 2 or not isinstance(x[1], str):
                raise GDLError(f"a declaration names what it declares: {_unparse(x)}")
            if x[0] == "sort":
                if x[1] in sorts:
                    raise GDLError(f"sort {x[1]} declared twice")
                mem = []
                for y in x[2:]:
                    if isinstance(y, list) and y and y[0] == "range":
                        mem += [str(n) for n in range(int(y[1]), int(y[2]) + 1)]
                    elif isinstance(y, list):
                        mem.append(tuple(y))
                    else:
                        mem.append(y)
                sorts[x[1]] = mem
            else:
                if x[1] in sigs:
                    raise GDLError(f"signature of {x[1]} declared twice")
                sigs[x[1]] = tuple(x[2:])
        else:
            rest.append(x)
    return sorts, sigs, rest


def clauses(sexprs: list) -> list:
    out = []
    for x in sexprs:
        if isinstance(x, list) and x and x[0] in DECLARATIONS:
            continue                                   # (plan 76.1: a declaration is no clause)
        text = _unparse(x)
        if isinstance(x, list) and x and x[0] == "<=":
            head = to_atom(x[1])
            bodies = [[]]
            for lit in x[2:]:
                alts = _literals(lit)
                bodies = [b + a for b in bodies for a in alts]
            for b in bodies:
                out.append(Clause(head, b, text))
        else:
            out.append(Clause(to_atom(x), [], text))
    return out


def term_vars(t, acc: set) -> set:
    if is_var(t):
        acc.add(t)
    elif isinstance(t, tuple):
        for y in t[1:]:
            term_vars(y, acc)
    return acc


def is_ground(t) -> bool:
    return not term_vars(t, set())


# ------------------------------------------------------------------------------------------------------------------------------------------ compiling a rule

class _Rule:
    """A clause compiled for joining: variables numbered, the body in its join order, each positive literal with the argument positions bound when it is reached."""

    __slots__ = ("clause", "nvars", "head_rel", "head", "steps", "rec_steps")

    def __init__(self, clause: Clause, scc: set):
        self.clause = clause
        names: dict = {}

        def comp(t):
            if is_var(t):
                if t not in names:
                    names[t] = len(names)
                return ("v", names[t])
            if isinstance(t, tuple):
                return ("f", t[0], tuple(comp(y) for y in t[1:]))
            return ("c", t)

        pos = [l for l in clause.body if l[0] == "pos"]
        rest = [l for l in clause.body if l[0] != "pos"]
        bound: set = set()
        order = []
        pending = list(pos)
        waiting = list(rest)

        def lit_vars(l):
            if l[0] == "distinct":
                return term_vars(l[1], set()) | term_vars(l[2], set())
            acc: set = set()
            for a in l[2]:
                term_vars(a, acc)
            return acc

        def flush():
            for l in list(waiting):
                if lit_vars(l) <= bound:
                    order.append(l)
                    waiting.remove(l)

        flush()
        while pending:
            def score(l):
                nb = sum(1 for a in l[2] if term_vars(a, set()) <= bound)
                return (nb == len(l[2]), nb, -len(l[2]))
            best = max(pending, key=score)
            pending.remove(best)
            order.append(best)
            bound |= lit_vars(best)
            flush()
        if waiting:
            raise GDLError(f"unsafe clause (a variable of a negation, a distinct or the head is bound by no positive literal): {clause.text}")
        hv = set()
        for a in clause.head[1]:
            term_vars(a, hv)
        if not hv <= bound:
            raise GDLError(f"unsafe clause (head variable unbound): {clause.text}")
        self.steps = []
        bound_now: set = set()
        for l in order:
            if l[0] == "distinct":
                self.steps.append(("distinct", comp(l[1]), comp(l[2])))
                continue
            args = tuple(comp(a) for a in l[2])
            if l[0] == "neg":
                self.steps.append(("neg", l[1], args))
                continue
            keypos = tuple(i for i, a in enumerate(l[2]) if term_vars(a, set()) <= bound_now)
            self.steps.append(("pos", l[1], args, keypos))
            bound_now |= lit_vars(l)
        self.nvars = len(names)
        self.head_rel = clause.head[0]
        self.head = tuple(comp(a) for a in clause.head[1])
        self.rec_steps = [i for i, s in enumerate(self.steps) if s[0] == "pos" and s[1] in scc]


def _build(p, env):
    k = p[0]
    if k == "c":
        return p[1]
    if k == "v":
        return env[p[1]]
    return (p[1],) + tuple(_build(q, env) for q in p[2])


def _match(p, val, env, trail) -> bool:
    k = p[0]
    if k == "c":
        return p[1] == val
    if k == "v":
        cur = env[p[1]]
        if cur is None:
            env[p[1]] = val
            trail.append(p[1])
            return True
        return cur == val
    if not isinstance(val, tuple) or len(val) != len(p[2]) + 1 or val[0] != p[1]:
        return False
    for q, v in zip(p[2], val[1:]):
        if not _match(q, v, env, trail):
            return False
    return True


# ------------------------------------------------------------------------------------------------------------------------------------------ the store of facts

class Relation:
    __slots__ = ("facts", "index")

    def __init__(self, facts=()):
        self.facts = set(facts)
        self.index = {}

    def lookup(self, keypos: tuple, key: tuple):
        if not keypos:
            return self.facts
        ix = self.index.get(keypos)
        if ix is None:
            ix = defaultdict(list)
            for t in self.facts:
                ix[tuple(t[i] for i in keypos)].append(t)
            self.index[keypos] = ix
        return ix.get(key, ())

    def add(self, t) -> bool:
        if t in self.facts:
            return False
        self.facts.add(t)
        for keypos, ix in self.index.items():
            ix[tuple(t[i] for i in keypos)].append(t)
        return True


class DB:
    """The facts of one evaluation: relation name -> Relation. `static` relations are shared (read only) with the loaded description."""

    def __init__(self, static: dict):
        self.rels = dict(static)
        self.own: set = set()

    def get(self, rel: str) -> Relation:
        r = self.rels.get(rel)
        if r is None:
            r = self.rels[rel] = Relation()
            self.own.add(rel)
        return r

    def writable(self, rel: str) -> Relation:
        if rel not in self.own:
            old = self.rels.get(rel)
            self.rels[rel] = Relation(old.facts if old is not None else ())
            self.own.add(rel)
        return self.rels[rel]

    def facts(self, rel: str) -> set:
        r = self.rels.get(rel)
        return r.facts if r is not None else set()


# ------------------------------------------------------------------------------------------------------------------------------------------ the program

def _sccs(nodes, edges) -> list:
    """Tarjan's strongly connected components, in reverse topological order of the condensation (a component after every component it depends on)."""
    index, low, on, stack, out = {}, {}, set(), [], []
    counter = [0]
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))

    def visit(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in edges.get(v, ()):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = set()
            while True:
                w = stack.pop()
                on.discard(w)
                comp.add(w)
                if w == v:
                    break
            out.append(comp)

    for v in sorted(nodes):
        if v not in index:
            visit(v)
    return out


INPUTS = ("does", "record")                  # the relations a turn supplies besides the state (`true.*`); plan 76.1 (b) (5): `record`, the game record's reading


def _is_input(rel: str) -> bool:
    return rel.startswith("true.") or rel in INPUTS


class Program:
    """A loaded GDL description: its clauses, its strata, and the static relations evaluated once."""

    def __init__(self, text: str, source: str = "<text>"):
        self.source = source
        self.sorts, self.signatures, _rest = split_declarations(parse(text))
        self.clauses = clauses(parse(text))
        self.rules_of: dict = defaultdict(list)
        deps: dict = defaultdict(set)
        neg: dict = defaultdict(set)
        nodes: set = set()
        for c in self.clauses:
            h = c.head[0]
            nodes.add(h)
            self.rules_of[h].append(c)
            for l in c.body:
                if l[0] in ("pos", "neg"):
                    nodes.add(l[1])
                    deps[h].add(l[1])
                    if l[0] == "neg":
                        neg[h].add(l[1])
        self.deps, self.neg, self.nodes = deps, neg, nodes
        comps = _sccs(nodes, deps)
        self.strata = []
        for comp in comps:
            for h in comp:
                if neg[h] & comp:
                    raise GDLError(f"not stratified: {h} depends negatively on {sorted(neg[h] & comp)} in its own component {sorted(comp)}")
            rules = [_Rule(c, comp) for h in sorted(comp) for c in self.rules_of.get(h, ())]
            self.strata.append((comp, rules))
        # what each relation reads, transitively; the static relations read no input
        self.reads_input: dict = {}
        for comp, _rules in self.strata:
            r = any(_is_input(h) for h in comp) or any(self.reads_input.get(d, _is_input(d)) for h in comp for d in deps[h])
            for h in comp:
                self.reads_input[h] = r
        static_db = DB({})
        for comp, rules in self.strata:
            if not any(self.reads_input[h] for h in comp):
                self._eval_stratum(static_db, comp, rules)
        self.static = {k: v for k, v in static_db.rels.items()}
        self._needed_cache: dict = {}

    # ---------------------------------------------------------------------------------------------- plan 76.1 (b) (1): the sorts

    def signature(self, rel: str) -> "tuple | None":
        """A relation's declared signature (its argument sorts); a fluent's `true` / `next` / `init` forms read the fluent's."""
        for k in FLAT:
            if rel.startswith(k + "."):
                return self.signatures.get(rel[len(k) + 1:])
        return self.signatures.get(rel)

    def member(self, t, sort: str, env: dict, where: str, bad: list) -> None:
        """Check the term `t` is a member of `sort` (a variable takes the sort; a constant is one of its members; a compound term matches one of its patterns)."""
        if sort not in self.sorts:
            bad.append(f"{where}: sort {sort} is not declared")
            return
        if is_var(t):
            if env.setdefault(t, sort) != sort:
                bad.append(f"{where}: variable {t} is of sort {env[t]} and of sort {sort}")
            return
        mem = self.sorts[sort]
        if isinstance(t, str):
            if t not in mem:
                bad.append(f"{where}: {t} is no member of the sort {sort}")
            return
        pat = next((m for m in mem if isinstance(m, tuple) and m[0] == t[0] and len(m) == len(t)), None)
        if pat is None:
            bad.append(f"{where}: the term {term_text(t)} matches no pattern of the sort {sort}")
            return
        for a, s in zip(t[1:], pat[1:]):
            self.member(a, s, env, where, bad)

    def typecheck(self) -> list:
        """THE TYPE CHECK (76.1 (b) (1)) -> what is wrong: a relation with no signature, an argument outside its slot's sort, a variable of two sorts, a `distinct`
        of two sorts. Empty: every clause is well sorted."""
        bad = []
        for rel in sorted(self.nodes):
            sig = self.signature(rel)
            if sig is None:
                bad.append(f"{rel}: no signature")
            else:
                bad += [f"{rel}: its signature's sort {s} is not declared" for s in sig if s not in self.sorts]
        if bad:
            return bad
        for c in self.clauses:
            env: dict = {}
            where = c.text[:80]
            for rel, args in [c.head] + [(l[1], l[2]) for l in c.body if l[0] in ("pos", "neg")]:
                sig = self.signature(rel)
                if len(sig) != len(args):
                    bad.append(f"{where}: {rel} takes {len(sig)} arguments, given {len(args)}")
                    continue
                for a, s in zip(args, sig):
                    self.member(a, s, env, where, bad)
            for l in c.body:
                if l[0] != "distinct":
                    continue
                x, y = l[1], l[2]
                sx = env.get(x) if is_var(x) else None
                sy = env.get(y) if is_var(y) else None
                if sx and sy and sx != sy:
                    bad.append(f"{where}: distinct of the sorts {sx} and {sy}")
                elif sx and not is_var(y):
                    self.member(y, sx, env, where, bad)
                elif sy and not is_var(x):
                    self.member(x, sy, env, where, bad)
        return bad

    def needed(self, wanted: tuple) -> frozenset:
        """The relations the `wanted` ones depend on (themselves included)."""
        key = tuple(sorted(wanted))
        got = self._needed_cache.get(key)
        if got is None:
            seen, todo = set(), list(wanted)
            while todo:
                r = todo.pop()
                if r in seen:
                    continue
                seen.add(r)
                todo += self.deps.get(r, ())
            got = self._needed_cache[key] = frozenset(seen)
        return got

    # ---------------------------------------------------------------------------------------------- evaluation

    def run(self, state, does=(), wanted: "tuple | None" = None, inputs=()) -> DB:
        """Evaluate the description on a state (an iterable of (rel, args) facts, rel 'true.f') and the moves `does` (a list of (role, move term)). `wanted`: the
        relations to compute (and what they need); None -- every relation. `inputs` (plan 76.1 (b) (5)): the other input relations' facts ((rel, args), rel in
        INPUTS: the game record's reading)."""
        db = DB(self.static)
        for rel, args in state:
            db.writable(rel).add(tuple(args))
        if does:
            d = db.writable("does")
            for role, m in does:
                d.add((role, m))
        for rel, args in inputs:
            if rel not in INPUTS:
                raise GDLError(f"{rel} is no input relation ({INPUTS})")
            db.writable(rel).add(tuple(args))
        need = self.needed(tuple(wanted)) if wanted is not None else None
        for comp, rules in self.strata:
            if not any(self.reads_input[h] for h in comp):
                continue                                   # static: already in db
            if need is not None and not (comp & need):
                continue
            self._eval_stratum(db, comp, rules)
        return db

    def _eval_stratum(self, db: DB, comp: set, rules: list) -> None:
        for h in comp:
            db.writable(h)
        recursive = any(r.rec_steps for r in rules)
        delta: dict = defaultdict(set)
        for r in rules:
            for t in self._fire(db, r, None, None):
                if db.writable(r.head_rel).add(t):
                    delta[r.head_rel].add(t)
        if not recursive:
            return
        while any(delta.values()):
            dview = {rel: Relation(ts) for rel, ts in delta.items() if ts}
            new: dict = defaultdict(set)
            for r in rules:
                for i in r.rec_steps:
                    if r.steps[i][1] not in dview:
                        continue
                    for t in self._fire(db, r, i, dview):
                        if t not in db.rels[r.head_rel].facts:
                            new[r.head_rel].add(t)
            for rel, ts in new.items():
                w = db.writable(rel)
                for t in ts:
                    w.add(t)
            delta = new

    def _fire(self, db: DB, rule: _Rule, delta_at, dview) -> list:
        """Every head tuple the rule derives (literal `delta_at` read from `dview`, the others from `db`)."""
        out = []
        env = [None] * rule.nvars
        steps = rule.steps
        n = len(steps)
        head = rule.head

        def go(k):
            if k == n:
                out.append(tuple(_build(p, env) for p in head))
                return
            s = steps[k]
            kind = s[0]
            if kind == "pos":
                _, rel, args, keypos = s
                src = dview[rel] if k == delta_at else db.rels.get(rel)
                if src is None:
                    return
                key = tuple(_build(args[i], env) for i in keypos)
                for t in src.lookup(keypos, key):
                    trail: list = []
                    ok = True
                    for i, a in enumerate(args):
                        if not _match(a, t[i], env, trail):
                            ok = False
                            break
                    if ok:
                        go(k + 1)
                    for v in trail:
                        env[v] = None
            elif kind == "neg":
                _, rel, args = s
                r = db.rels.get(rel)
                if r is None or tuple(_build(a, env) for a in args) not in r.facts:
                    go(k + 1)
            else:
                if _build(s[1], env) != _build(s[2], env):
                    go(k + 1)

        go(0)
        return out

    # ---------------------------------------------------------------------------------------------- the GDL keywords

    def init(self) -> frozenset:
        """The initial state: every `init.f` fact as the fluent `true.f`."""
        db = self.run((), wanted=tuple(r for r in self.nodes if r.startswith("init.")))
        return frozenset(("true." + rel[5:], t) for rel in db.rels if rel.startswith("init.") for t in db.facts(rel))

    def roles(self) -> list:
        return sorted(t[0] for t in self.static.get("role", Relation()).facts)

    def legal(self, state, db: "DB | None" = None) -> dict:
        db = db or self.run(state)
        out: dict = defaultdict(set)
        for role, m in db.facts("legal"):
            out[role].add(m)
        return dict(out)

    def next_state(self, state, does) -> frozenset:
        """The state after the joint move `does` ([(role, move term)]): every `next.f` fact as the fluent `true.f`."""
        wanted = tuple(r for r in self.nodes if r.startswith("next."))
        db = self.run(state, does, wanted=wanted)
        return frozenset(("true." + rel[5:], t) for rel in wanted for t in db.facts(rel))

    def holds(self, db: DB, rel: str) -> bool:
        """A 0-ary relation holds (`terminal`, `mate`, ...)."""
        return () in db.facts(rel)

    def goals(self, db: DB) -> dict:
        out: dict = defaultdict(set)
        for role, v in db.facts("goal"):
            out[role].add(v)
        return dict(out)


def load(path) -> Program:
    from pathlib import Path
    p = Path(path)
    return Program(p.read_text(), str(p))


def term_text(t) -> str:
    return t if isinstance(t, str) else "(" + " ".join(term_text(y) for y in t) + ")"


def report(prog: Program, log=print) -> None:
    nrules = sum(1 for c in prog.clauses if c.body)
    nfacts = sum(1 for c in prog.clauses if not c.body)
    nstatic = sum(len(r.facts) for r in prog.static.values())
    log(f"OBS gdl {prog.source}: clauses {len(prog.clauses)} (rules {nrules}, facts {nfacts}), relations {len(prog.nodes)}, strata {len(prog.strata)}, "
        f"static relations {sum(1 for h in prog.nodes if not prog.reads_input.get(h, _is_input(h)))} with {nstatic} facts")
    bad = prog.typecheck()
    log(f"OBS gdl {prog.source} sorts: {len(prog.sorts)} declared ({', '.join(f'{k} {len(v)}' for k, v in prog.sorts.items())}), signatures {len(prog.signatures)}; "
        f"type check: problems {len(bad)}" + ("".join(f"\n  {x}" for x in bad[:20]) if bad else ""))


if __name__ == "__main__":
    report(load(sys.argv[1] if len(sys.argv) > 1 else "rules/chess.gdl"))
