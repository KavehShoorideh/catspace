"""plan 76.4a THE REGRESSION: a goal's DEFECTS and FIXES derived from the rule spec (Kaveh 2026-10-28, "how are the defects found and raised?" -- until now by hand, a
fixed list per goal kind and a hand table from defect to fix). GENERIC: this module names no game object. It reads a GDL description (`gdl.Program`), a state given
through a BINDING (the game's implementation of the spec's state, its certified relations and its moves: `gdl_chess.Binding` for chess) and a GOAL (a relation head of
the spec with arguments, from the GOAL entry's `gdl` param), and answers:

  (1) THE DERIVATION -- which clause of the head applies and which of its body literals are TRUE / FALSE in the state, evaluated by the interpreter (a top-down,
      tabled reading of the same clauses `gdl.py` evaluates bottom-up: recursion through the ray relations is a table iterated to its fixpoint, negation is negation
      as failure over a lower stratum -- the description is stratified);
  (2) THE DEFECTS -- the FALSE positive literals (with the deepest atom whose failure is a single point of the clause: the squares that make it false) and the TRUE
      negated literals, each TRUE negated literal with its WITNESSES (the derivations that make the atom true: for a GDL `legal` atom, an action of a role); a witness
      that is an action is applied (the certified next state) and the LEAVES of the goal's positive literals' derivations it FLIPS are read -- the leaf (a fluent of
      the state, or a negated atom) its successor makes false, beyond what passing the turn flips. A defect may also be the goal's VALUE falling short of its need
      (a valuation of the store, e.g. the exchange law on a capture), with the atoms the valuation reads;
  (3) THE FIXES -- for a defect, the legal actions of the goal's side (the spec's `legal`, read through its certified implementation) after which, in the next state
      (the spec's `next`, read through its certified implementation), the defect is gone and no defect is new;
  (4) THE ROUTES -- those actions.
Every answer is cached per (the state's key, the goal, the question's extra key). The NAMES of the defects (the words the trace prints: our vocabulary) are the store's
DEFECT_FIX entries' `gdl` params, each bound to a POSITION of the spec (the relation, clause and literal of the leaf it names, or the goal literal that is false, or the
value) -- `name` reads them; nothing here knows what a name means.

SOUNDNESS (by construction over certified relations; no new theorem): a relation read through the binding is one whose RULE entry is PROVED equal to the kernel --
`legal` by Ideas.LegalEquiv.legal_iff (on every position with one king of the side to move and the e.p. square where the rule puts it; hasMove_iff, mate_iff for the
terminal), the next state by Ideas.NextEquiv.applyMoveSpec_eq_applyMove (a function, equal to the kernel's applyMove for every pseudo-legal move); every other atom is
the spec's own clause evaluated by the interpreter. THE READ SET of a regression is the atoms its derivations and witnesses read; a step that touches none of them
leaves them as they were (the frame lemma, Ideas.FrameLemma.frame_unchanged, for the board's squares). CONJECTURE (stated, not proved, not relied on for a claim): that a
fix mending one defect preserves the others is NOT assumed -- the fix test re-reads every defect after the action (no defect may be new), it does not infer it.

PLAN 76.4b THE REGRESSION IN THE NATIVE CORE (76.4a's reading cost ~3x the hand tables at the count budget): `Regressor` is a call-through to the binding's native
core (the same reading, mirrored line by line: the description loaded there from its own text, the binding's readers chosen by the ontology shelf's table, the names
bound by the same DEFECT_FIX params handed over as data; the texts stay here); `PyRegressor` -- the Python reading below, unchanged -- is its TEST ORACLE. A
regression is a function of the state's reading and the goal alone (76.4a's validation of a near regression returned what a fresh reading returns), so the native
core reads afresh and caches per (state key, goal) with the read set kept beside the names.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from catspace.incubator.analyst.discovery import gdl as GDL

# ------------------------------------------------------------------------------------------------------------------------------------------ compiled terms

_V, _C, _F = 0, 1, 2


def _comp(t, names: dict):
    if GDL.is_var(t):
        if t not in names:
            names[t] = len(names)
        return (_V, names[t])
    if isinstance(t, tuple):
        return (_F, t[0], tuple(_comp(y, names) for y in t[1:]))
    return (_C, t)


def _build(p, env):
    """The ground term of a compiled term under `env`, or None when a variable in it is unbound."""
    k = p[0]
    if k == _C:
        return p[1]
    if k == _V:
        return env[p[1]]
    out = [p[1]]
    for q in p[2]:
        v = _build(q, env)
        if v is None:
            return None
        out.append(v)
    return tuple(out)


def _match(p, val, env, trail) -> bool:
    k = p[0]
    if k == _C:
        return p[1] == val
    if k == _V:
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


def _vars(p, acc: set) -> set:
    if p[0] == _V:
        acc.add(p[1])
    elif p[0] == _F:
        for q in p[2]:
            _vars(q, acc)
    return acc


# ------------------------------------------------------------------------------------------------------------------------------------------ the spec

class Spec:
    """A loaded description read top-down: its clauses per relation, the relations' classes (STATIC: evaluated once by `gdl.Program`; INPUT: the state's fluents
    and the turn's inputs; CERTIFIED: read through the binding; DERIVED: the clauses), and the body plans per call pattern."""

    def __init__(self, prog: GDL.Program, certified=()):
        self.prog = prog
        self.certified = frozenset(certified)
        self.rules = {rel: list(cs) for rel, cs in prog.rules_of.items()}
        self.static = {rel for rel in prog.nodes if not prog.reads_input.get(rel, GDL._is_input(rel)) and rel in prog.static}
        self.recursive = set()                              # the relations of a strongly connected component with a recursive clause (a table iterated to its fixpoint)
        for comp, rules in prog.strata:
            if any(r.rec_steps for r in rules):
                self.recursive |= set(comp)
        self._plans: dict = {}
        self._kinds: dict = {}

    def kind(self, rel: str) -> str:
        k = self._kinds.get(rel)
        if k is None:
            k = self._kinds[rel] = ("certified" if rel in self.certified else "static" if rel in self.static else "input" if GDL._is_input(rel) else "derived")
        return k

    def plans(self, rel: str, bound: tuple) -> list:
        """The compiled clauses of `rel` for a call whose argument positions `bound` are ground: (clause index, head, nvars, steps, written) -- `steps` the body in
        its join order, each (kind, rel, args, body index), `written` the body's compiled literals in the written order (the explanation reads them)."""
        key = (rel, bound)
        got = self._plans.get(key)
        if got is None:
            got = self._plans[key] = [self._plan(i, c, bound) for i, c in enumerate(self.rules.get(rel, ()))]
        return got

    def _base(self, rel: str) -> bool:
        return self.kind(rel) != "derived"

    def _plan(self, idx: int, clause: GDL.Clause, bound: tuple):
        names: dict = {}
        head = tuple(_comp(a, names) for a in clause.head[1])
        have: set = set()
        for i in bound:
            _vars(head[i], have)
        lits = []
        for j, l in enumerate(clause.body):
            if l[0] == "distinct":
                lits.append(("distinct", None, (_comp(l[1], names), _comp(l[2], names)), j))
            else:
                lits.append((l[0], l[1], tuple(_comp(a, names) for a in l[2]), j))
        written = list(lits)
        order, pending = [], [l for l in lits if l[0] == "pos"]
        waiting = [l for l in lits if l[0] != "pos"]

        def lv(l) -> set:
            acc: set = set()
            for a in l[2]:
                _vars(a, acc)
            return acc

        def flush():
            for l in list(waiting):
                if lv(l) <= have:
                    order.append(l)
                    waiting.remove(l)

        flush()
        while pending:
            def score(l):                                   # the most bound arguments first (a ray read back from its bound end), a base relation on a tie
                nb = sum(1 for a in l[2] if _vars(a, set()) <= have)
                return (nb == len(l[2]), nb, self._base(l[1]), -len(l[2]))
            best = max(pending, key=score)
            pending.remove(best)
            order.append(best)
            have |= lv(best)
            flush()
        if waiting:
            raise GDL.GDLError(f"unsafe clause for the top-down reading: {clause.text}")
        return (idx, head, len(names), order, written)


# ------------------------------------------------------------------------------------------------------------------------------------------ the prover

class _Table:
    __slots__ = ("rows", "seen", "done", "active", "depth")

    def __init__(self):
        self.rows: list = []
        self.seen: set = set()
        self.done = False
        self.active = False
        self.depth = 0


class Prover:
    """THE INTERPRETER READ TOP-DOWN on one state: `ans(rel, pattern)` -> the ground rows of `rel` matching `pattern` (a tuple, None where free). A derived relation's
    call is a table: a recursive one (a ray) is iterated until no table grows (naive iteration over the strongly connected calls; every table is finite); a negated
    literal is ground when it is reached (GDL's safety) and reads a lower stratum. `facts`: the state's `true.*` facts and the turn's inputs -- (rel, args) pairs, or
    a function (rel, pattern) -> the matching rows (the binding reads them off its state where they are asked); `certified(rel, pattern)`: the rows of a certified
    relation matching the pattern (the binding's kernel: its TRUTH is read there, licensed by its RULE entry's theorem; its DERIVATION, where one is asked, is
    still its clauses -- `leaves`)."""

    def __init__(self, spec: Spec, facts, certified=None):
        self.spec = spec
        if callable(facts):
            self.inp = facts
        else:
            rels: dict = {}
            for rel, args in facts:
                r = rels.get(rel)
                if r is None:
                    r = rels[rel] = GDL.Relation()
                r.add(tuple(args))

            def inp(rel: str, pat: tuple, _rels=rels):
                r = _rels.get(rel)
                if r is None:
                    return ()
                keypos = tuple(i for i, v in enumerate(pat) if v is not None)
                return r.lookup(keypos, tuple(pat[i] for i in keypos))
            self.inp = inp
        self._in: dict = {}
        self.cert = certified
        self.tables: dict = {}
        self.stack: list = []
        self.low = 1 << 30
        self.added = 0
        self.pending: list = []
        self._cert: dict = {}

    # ----------------------------------------------------------------------------------- reading a relation
    def ans(self, rel: str, pat: tuple):
        k = self.spec.kind(rel)
        if k == "static":
            r = self.spec.prog.static.get(rel)
            if r is None:
                return ()
            keypos = tuple(i for i, v in enumerate(pat) if v is not None)
            return r.lookup(keypos, tuple(pat[i] for i in keypos))
        if k == "input":
            key = (rel, pat)
            got = self._in.get(key)
            if got is None:
                got = self._in[key] = self.inp(rel, pat)
            return got
        if k == "certified":
            key = (rel, pat)
            got = self._cert.get(key)
            if got is None:
                got = self._cert[key] = self.cert(rel, pat)
            return got
        key = (rel, pat)
        t = self.tables.get(key)
        if t is not None:
            if t.done:
                return t.rows
            if t.active:
                self.low = min(self.low, t.depth)
                return list(t.rows)
        elif rel not in self.spec.recursive:              # a relation of no recursive component: one pass is its whole answer (nothing it reads can read it)
            t = self.tables[key] = _Table()
            bound = tuple(i for i, v in enumerate(pat) if v is not None)
            for plan in self.spec.plans(rel, bound):
                for row in self._fire(plan, pat):
                    if row not in t.seen:
                        t.seen.add(row)
                        t.rows.append(row)
            t.done = True
            return t.rows
        else:
            t = self.tables[key] = _Table()
        return self._eval(rel, pat, t)

    def holds(self, rel: str, args: tuple) -> bool:
        return bool(self.ans(rel, tuple(args)))

    def _eval(self, rel: str, pat: tuple, t: _Table):
        depth = len(self.stack)
        self.stack.append(t)
        t.active = True
        t.depth = depth
        outer = self.low
        mark = len(self.pending)
        bound = tuple(i for i, v in enumerate(pat) if v is not None)
        plans = self.spec.plans(rel, bound)
        while True:
            self.low = depth + 1
            n0 = self.added
            for plan in plans:
                for row in self._fire(plan, pat):
                    if row not in t.seen:
                        t.seen.add(row)
                        t.rows.append(row)
                        self.added += 1
            mine = self.low
            if mine > depth or self.added == n0:
                break
        self.stack.pop()
        t.active = False
        if mine >= depth:
            t.done = True
            for p in self.pending[mark:]:
                p.done = True
            del self.pending[mark:]
        else:
            self.pending.append(t)
        self.low = min(outer, mine)
        return t.rows

    def _fire(self, plan, pat: tuple) -> list:
        _idx, head, nvars, steps, _w = plan
        env = [None] * nvars
        trail: list = []
        for i, v in enumerate(pat):
            if v is not None and not _match(head[i], v, env, trail):
                return []
        out = []
        n = len(steps)

        def go(k):
            if k == n:
                row = tuple(_build(h, env) for h in head)
                out.append(row)
                return
            kind, rel, args, _j = steps[k]
            if kind == "pos":
                callpat = tuple(_build(a, env) for a in args)
                for row in self.ans(rel, callpat):
                    tr: list = []
                    ok = True
                    for a, v in zip(args, row):
                        if not _match(a, v, env, tr):
                            ok = False
                            break
                    if ok:
                        go(k + 1)
                    for x in tr:
                        env[x] = None
            elif kind == "neg":
                if not self.ans(rel, tuple(_build(a, env) for a in args)):
                    go(k + 1)
            else:
                if _build(args[0], env) != _build(args[1], env):
                    go(k + 1)

        go(0)
        return out

    # ----------------------------------------------------------------------------------- explaining an atom
    def derivations(self, rel: str, args: tuple) -> list:
        """The clause instances that derive the ground atom (rel, args) here: [(clause index, [(kind, rel, ground args) in the written order])]."""
        out = []
        pat = tuple(args)
        for plan in self.spec.plans(rel, tuple(range(len(pat)))):
            idx, head, nvars, steps, written = plan
            env = [None] * nvars
            trail: list = []
            if not all(_match(head[i], v, env, trail) for i, v in enumerate(pat)):
                continue
            n = len(steps)

            def go(k):
                if k == n:
                    lits = []
                    for kind, r2, av, _j in written:
                        lits.append((kind, r2, tuple(_build(a, env) for a in av)))
                    out.append((idx, lits))
                    return
                kind, r2, av, _j = steps[k]
                if kind == "pos":
                    for row in self.ans(r2, tuple(_build(a, env) for a in av)):
                        tr: list = []
                        if all(_match(a, v, env, tr) for a, v in zip(av, row)):
                            go(k + 1)
                        for x in tr:
                            env[x] = None
                elif kind == "neg":
                    if not self.ans(r2, tuple(_build(a, env) for a in av)):
                        go(k + 1)
                else:
                    if _build(av[0], env) != _build(av[1], env):
                        go(k + 1)

            go(0)
        return out

    def leaves(self, rel: str, args: tuple, _seen: "frozenset | None" = None) -> frozenset:
        """THE LEAVES of every derivation of the ground TRUE atom (rel, args): the fluents of the state and the negated atoms its derivations rest on, each a `Leaf`
        (its place in the spec and the atom); static facts are no leaves (no action changes them); a certified atom is a leaf of its own."""
        seen = (_seen or frozenset()) | {(rel, args)}
        out = set()
        for idx, lits in self.derivations(rel, args):
            sub = set()
            ok = True
            for j, (kind, r2, av) in enumerate(lits):
                if kind == "distinct":
                    continue
                k = self.spec.kind(r2)
                if kind == "neg":
                    sub.add(Leaf((rel, idx, j), (r2, av), True))
                elif k == "static":
                    continue
                elif k == "input" or (k == "certified" and not self.spec.rules.get(r2)):
                    sub.add(Leaf((rel, idx, j), (r2, av), False))
                elif (r2, av) in seen:
                    ok = False                    # a derivation through the atom itself is not minimal
                    break
                else:
                    sub |= self.leaves(r2, av, seen)
            if ok:
                out |= sub
        return frozenset(out)

    def why_not(self, rel: str, args: tuple) -> tuple:
        """THE DEEPEST FALSE ATOM of the false atom (rel, args): its relation's one clause read in the written order while each literal before the failing one has
        exactly one answer (the failure is a single point); the atom is returned, its free arguments None."""
        cur = (rel, tuple(args))
        for _ in range(64):
            r, a = cur
            if self.spec.kind(r) != "derived":
                return cur
            plans = self.spec.plans(r, tuple(i for i, v in enumerate(a) if v is not None))
            if len(plans) != 1:
                return cur
            idx, head, nvars, steps, written = plans[0]
            env = [None] * nvars
            trail: list = []
            if not all(_match(head[i], v, env, trail) for i, v in enumerate(a) if v is not None):
                return cur
            nxt = None
            for kind, r2, av, _j in written:
                if kind == "distinct":
                    if _build(av[0], env) is not None and _build(av[1], env) is not None and _build(av[0], env) == _build(av[1], env):
                        return cur
                    continue
                callpat = tuple(_build(x, env) for x in av)
                if kind == "neg":
                    if None in callpat or self.ans(r2, callpat):
                        return cur
                    continue
                rows = list(self.ans(r2, callpat))
                if not rows:
                    nxt = (r2, callpat)
                    break
                if len(rows) > 1:
                    return cur
                tr: list = []
                if not all(_match(x, v, env, tr) for x, v in zip(av, rows[0])):
                    return cur
            if nxt is None:
                return cur
            cur = nxt
        return cur


@dataclass(frozen=True)
class Leaf:
    at: tuple                  # (relation, clause index, body literal index): the leaf's place in the spec
    atom: tuple                # (relation, ground args)
    neg: bool = False          # a negated literal (its atom false in the state)


# ------------------------------------------------------------------------------------------------------------------------------------------ goals and defects

@dataclass(frozen=True)
class Goal:
    """A goal read from a GOAL entry's `gdl` param, its parameters substituted: `head` (relation, args: constants and '?x' variables), `on_move` (the role whose
    turn the head is read on: None as the state is), `value` (None, or (law, args, need, reads) -- the valuation the binding computes, the least it must reach, and
    the atom pattern the valuation reads), `extra` (the question's own key: a placement)."""
    name: str
    head: tuple
    on_move: "str | None" = None
    value: "tuple | None" = None
    extra: tuple = ()


def goal_of(name: str, gdl: dict, B, us: str, target=None, unit=None, extra: tuple = (), frm=None) -> Goal:
    """The Goal of a GOAL entry's `gdl` for one question: '$us' / '$them' the roles, '$target' / '$from' a location's terms (the binding's), '$unit' the
    valuation's least gain."""
    them = B.other(us)

    def sub(tok) -> tuple:
        if tok == "$us":
            return (us,)
        if tok == "$them":
            return (them,)
        if tok == "$target":
            return tuple(B.terms(target))
        if tok == "$from":
            return tuple(B.terms(frm))
        if tok == "$unit":
            return (unit,)
        return (tok,)

    def subs(toks) -> tuple:
        out = ()
        for t in toks:
            out += sub(t)
        return out

    head = (gdl["head"][0], subs(gdl["head"][1:]))
    om = sub(gdl["on_move"])[0] if gdl["on_move"] is not None else None
    v = gdl["value"]
    value = None if v is None else (v["law"], subs(v["args"]), sub(v["need"])[0], (v["reads"][0], subs(v["reads"][1:])))
    return Goal(name, head, om, value, tuple(extra))


@dataclass
class Witness:
    atom: tuple                # the negated literal's atom (true in the state)
    body: tuple                # the witness's derivation, one level: the inner clause's ground literals
    action: object = None      # the binding's action, when the witness reads a `legal` atom of a role
    flips: frozenset = frozenset()     # the goal's positive leaves the action's next state makes false, beyond the pass


@dataclass
class Derived:
    kind: str                  # 'false' (a positive literal no binding satisfies) | 'witness' (a witness of a true negated literal) | 'value' (the value short of the need)
    clause: int                # the head's clause the literal belongs to (-1: the head itself, or the value)
    lit: int                   # the literal's index in the clause body (-1: the head)
    atom: tuple                # the literal's atom, as far as the clause's bindings ground it
    deep: tuple = ()           # 'false': the deepest false atom (`Prover.why_not`)
    witness: "Witness | None" = None
    value: "tuple | None" = None       # 'value': (the value, the need, the atoms the valuation read)


@dataclass
class Regression:
    goal: Goal
    holds: bool                # the head holds (and its value reaches the need)
    clauses: list              # [(clause index, [(lit index, kind, atom, truth)])]: the derivation read
    defects: list              # [Derived]
    reads: frozenset           # the atoms read: the literals, their deepest false atoms, the leaves, the witnesses' bodies, the valuation's reads
    ms: float = 0.0
    rows: list = field(default_factory=list)          # the head's rows where it holds (its derivations' bindings)
    value: object = None                              # the value the goal's valuation read (None: not read)
    leaves: frozenset = frozenset()                   # the leaves of the clause's TRUE positive literals (what a witness, or a keeping move of the other side, must flip)
    state: object = None       # the state it was read on
    queries: dict = field(default_factory=dict)      # every BASE query it asked there (an input or a certified relation, its pattern) -> the rows answered
    values: list = field(default_factory=list)       # every value it read: (law, args, value)
    locs: frozenset = frozenset()                    # the locations its witnesses' next states were read at (their leaves', the actions' touched ones)


class Regressor:
    """THE REGRESSION (plan 76.4b): read by the binding's NATIVE core -- `B.native_regress(x, goal)` (the whole regression, regression.rs: the same reading as
    `PyRegressor`'s below, the description loaded there from its own text), `B.native_names(x, goal, entries, theirs, future)` (the names' records, cached there per
    (the state's key, the goal); their texts filled here from the entries' templates) and `B.native_fixes(...)` (the fix test of a list of actions in one crossing).
    The whole regression is cached here per (state key, goal), as 76.4a's was. `PyRegressor` (76.4a's Python reading) is its TEST ORACLE: equal on every question
    of the guard positions (scripts/plan76_4a.py compare). The binding contract is PyRegressor's, plus `future(x, fut)` (the goal's future of a state, as data:
    None -- the state itself) and the native methods."""

    CACHE_MAX = 400_000                                         # a fuse on the memory the cache may hold (a count, named as one: cleared whole when reached, never a cut of an answer)

    def __init__(self, spec: Spec, binding):
        self.spec = spec
        self.B = binding
        self.cache: dict = {}
        self.stats = {"regressions": 0, "cached": 0, "validated": 0, "ms": 0.0}

    def clear(self) -> None:
        self.cache.clear()
        self.B.native_clear()

    def prover(self, x) -> Prover:
        return Prover(self.spec, self.B.inputs(x), self.B.certified(x))

    def regress(self, x, goal: Goal, near: "Regression | None" = None) -> Regression:
        """The regression of `goal` on the state `x` (the goal's role on move): from the cache (the same state, the same goal), else read by the native core.
        (`near`: 76.4a's validation hint, read by the oracle only -- the native reading is cheaper than a validation.)"""
        if goal.on_move is not None:
            x = self.B.to_move(x, goal.on_move)
        key = (self.B.key(x), goal)
        got = self.cache.get(key)
        if got is not None:
            self.stats["cached"] += 1
            return got
        t0 = time.perf_counter()
        got = self.B.native_regress(x, goal)
        got.ms = (time.perf_counter() - t0) * 1000
        self.stats["regressions"] += 1
        self.stats["ms"] += got.ms
        if len(self.cache) >= self.CACHE_MAX:
            self.cache.clear()
        self.cache[key] = got
        return got

    def names(self, x, goal: Goal, entries: list, theirs: bool = False, future=None) -> list:
        """THE NAMES of the goal's defects on the future of `x` (`future`: the binding's counterfactual of it, as data; None -- `x`) -- `name(regress(B.future(x,
        future), goal), entries, B, ...)`, read by the native core in one crossing (its cache per (key, goal) replays the read set with the names)."""
        return self.B.native_names(x, goal, entries, theirs, future)

    def fixes(self, x, goal: Goal, key, actions, future, entries: list, before: "list | None" = None) -> list:
        """regression.fixes read by the native core: every action's fix test in one crossing (no callback per action)."""
        return self.B.native_fixes(x, goal, key, actions, future, entries, before)


class PyRegressor:
    """THE TEST ORACLE (plan 76.4b): the Python reading of 76.4a -- THE REGRESSION over one spec and one binding, cached per (state key, goal). `binding` supplies:
    `inputs(x)` -> a function (rel, pattern) -> the state's `true.*` and input rows (the spec's state, read where asked), `certified(x)` -> a function (rel, pattern)
    -> rows for the certified relations (`spec.certified`), `apply(x, action)` (the certified next state), `pass_(x)` (the turn passed), `to_move(x, role)`, `key(x)`,
    `action(row)` (a certified `legal` row -> the binding's action or None), `value(x, law, args)` (None where the valuation reads no position), `future(x, fut)`."""

    CACHE_MAX = 400_000                                         # a fuse on the memory the cache may hold (a count, named as one: cleared whole when reached, never a cut of an answer)

    def __init__(self, spec: Spec, binding):
        self.spec = spec
        self.B = binding
        self.cache: dict = {}
        self.last: dict = {}                                    # goal -> its last regression (the first one a new state is validated against)
        self.stats = {"regressions": 0, "cached": 0, "validated": 0, "ms": 0.0}

    def clear(self) -> None:
        self.cache.clear()
        self.last.clear()

    def names(self, x, goal: Goal, entries: list, theirs: bool = False, future=None) -> list:
        y = self.B.future(x, future)
        return name(self.regress(y, goal), entries, self.B, y, theirs=theirs)

    def fixes(self, x, goal: Goal, key, actions, future, entries: list, before: "list | None" = None) -> list:
        """76.4a's fix test, action by action (the oracle of the native one)."""
        fut = self.B.future
        r0 = self.regress(fut(x, future), goal)
        if before is None:
            before = {(n.kind, n.sq) for n in name(r0, entries, self.B, fut(x, future))} - {key}
        out = []
        for a in actions:
            x1 = self.B.apply(x, a)
            y = fut(x1, future)
            r1 = self.regress(y, goal, near=r0)
            keys = [(n.kind, n.sq) for n in name(r1, entries, self.B, y)]
            out.append((a, key not in keys, [k for k in keys if k not in before]))
        return out

    def prover(self, x) -> Prover:
        return Prover(self.spec, self.B.inputs(x), self.B.certified(x))

    def regress(self, x, goal: Goal, near: "Regression | None" = None) -> Regression:
        """The regression of `goal` on the state `x`: from the cache (the same state, the same goal); else a regression of the same goal VALIDATED on `x` (`near`,
        then the goal's last one: every base query it asked answers the same on `x`, every value it read is the same, and `x` agrees with its state at every
        location its witnesses' next states were read at -- then, the evaluation being a function of those answers alone, it IS the regression on `x`); else
        read afresh."""
        if goal.on_move is not None:
            x = self.B.to_move(x, goal.on_move)
        key = (self.B.key(x), goal)
        got = self.cache.get(key)
        if got is not None:
            self.stats["cached"] += 1
            return got
        t0 = time.perf_counter()
        for cand in (near, self.last.get(goal)):
            if cand is not None and cand.goal == goal and self._valid(cand, x):
                self.stats["validated"] += 1
                self.stats["ms"] += (time.perf_counter() - t0) * 1000
                self._put(key, goal, cand)
                return cand
        got = self._regress(x, goal)
        got.ms = (time.perf_counter() - t0) * 1000
        self.stats["regressions"] += 1
        self.stats["ms"] += got.ms
        self._put(key, goal, got)
        return got

    def _put(self, key, goal: Goal, reg: Regression) -> None:
        if len(self.cache) >= self.CACHE_MAX:
            self.cache.clear()
        self.cache[key] = reg
        self.last[goal] = reg

    def _valid(self, reg: Regression, x) -> bool:
        if reg.state is None:
            return False
        inp, cert = self.B.inputs(x), self.B.certified(x)
        for (rel, pat), ans in reg.queries.items():
            got = inp(rel, pat) if self.spec.kind(rel) == "input" else cert(rel, pat)
            if tuple(got) != ans:
                return False
        for law, args, v in reg.values:
            if self.B.value(x, law, args) != v:
                return False
        return self.B.same_at(reg.state, x, reg.locs)

    def _regress(self, x, goal: Goal) -> Regression:
        P = self.prover(x)
        self._values: list = []
        self._locs: set = set()
        got = self._regress_on(P, x, goal)
        got.state = x
        got.queries = {k: tuple(v) for k, v in P._in.items()}
        got.queries.update((k, tuple(v)) for k, v in P._cert.items())
        got.values = self._values
        got.locs = frozenset(self._locs)
        return got

    def _regress_on(self, P: Prover, x, goal: Goal) -> Regression:
        rel, args = goal.head
        names: dict = {}
        hpat = tuple(_build(_comp(a, names), [None] * len(names)) if not GDL.is_var(a) else None for a in args)
        reads: set = set()
        rows = list(P.ans(rel, hpat))
        if rows:
            reads |= {(rel, r) for r in rows}
            if goal.value is None:
                self._leaves = frozenset()                     # (a holding rule head: its clause is read for the leaves its true literals rest on -- what a keeper must flip)
                if self.spec.kind(rel) == "derived":
                    for idx, head, nvars, _steps, written in self.spec.plans(rel, tuple(i for i, v in enumerate(hpat) if v is not None)):
                        env = [None] * nvars
                        if all(_match(head[i], v, env, []) for i, v in enumerate(hpat) if v is not None):
                            self._read_clause(P, x, idx, written, env, reads)
                return Regression(goal, True, [], [], frozenset(reads), rows=rows, leaves=self._leaves)
            law, vargs, need, vreads = goal.value
            v = self.B.value(x, law, vargs)
            self._values.append((law, vargs, v))
            rr = self._pattern_rows(P, vreads)
            reads |= rr
            if v is None or v >= need:                          # (None: the valuation reads no position here -- the head alone decides; the binder verifies)
                return Regression(goal, True, [], [], frozenset(reads), rows=rows, value=v)
            return Regression(goal, False, [], [Derived("value", -1, -1, (rel, hpat), value=(v, need, tuple(sorted(rr))))], frozenset(reads), rows=rows, value=v)
        clauses, defects = [], []
        if self.spec.kind(rel) != "derived" or not self.spec.rules.get(rel):
            deep = P.why_not(rel, hpat)
            reads.add(deep)
            return Regression(goal, False, [], [Derived("false", -1, -1, (rel, hpat), deep=deep)], frozenset(reads))
        self._leaves = frozenset()
        for idx, head, nvars, _steps, written in self.spec.plans(rel, tuple(i for i, v in enumerate(hpat) if v is not None)):
            env = [None] * nvars
            trail: list = []
            if not all(_match(head[i], v, env, trail) for i, v in enumerate(hpat) if v is not None):
                continue
            lits, ds = self._read_clause(P, x, idx, written, env, reads)
            clauses.append((idx, lits))
            defects += ds                                        # (a disjunctive head: each clause's defects are an alternative; the names say which clause they read)
        return Regression(goal, False, clauses, defects, frozenset(reads), leaves=self._leaves)

    def _pattern_rows(self, P: Prover, pattern: tuple) -> set:
        rel, args = pattern
        pat = tuple(None if GDL.is_var(a) else a for a in args)
        return {(rel, r) for r in P.ans(rel, pat)}

    def _read_clause(self, P: Prover, x, idx: int, written: list, env: list, reads: set) -> tuple:
        """One clause of the false head, its literals in the written order: a positive literal read for its bindings (the first binding continues the clause; a
        literal with no binding is a 'false' defect and ends the clause's reading), a negated literal's atom read (true: its witnesses are 'witness' defects)."""
        lits, defects = [], []
        positives = []                                          # the true positive atoms (their leaves are what a witness may flip)
        negs = []
        for j, (kind, rel, args, _jj) in enumerate(written):
            if kind == "distinct":
                ok = _build(args[0], env) != _build(args[1], env)
                lits.append((j, kind, None, ok))
                if not ok:
                    defects.append(Derived("false", idx, j, ("distinct", (_build(args[0], env), _build(args[1], env)))))
                    return lits, defects
                continue
            callpat = tuple(_build(a, env) for a in args)
            if kind == "pos":
                rows = list(P.ans(rel, callpat))
                if not rows:
                    deep = P.why_not(rel, callpat)
                    reads.add((rel, callpat))
                    reads.add(deep)
                    lits.append((j, kind, (rel, callpat), False))
                    defects.append(Derived("false", idx, j, (rel, callpat), deep=deep))
                    return lits, defects
                row = rows[0]
                tr: list = []
                for a, v in zip(args, row):
                    _match(a, v, env, tr)
                lits.append((j, kind, (rel, row), True))
                reads.add((rel, row))
                positives.append((rel, row))
            else:
                negs.append((j, rel, callpat))
                lits.append((j, kind, (rel, callpat), not P.ans(rel, callpat)))
        leaves = frozenset()
        for rel, row in positives:
            leaves |= P.leaves(rel, row)
        reads |= {lf.atom for lf in leaves}
        self._leaves = self._leaves | leaves
        self._locs |= {self.B.loc(lf.atom) for lf in leaves}
        base_flips = None
        for j, rel, callpat in negs:
            if not P.ans(rel, callpat):
                continue
            for w in self._witnesses(P, rel, callpat):
                reads |= {(r, a) for _k, r, a in w.body if r is not None}
                if w.action is not None and leaves:
                    if base_flips is None:
                        base_flips = self._flips(self.B.pass_(x), leaves, x)
                        self._locs |= {self.B.loc(lf.atom) for lf in leaves}
                    w.flips = self._flips(self.B.apply(x, w.action), leaves, x) - base_flips
                    self._locs |= set(self.B.touched(x, w.action))
                defects.append(Derived("witness", idx, j, (rel, callpat), witness=w))
        return lits, defects

    def _witnesses(self, P: Prover, rel: str, args: tuple) -> list:
        """The witnesses of the true atom (rel, args): its derivations one level down; an action when the derivation reads a certified `legal` row (the GDL keyword's
        atom: a role's move), in the order the certified relation lists them."""
        out = []
        for _idx, lits in P.derivations(rel, args):
            action = None
            for kind, r2, av in lits:
                if kind == "pos" and r2 == "legal" and self.spec.kind(r2) == "certified":
                    action = self.B.action(av)
            out.append(Witness((rel, args), tuple(lits), action))
        return out

    def _flips(self, x2, leaves: frozenset, x) -> frozenset:
        """The leaves false in the state `x2` (true in `x`, where they were read): a fluent leaf read off the state, a negated leaf by its atom read there."""
        P2 = Prover(self.spec, self.B.inputs(x2), self.B.certified(x2))
        out = set()
        for lf in leaves:
            if P2.holds(*lf.atom) == (not lf.neg):
                continue
            out.add(lf)
        return frozenset(out)


# ------------------------------------------------------------------------------------------------------------------------------------------ the names (the store)

def _at_matches(at: tuple, pat) -> bool:
    """A leaf's place against a name's pattern [relation, clause index or null, literal index]."""
    return at[0] == pat[0] and (pat[1] is None or at[1] == pat[1]) and at[2] == pat[2]


@dataclass
class Named:
    kind: str                  # the name (the DEFECT_FIX entry)
    sq: object                 # the location the name is about (the binding's)
    by: list                   # the locations involved
    why: str
    src: list = field(default_factory=list)       # the derived defects it names


def name(reg: Regression, entries: list, B, x, theirs: bool = False) -> list:
    """THE NAMES of a regression's defects: every DEFECT_FIX entry whose `gdl` param names a defect of the goal (entries in their `order`), each binding a derived
    defect by its place in the spec -- a 'witness' by the leaf it flips (`leaf`, and not when `unless` flips too), a 'false' literal by its place (`clause`, `lit`), a
    'value' -- grouped per witness or per location (`group`), its location and its locations involved read through the binding (`sq`, `by`), its text filled from
    the template (`text`; a dict: by the deepest false atom's relation). A derived defect no name binds is not named (the comparison counts it).
    `theirs`: the goal is the OTHER side's, read for the side that must keep (the names bound `for` 'theirs'): what makes it stand is the keeper's defect -- 'holds'
    (the head holds and its value meets the need: the head's rows, the value read) or 'leaf' (a leaf of a TRUE positive literal of its clause, one per location)."""
    out = []
    side = "theirs" if theirs else "ours"
    mine = sorted((e for e in entries if e["gdl"]["goal"] == reg.goal.name and e["gdl"].get("for", "ours") == side), key=lambda e: e["gdl"]["order"])
    for e in mine:
        g = e["gdl"]
        groups: dict = {}
        order = []
        if g["defect"] == "holds":
            if reg.holds and reg.rows:
                sq = B.loc((reg.goal.head[0], tuple(None if GDL.is_var(a) else a for a in reg.goal.head[1])))
                by = sorted({B.loc((reg.goal.head[0], r)) for r in reg.rows} - {None}, key=B.order_key)
                out.append(Named(e["name"], sq, by, _text(g["text"], dict(sq=B.loc_name(sq), n=len(by), value=reg.value)), []))
            continue
        if g["defect"] == "leaf":
            for lf in sorted((lf for lf in reg.leaves if _at_matches(lf.at, g["leaf"])), key=lambda lf: B.order_key(B.loc(lf.atom))):
                sq = B.loc(lf.atom)
                out.append(Named(e["name"], sq, [], _text(g["text"], dict(sq=B.loc_name(sq), unit=B.unit_letter(x, sq))), []))
            continue
        for i, d in enumerate(reg.defects):
            if d.kind != g["defect"]:
                continue
            if d.kind == "witness":
                hit = [lf for lf in d.witness.flips if _at_matches(lf.at, g["leaf"])]
                if not hit or (g.get("unless") and any(_at_matches(lf.at, g["unless"]) for lf in d.witness.flips)):
                    continue
                lf = sorted(hit, key=lambda l: repr(l.atom))[0]
                fr, to = B.ends(d.witness.action)
                leaf_loc = B.loc(lf.atom)
                fields = {"leaf": leaf_loc, "to": to, "from": fr}
                sq = fields[g["sq"]]
                by = fields[g["by"]]
                gk = (sq,) if g["group"] == "sq" else (sq, i)
                if gk not in groups:
                    groups[gk] = Named(e["name"], sq, [], "", [])
                    order.append(((B.order_key(sq) if g["group"] == "sq" or g.get("sort") == "sq" else 0, i), gk))
                nd = groups[gk]
                nd.by.append(by)
                nd.src.append(d)
                nd.why = _text(g["text"], dict(sq=B.loc_name(sq), frm=B.loc_name(fr), to=B.loc_name(to), unit=B.unit_letter(x, fr), n=len(nd.by)))
            elif d.kind == "false":
                if g.get("clause") is not None and (d.clause, d.lit) != (g["clause"], g["lit"]):
                    continue
                rel = d.deep[0] if d.deep else d.atom[0]
                tmpl = g["text"].get(rel) if isinstance(g["text"], dict) else g["text"]
                if tmpl is None:
                    continue
                sq = B.loc(d.deep) if d.deep else B.loc(d.atom)
                gk = (sq, i)
                groups[gk] = Named(e["name"], sq, [], _text(tmpl, dict(sq=B.loc_name(sq) if sq is not None else "")), [d])
                order.append(((0, i), gk))
            elif d.kind == "value":
                sq = B.loc(d.atom)
                by = sorted({B.loc(a) for a in d.value[2]} - {None}, key=B.order_key)
                gk = (sq, i)
                groups[gk] = Named(e["name"], sq, by, _text(g["text"], dict(sq=B.loc_name(sq), n=len(by), value=d.value[0])), [d])
                order.append(((0, i), gk))
        for _k, gk in sorted(order, key=lambda t: t[0]):
            out.append(groups[gk])
    return out


def _text(tmpl: str, fields: dict) -> str:
    return tmpl.format(**fields)


# ------------------------------------------------------------------------------------------------------------------------------------------ a goal within n moves

def actions(R: "Regressor | PyRegressor", x, role: str) -> list:
    """The legal actions of `role` at `x`: the spec's `legal` (GDL's keyword: a role's moves), read through its certified relation, in its order."""
    P = R.prover(x)
    return [a for a in (R.B.action(row) for row in P.ans("legal", (role, None))) if a is not None]


@dataclass
class Within:
    action: object             # our action
    holds: bool                # the goal is reached within the moves, every reply read
    escapes: list              # [(their reply, the named defects of our best answer there, or of the position itself at the last move)]
    names: list = field(default_factory=list)      # depth 1: the goal's named defects after the action (none: it holds)
    near: tuple = (1, 0)                           # depth 1: how far the goal is -- (a positive literal of its clause false, the named defects): the best answer is the least


def within(R: "Regressor | PyRegressor", x, goal: Goal, us: str, n: int, entries: list) -> list:
    """THE GOAL WITHIN `n` MOVES OF OURS, every reply of theirs read -- an AND-OR over the spec's certified `legal` and next state, the regression of the goal at
    its leaves (the same regression: no table per depth): n = 1, the goal regressed after each of our actions (its named defects; none = it holds there); n > 1,
    an action holds when every reply leaves an action that holds within n - 1 (a reply is an ESCAPE otherwise, with the named defects of our best answer: the
    nearest -- no positive literal of the goal's clause false, then the fewest named defects). The DEFECTS of the goal within n moves at `x` are, per action of
    ours, its escapes; its FIXES the actions with none. -> [Within] in the actions' order."""
    out = []
    them = R.B.other(us)
    for a in actions(R, x, us):
        x1 = R.B.apply(x, a)
        if n <= 1:
            reg = R.regress(x1, goal)
            nm = name(reg, entries, R.B, x1)
            out.append(Within(a, reg.holds, [], nm, (int(any(d.kind == "false" for d in reg.defects)), len(nm))))
            continue
        escapes = []
        replies = actions(R, x1, them)
        if not replies:                                        # no reply: the goal is read where the line stops
            reg = R.regress(x1, goal)
            out.append(Within(a, reg.holds, [], name(reg, entries, R.B, x1)))
            continue
        for r in replies:
            x2 = R.B.apply(x1, r)
            sub = within(R, x2, goal, us, n - 1, entries)
            if any(w.holds for w in sub):
                continue
            best = min(sub, key=lambda w: (w.near if n == 2 else (len(w.escapes),), repr(w.action)), default=None)
            escapes.append((r, best.names if best is not None and n == 2 else (best.escapes if best is not None else [])))
        out.append(Within(a, not escapes, escapes))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ fixes

def fixes(R: "Regressor | PyRegressor", x, goal: Goal, key, actions, future, entries: list, before: "list | None" = None) -> list:
    """THE FIXES of the named defect `key` (its (name, location)) of `goal`, read on the state `x` (the goal's side to move): for each action of `actions` (legal
    there: the caller lists them from the certified relation), its certified next state, the goal's future of it (`future`: as data, the binding's counterfactual
    -- `B.future(x1, future)`: the state the goal is read on, e.g. the question's placement; None -- the next state itself), the regression there, named ->
    [(action, mended, new)] where `mended` = no named defect with the key is left, `new` = the named defects whose key was not a defect before (`before`: the keys
    before, less the one fixed; None -- read from `x`). A FIX is an action mended and with nothing new. plan 76.4b: a call-through -- the native core reads every
    action in one crossing (`Regressor.fixes`); the oracle reads them one by one (`PyRegressor.fixes`)."""
    return R.fixes(x, goal, key, actions, future, entries, before)
