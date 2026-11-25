"""PLAN 79.55: A RULE LEMMA OVER THE BITS OF A STATE, CLOSED BY LEAN'S `bv_decide` (SYSTEM_SPEC 11 "One lemma machinery, used twice" and "Layer 2 as a port for games";
SYSTEM_SPEC 13 the decision of 2026-11-25 on solver proofs; piece: the long-term memory's statements; ledger rows 49, 56, 33).

THIS TEXT IS THE INTERFACE, written before the body. Game-free: a game, a lemma and a document of hypotheses are arguments; the module reads a lemma through `rule_lemma.py` and the
description through the reader (discovery/gdl.py) and names no relation, constant or sort of any description. It sits beside `rule_lemma_lean.py` (the obligation over the
description's relations, in the generator's inductive predicates, which this module does not touch) and `rule_lemma_prove.py`.

WHAT IT DOES. The facts of a state are Boolean variables, one for each fact a relation of the lemma can read (a fluent with its places filled by members of their sorts). Each relation
the lemma reads is UNFOLDED at ground places into a Boolean expression over them: a static relation (one that reads no fluent) is looked up in the reader's own table and is a constant;
a fluent is its variable; a derived relation is the OR over its clauses of the AND of the clause's literals, a negation being the negation of the expression of the ground atom (the
description is stratified, so the unfolding ends; a cycle among ground atoms is refused). The unfolding enumerates only what the description's literals bind: static literals bind
first, and a variable no literal binds is taken over the members of the sort the signature gives its place. Expressions are shared (hash-consed) and simplified by constants only.
A BINDING is one choice of members for the places of the lemma's two relations and of the variables of its condition. For each binding the statement is
    (the facts are any Booleans that meet the HYPOTHESES) and (the condition holds)  imply  (left = right)   [SAME]   or   (left -> right)   [IMPLIES]
and the Lean text carries it as one theorem closed by `bv_decide` (a SAT solver finds no counterexample and Lean's kernel verifies the solver's certificate). Shared expressions are
named by hypotheses `q = (expression)`, so the text is as long as the shared expression, not as long as its tree.
THE HYPOTHESES are the conditions on a state that the statement needs and the description does not state: a document of "at most one" groups (`{"fluent": f, "fixed": {place: member},
"key": [places]}`: among the facts of the fluent f whose places in `fixed` hold those members, at most one for each value of the places in `key`), read from an argument, and only the
groups whose facts occur in the statement are written. They are ASSUMPTIONS until shown to hold at every state a game reaches; they are written in the Lean text and listed in the report.
WHAT IS NOT CLAIMED. The statement over bits is made from the description by this program: the program is trusted, as the translator is, until the statement over bits is tied in Lean to
the statement over the description's relations (not built). A binding whose two expressions are constants is settled by this program's own evaluation and not by Lean; it is counted
and said (`settled`), never counted as closed by the solver.
THE AGREEMENT TEST that the expressions are the right ones: `agree` evaluates each binding's expressions at the bits of a corpus state and compares them with the rows of the reader's evaluator.
THE PLANTED FALSE LEMMA: a lemma in memory that is false is refused by the solver; `counterexample` reads the assignment the solver prints back as the facts of a state.

INTERFACE (who calls it: the tests; `python -m catspace.incubator.analyst.discovery.rule_lemma_bits <verb> <game> <lemma> [--hyps FILE]`, verbs plan | write | agree | run):
  `Dag`, `Grounder(prog)`                      the shared expressions and their unfolding; `Grounder.row(rel, args) -> node`
  `hypotheses_of(path_or_none) -> list`        the document of "at most one" groups
  `Bits(lemma, hyps)`                          one lemma over bits: `.bindings()` the choices; `.statement(binding) -> Statement`
  `Statement`                                  `.cond`, `.left`, `.right` (nodes), `.settled` (None, or True/False for a binding the program settles itself)
  `Bits.text(bindings, name_of=None) -> str`   the Lean text of the theorems of those bindings
  `Bits.write(dest, bindings) -> Path`         the text written; `path_of(lemma) -> Path` the place under lean/ideas/Ideas/Mined
  `run_lean(path, fuse) -> dict`               `lake env lean` under a fuse: {"ok", "secs", "timeout", "out"}
  `counterexample(out) -> list`                the variables the solver's message sets true, as facts
  `agree(bits, states, bindings) -> dict`      the test above
Failure modes: `Untranslatable` (a relation that reads an input the state does not hold, a compound term in a literal, a cycle among ground atoms, an unsafe clause); a Lean run that
times out is reported as such, never as closed.
"""
from __future__ import annotations

import itertools
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.incubator.analyst.discovery import rule_lemma as RL

ROOT = RL.ROOT
MINED = ROOT / "lean" / "ideas" / "Ideas" / "Mined"
LEAN_ROOT = ROOT / "lean" / "ideas"
INPUT_RELATIONS = ("does", "record")
LADDER_FROM = 3                           # a group of more facts than this is written as a running OR, not as every pair


class Untranslatable(ValueError):
    pass


# ------------------------------------------------------------------------------------------------------------------------------------------ the shared expressions

FALSE, TRUE = 0, 1


class Dag:
    """Boolean expressions, hash-consed. A node is an integer; 0 is false, 1 is true. Simplified by constants, flattening, duplicates and `x and not x` only."""

    def __init__(self):
        self.nodes: list = [("c", False), ("c", True)]
        self.intern: dict = {}
        self.names: dict = {}                 # variable name -> node

    def _mk(self, key) -> int:
        n = self.intern.get(key)
        if n is None:
            n = len(self.nodes)
            self.nodes.append(key)
            self.intern[key] = n
        return n

    def var(self, name: str) -> int:
        if name not in self.names:
            self.names[name] = self._mk(("v", name))
        return self.names[name]

    def neg(self, a: int) -> int:
        if a == FALSE:
            return TRUE
        if a == TRUE:
            return FALSE
        k = self.nodes[a]
        return k[1] if k[0] == "not" else self._mk(("not", a))

    def _join(self, kind: str, xs, unit: int, zero: int) -> int:
        out = set()
        for x in xs:
            if x == unit:
                continue
            if x == zero:
                return zero
            k = self.nodes[x]
            if k[0] == kind:
                out.update(k[1])
            else:
                out.add(x)
        if any(self.neg(x) in out for x in out):
            return zero
        if not out:
            return unit
        if len(out) == 1:
            return next(iter(out))
        return self._mk((kind, tuple(sorted(out))))

    def conj(self, xs) -> int:
        return self._join("and", xs, TRUE, FALSE)

    def disj(self, xs) -> int:
        return self._join("or", xs, FALSE, TRUE)

    def eval(self, a: int, true_vars, memo: dict) -> bool:
        if a in memo:
            return memo[a]
        k = self.nodes[a]
        if k[0] == "c":
            r = k[1]
        elif k[0] == "v":
            r = k[1] in true_vars
        elif k[0] == "not":
            r = not self.eval(k[1], true_vars, memo)
        elif k[0] == "and":
            r = all(self.eval(x, true_vars, memo) for x in k[1])
        else:
            r = any(self.eval(x, true_vars, memo) for x in k[1])
        memo[a] = r
        return r

    def cone(self, roots) -> list:
        """The nodes the roots reach, arguments before the nodes that use them."""
        seen, order = set(), []
        stack = [(r, False) for r in roots]
        while stack:
            n, done = stack.pop()
            if done:
                order.append(n)
                continue
            if n in seen or n in (FALSE, TRUE):
                continue
            seen.add(n)
            stack.append((n, True))
            k = self.nodes[n]
            if k[0] == "not":
                stack.append((k[1], False))
            elif k[0] in ("and", "or"):
                for x in k[1]:
                    stack.append((x, False))
        return order


# ------------------------------------------------------------------------------------------------------------------------------------------ the unfolding

def var_name(rel: str, args) -> str:
    return "f_" + re.sub(r"[^A-Za-z0-9]", "_", rel.split(".", 1)[-1] + "_" + "_".join(str(a) for a in args))


class Grounder:
    """A relation at ground places as a Boolean expression over the facts of a state (the module's description)."""

    def __init__(self, prog: GDL.Program, assume: "dict | None" = None):
        self.prog = prog
        self.assume = assume or {}            # variable name -> True / False: facts taken as known (a case of a split), unfolded with their value
        self.dag = Dag()
        self.memo: dict = {}
        self.busy: set = set()
        self.facts: dict = {}                 # variable name -> (fluent relation, args)
        self.calls = 0

    # ---- the sorts
    def domain(self, sort: str) -> list:
        mem = self.prog.sorts.get(sort)
        if mem is None or any(not isinstance(m, str) for m in mem):
            raise Untranslatable(f"the sort {sort} is not a list of constants")
        return list(mem)

    def sig_of(self, rel: str) -> tuple:
        s = self.prog.signature(rel)
        if s is None:
            raise Untranslatable(f"{rel} has no signature")
        return s

    def is_static(self, rel: str) -> bool:
        return not rel.startswith("true.") and not self.prog.reads_input.get(rel, False) and rel not in INPUT_RELATIONS

    # ---- one atom
    def row(self, rel: str, args: tuple) -> int:
        key = (rel, args)
        got = self.memo.get(key)
        if got is not None:
            return got
        self.calls += 1
        if rel.startswith("true."):
            sig = self.sig_of(rel)
            if len(sig) != len(args) or any(a not in self.domain(s) for a, s in zip(args, sig)):
                r = FALSE
            else:
                name = var_name(rel, args)
                self.facts[name] = (rel, args)
                r = (TRUE if self.assume[name] else FALSE) if name in self.assume else self.dag.var(name)
        elif rel in INPUT_RELATIONS or rel.startswith(("next.", "init.")):
            raise Untranslatable(f"{rel} reads an input the state does not hold")
        elif self.is_static(rel):
            r = TRUE if args in self.prog.static.get(rel, GDL.Relation()).facts else FALSE
        else:
            if key in self.busy:
                raise Untranslatable(f"a cycle among ground atoms at {rel}{args}")
            self.busy.add(key)
            ors = []
            for cl in self.prog.rules_of.get(rel, ()):
                env = self.unify_head(cl.head[1], args)
                if env is not None:
                    self.solve(list(cl.body), env, [], ors)
            self.busy.discard(key)
            r = self.dag.disj(ors)
        self.memo[key] = r
        return r

    def unify_head(self, pattern, args):
        env: dict = {}
        if len(pattern) != len(args):
            return None
        for p, a in zip(pattern, args):
            if isinstance(p, tuple):
                raise Untranslatable("a compound term in a clause head")
            if GDL.is_var(p):
                if env.setdefault(p, a) != a:
                    return None
            elif p != a:
                return None
        return env

    # ---- one clause body
    def val(self, t, env):
        if isinstance(t, tuple):
            raise Untranslatable("a compound term in a literal")
        if GDL.is_var(t):
            return env.get(t)
        return t

    def solve(self, lits: list, env: dict, acc: list, out: list) -> None:
        if not lits:
            out.append(self.dag.conj(acc))
            return
        best, bi = None, -1
        for i, l in enumerate(lits):
            if l[0] == "distinct":
                if self.val(l[1], env) is not None and self.val(l[2], env) is not None:
                    pri = (0, 0)
                else:
                    continue
            else:
                vals = [self.val(a, env) for a in l[2]]
                unbound = sum(v is None for v in vals)
                if l[0] == "neg":
                    if unbound:
                        continue
                    pri = (1, 1)
                elif self.is_static(l[1]):
                    if unbound == 0:
                        pri = (0, 1)
                    else:
                        n = self.static_count(l[1], l[2], env)
                        pri = (0, 2) if n <= 1 else (2, n)
                else:
                    pri = (1, 0) if unbound == 0 else (3, unbound)
            if best is None or pri < best:
                best, bi = pri, i
        if best is None:
            raise Untranslatable("a clause with a negation or a distinct over a variable no literal binds")
        l = lits[bi]
        rest = lits[:bi] + lits[bi + 1:]
        if l[0] == "distinct":
            if self.val(l[1], env) != self.val(l[2], env):
                self.solve(rest, env, acc, out)
            return
        if l[0] == "neg":
            n = self.dag.neg(self.row(l[1], tuple(self.val(a, env) for a in l[2])))
            if n != FALSE:
                self.solve(rest, env, acc + [n], out)
            return
        rel, targs = l[1], l[2]
        if self.is_static(rel):
            for row in self.static_rows(rel, targs, env):
                env2 = self.bind(targs, row, env)
                if env2 is not None:
                    self.solve(rest, env2, acc, out)
            return
        sig = self.sig_of(rel)
        free = []
        for a, s in zip(targs, sig):
            if GDL.is_var(a) and a not in env and a not in [f[0] for f in free]:
                free.append((a, s))
        # a free variable that no other literal reads and that occurs once here is existential: the literal is the OR over its members, not a branch of the clause
        elsewhere = [a for r in rest for a in (r[2] if r[0] != "distinct" else r[1:]) if isinstance(a, str)]
        once = [(v, s) for v, s in free if v not in elsewhere and sum(a == v for a in targs) == 1]
        multi = [(v, s) for v, s in free if (v, s) not in once]
        for combo in itertools.product(*[self.domain(s) for _, s in multi]):
            env2 = dict(env)
            env2.update({v: c for (v, _), c in zip(multi, combo)})
            if once:
                n = self.dag.disj(self.row(rel, tuple(self.val(a, {**env2, **{v: c for (v, _), c in zip(once, inner)}}) for a in targs))
                                  for inner in itertools.product(*[self.domain(s) for _, s in once]))
            else:
                n = self.row(rel, tuple(self.val(a, env2) for a in targs))
            if n != FALSE:
                self.solve(rest, env2, acc + [n], out)

    def static_rows(self, rel, targs, env):
        r = self.prog.static.get(rel)
        if r is None:
            return ()
        pos = tuple(i for i, a in enumerate(targs) if self.val(a, env) is not None)
        return r.lookup(pos, tuple(self.val(targs[i], env) for i in pos))

    def static_count(self, rel, targs, env) -> int:
        return len(self.static_rows(rel, targs, env))

    def bind(self, targs, row, env):
        env2 = dict(env)
        for a, v in zip(targs, row):
            if GDL.is_var(a):
                if env2.setdefault(a, v) != v:
                    return None
            elif a != v:
                return None
        return env2


# ------------------------------------------------------------------------------------------------------------------------------------------ one lemma

class Statement:
    __slots__ = ("binding", "cond", "left", "right", "settled")

    def __init__(self, binding, cond, left, right, settled):
        self.binding, self.cond, self.left, self.right, self.settled = binding, cond, left, right, settled


def hypotheses_of(path) -> list:
    if not path:
        return []
    doc = json.loads(Path(path).read_text())
    return doc["hypotheses"] if isinstance(doc, dict) else doc


def ident(lemma: RL.Lemma) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", f"rule_lemma_bits_{lemma.game}_{lemma.name}")


def path_of(lemma: RL.Lemma) -> Path:
    return MINED / (ident(lemma) + ".lean")


class Bits:
    """One lemma over bits."""

    def __init__(self, lemma: RL.Lemma, hyps: "list | None" = None, assume: "dict | None" = None, prog: "GDL.Program | None" = None):
        self.lemma = lemma
        self.prog = prog or RL.program(lemma)
        self.assume = assume or {}
        self.g = Grounder(self.prog, self.assume)
        self.hyps = hyps or []
        lf, rg = lemma.left, lemma.right
        self.lsig = self.g.sig_of(lf["relation"])
        self.rsig = self.g.sig_of(rg["relation"])
        if len(self.lsig) != lf["places"] or len(self.rsig) != rg["places"] or lf["places"] != rg["places"]:
            raise Untranslatable("the two relations of a lemma take the same number of places, the signatures say " f"{self.lsig} and {self.rsig}")
        self.cond_vars: list = []
        self.cond_sorts: dict = {}
        for l in lemma.condition:
            if l["relation"] == RL.DIFFER:
                continue
            rel = RL.fluent_or_relation(self.prog, l["relation"])
            for a, s in zip(l["args"], self.g.sig_of(rel)):
                if a.startswith("?") and a not in self.cond_sorts:
                    self.cond_vars.append(a)
                    self.cond_sorts[a] = s

    # ---- the choices
    def bindings(self) -> list:
        """Every choice of members for the places of the left relation and the variables of the condition."""
        doms = [self.g.domain(s) for s in self.lsig] + [self.g.domain(self.cond_sorts[v]) for v in self.cond_vars]
        n = len(self.lsig)
        return [(c[:n], c[n:]) for c in itertools.product(*doms)]

    def statement(self, binding) -> Statement:
        places, cvals = binding
        g, d = self.g, self.g.dag
        env = dict(zip(self.cond_vars, cvals))
        cond = TRUE
        parts = []
        for l in self.lemma.condition:
            if l["relation"] == RL.DIFFER:
                x, y = [env.get(a, a) for a in l["args"]]
                parts.append(TRUE if x != y else FALSE)
                continue
            rel = RL.fluent_or_relation(self.prog, l["relation"])
            n = g.row(rel, tuple(env.get(a, a) for a in l["args"]))
            parts.append(d.neg(n) if l.get("neg") else n)
        cond = d.conj(parts)
        left = g.row(self.lemma.left["relation"], tuple(places))
        right = g.row(self.lemma.right["relation"], tuple(places))
        settled = None
        if cond == FALSE:
            settled = True
        elif left in (FALSE, TRUE) and right in (FALSE, TRUE):
            settled = (left == right) if self.lemma.form == "SAME" else (left == FALSE or right == TRUE)
        elif self.lemma.form == "IMPLIES" and left == FALSE:
            settled = True
        return Statement(binding, cond, left, right, settled)

    # ---- the hypotheses
    def hypothesis_lines(self, present: set) -> list:
        """The "at most one" groups restricted to the facts that occur: [(group label, [variable names])]."""
        out = []
        for h in self.hyps:
            fl = h["fluent"]
            rel = "true." + fl
            sig = self.g.sig_of(rel)
            fixed = {int(k): v for k, v in h.get("fixed", {}).items()}
            key = h.get("key", [])
            groups: dict = {}
            for name, (r, args) in self.g.facts.items():
                if r != rel or name not in present:
                    continue
                if any(args[i] != v for i, v in fixed.items()):
                    continue
                groups.setdefault(tuple(args[i] for i in key), []).append(name)
            for k, names in sorted(groups.items()):
                if len(names) > 1:
                    out.append((f"{fl} fixed {fixed} key {list(key)}={list(k)}", sorted(names)))
        return out

    # ---- the Lean text
    def expr(self, n: int, name_of: dict, top: bool = False) -> str:
        if n == FALSE:
            return "false"
        if n == TRUE:
            return "true"
        if n in name_of and not top:
            return name_of[n]
        k = self.g.dag.nodes[n]
        if k[0] == "v":
            return k[1]
        if k[0] == "not":
            return "!" + self.expr(k[1], name_of)
        op = " && " if k[0] == "and" else " || "
        return "(" + op.join(self.expr(x, name_of) for x in k[1]) + ")"

    def theorem(self, st: Statement, label: str) -> str:
        d = self.g.dag
        form = self.lemma.form
        roots = [st.cond, st.left, st.right]
        order = d.cone(roots)
        uses: dict = {}
        for n in order:
            k = d.nodes[n]
            if k[0] == "not":
                uses[k[1]] = uses.get(k[1], 0) + 1
            elif k[0] in ("and", "or"):
                for x in k[1]:
                    uses[x] = uses.get(x, 0) + 1
        for r in roots:
            uses[r] = uses.get(r, 0) + 1
        name_of: dict = {}
        shared = []
        for n in order:
            k = d.nodes[n]
            if k[0] in ("and", "or", "not") and (uses.get(n, 0) >= 2):
                name_of[n] = f"q{len(shared)}"
                shared.append(n)
        varnames = sorted({d.nodes[n][1] for n in order if d.nodes[n][0] == "v"})
        hyps = self.hypothesis_lines(set(varnames))
        binders = []
        if varnames:
            binders.append("(" + " ".join(varnames) + " : Bool)")
        if shared:
            binders.append("(" + " ".join(name_of[n] for n in shared) + " : Bool)")
        for i, n in enumerate(shared):
            binders.append(f"(hq{i} : {name_of[n]} = {self.expr(n, name_of, top=True)})")
        aux, ladder = [], []
        for i, (_label, names) in enumerate(hyps):
            if len(names) <= LADDER_FROM:
                for j, (a, b) in enumerate(itertools.combinations(names, 2)):
                    ladder.append(f"(hy{i}x{j} : (!({a} && {b})) = true)")
                continue
            # at most one of many: a running OR `ld` (defined by its own equations) and the test that no fact comes after one that held
            prev = names[0]
            for k, x in enumerate(names[1:], 1):
                aux.append(f"ld{i}_{k}")
                ladder.append(f"(hl{i}s{k} : ld{i}_{k} = ({prev} || {x}))")
                ladder.append(f"(hy{i}x{k} : (!({prev} && {x})) = true)")
                prev = f"ld{i}_{k}"
        if aux:
            binders.append("(" + " ".join(aux) + " : Bool)")
        binders.extend(ladder)
        binders.append(f"(hcond : {self.expr(st.cond, name_of)} = true)")
        goal = (f"({self.expr(st.left, name_of)} == {self.expr(st.right, name_of)}) = true" if form == "SAME"
                else f"(!{self.expr(st.left, name_of)} || {self.expr(st.right, name_of)}) = true")
        return f"-- binding {label}: {len(varnames)} variables, {len(shared)} shared expressions, {len(hyps)} at-most-one groups\n" \
               f"theorem {ident(self.lemma)}_b{label.split(' ')[0]} {' '.join(binders)} :\n    {goal} := by\n  bv_decide\n"

    def header(self, count: int) -> str:
        lm = self.lemma
        hy = "\n".join(f"--   {json.dumps(h, sort_keys=True)}" for h in self.hyps) or "--   none"
        return (f"/- PLAN 79.55: the lemma {lm.name} ({lm.form}: {lm.left['relation']} / {lm.right['relation']}) of the game {lm.game}, stated over the bits of a state by\n"
                f"   discovery/rule_lemma_bits.py and closed by bv_decide, one theorem a binding ({count} here). GENERATED: do not edit.\n"
                f"   Each variable is one fact of a state. Each theorem says: for any bits that meet the hypotheses named, if the condition holds then the two sides agree.\n"
                f"   HYPOTHESES (assumptions, not shown to hold at every reached state):\n{hy}\n"
                f"   TRUST: the statement is made from the description by that program, which is trusted; it is not tied in Lean to the description's relations. -/\n"
                "import Std.Tactic.BVDecide\n\nset_option maxRecDepth 100000\nset_option linter.unusedVariables false\nset_option maxHeartbeats 0\n\n")

    def text(self, bindings: list) -> str:
        out, count = [], 0
        for b in bindings:
            st = self.statement(b)
            if st.settled is not None:
                continue
            count += 1
            label = f"{self.label_of(b)}"
            out.append(self.theorem(st, label))
        return self.header(count) + "\n".join(out)

    def label_of(self, b) -> str:
        """`<index> <the places and the condition's members>`: the index of the binding is its position in `bindings()`, found by the members."""
        places, cvals = b
        return "_".join(list(places) + list(cvals)) + " (" + " ".join(list(places) + list(cvals)) + ")"

    def write(self, dest: Path, bindings: list) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self.text(bindings))
        return dest


# ------------------------------------------------------------------------------------------------------------------------------------------ Lean

def run_lean(path: Path, fuse: int) -> dict:
    """`lake env lean <path>` in the Lean project under a fuse in seconds; on the fuse the whole process group (lake, lean, the solver) is killed."""
    import os
    import signal
    t0 = time.time()
    p = subprocess.Popen(["lake", "env", "lean", str(path)], cwd=str(LEAN_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=fuse)
        return {"ok": p.returncode == 0, "secs": time.time() - t0, "timeout": False, "out": out}
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        return {"ok": False, "secs": time.time() - t0, "timeout": True, "out": out or ""}


def counterexample(out: str) -> list:
    """The variables the solver's message sets to true, as (fluent, args) facts, for a state printed back; variables named by `var_name`."""
    facts = []
    for m in re.finditer(r"^\s*(f_[A-Za-z0-9_]+) = true\s*$", out, flags=re.M):
        facts.append(m.group(1))
    return facts


# ------------------------------------------------------------------------------------------------------------------------------------------ the agreement with the evaluator

def agree(bits: Bits, states, bindings: list, limit_states: "int | None" = None) -> dict:
    """At each state: the variables true are the state's facts; each binding's left and right expressions evaluated equal the evaluator's row of the lemma's relations; the condition
    and the hypotheses evaluated equal what the evaluator and the state say. Returns the counts."""
    lm, prog, d = bits.lemma, bits.prog, bits.g.dag
    stmts = [bits.statement(b) for b in bindings]
    present = {n for st in stmts for n in d.cone([st.cond, st.left, st.right]) if d.nodes[n][0] == "v"}
    groups = bits.hypothesis_lines({d.nodes[n][1] for n in present})
    wanted = tuple(sorted({lm.left["relation"], lm.right["relation"], *RL.condition_relations(lm, prog)}))
    res = {"states": 0, "bindings": len(stmts), "compared": 0, "disagree": 0, "cond_held": 0, "hyp_broken": 0, "left_true": 0, "examples": []}
    for label, state in states:
        if limit_states is not None and res["states"] >= limit_states:
            break
        db = prog.run(state, wanted=wanted)
        lrows = db.facts(lm.left["relation"])
        rrows = db.facts(lm.right["relation"])
        true_vars = {var_name(rel, args) for rel, args in state}
        memo: dict = {}
        res["states"] += 1
        for (places, cvals), st in zip(bindings, stmts):
            lv, rv = d.eval(st.left, true_vars, memo), d.eval(st.right, true_vars, memo)
            res["compared"] += 1
            res["left_true"] += lv
            if lv != (tuple(places) in lrows) or rv != (tuple(places) in rrows):
                res["disagree"] += 1
                if len(res["examples"]) < 3:
                    res["examples"].append((label, places, lv, rv, tuple(places) in lrows, tuple(places) in rrows))
        for _lab, names in groups:
            if sum(n in true_vars for n in names) > 1:
                res["hyp_broken"] += 1
    return res


# ------------------------------------------------------------------------------------------------------------------------------------------ runs

def record_path(lemma: RL.Lemma) -> Path:
    """Where the record of a lemma closed by the solver is kept: beside the lemma's document, in its `bits` folder (not in the document)."""
    return RL.lemma_dir(lemma.game) / "bits" / f"{lemma.name}.proof.json"


def file_hash(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def standing(lemma: RL.Lemma) -> tuple:
    """The standing the solver's proof gives a lemma, read from the record beside its document and not from the document: ("SOLVER-PROVED", detail) only while the record names a Lean
    file that exists, holds no gap, no `native_decide` and no axiom, hashes to the record's hash, and says every binding the solver was given was closed; otherwise ("NONE", the cause).
    It is no standing of `rule_lemma.standing_of` (extending that function changes a source every lemma certificate is keyed on); it is the record a later extension reads."""
    rp = record_path(lemma)
    if not rp.is_file():
        return "NONE", "no record"
    rec = json.loads(rp.read_text())
    f = ROOT / rec["lean_file"]
    if not f.is_file():
        return "NONE", f"the file {rec['lean_file']} is missing"
    code = re.sub(r"/-.*?-/", "", f.read_text(), flags=re.S)
    if "sorry" in code or "native_decide" in code or re.search(r"^\s*axiom\b", code, re.M):
        return "NONE", "the file holds a gap, native_decide or an axiom"
    if file_hash(f) != rec["file_hash"]:
        return "NONE", "the file is not the one the record names"
    if code.count("bv_decide") < rec["theorems"]:
        return "NONE", "fewer proofs than theorems"
    if rec["closed"] != rec["to_solver"] or rec["failed"] or rec["fuse"]:
        return "NONE", "not every binding was closed"
    return "SOLVER-PROVED", f"{rec['closed']} bindings closed by bv_decide, {rec['settled']} settled by the writer, {len(rec['hypotheses'])} hypotheses"


def state_text(bits: Bits, names) -> str:
    """The facts a counterexample sets true, as the fixture format writes a state: `(f a b) (g c)`."""
    out = []
    for n in sorted(names):
        if n in bits.g.facts:
            rel, args = bits.g.facts[n]
            out.append("(" + " ".join([rel.split(".", 1)[1], *args]) + ")")
    return " ".join(out)


def sample_of(items: list, k: int) -> list:
    if k >= len(items):
        return items
    import random
    return [items[i] for i in sorted(random.Random(55).sample(range(len(items)), k))]


def run_each(bits: Bits, bindings: list, dest_dir: Path, jobs: int, fuse: int) -> list:
    """One Lean file a binding (the binding alone), `jobs` at once (at most four), each under the fuse: [(binding, ok, secs, timeout, out)]."""
    from concurrent.futures import ThreadPoolExecutor
    dest_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for b in bindings:
        st = bits.statement(b)
        if st.settled is None:
            todo.append(b)
    paths = [bits.write(dest_dir / (bits.label_of(b).split(" ")[0] + ".lean"), [b]) for b in todo]
    with ThreadPoolExecutor(max_workers=min(jobs, 4)) as ex:
        res = list(ex.map(lambda p: run_lean(p, fuse), paths))
    return [(b, r["ok"], r["secs"], r["timeout"], r["out"]) for b, r in zip(todo, res)]


def baseline_secs() -> float:
    """The wall of a Lean file that only imports the solver's module: the start-up every run below carries."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "empty.lean"
        p.write_text("import Std.Tactic.BVDecide\n")
        return run_lean(p, 300)["secs"]


def summary(label: str, runs: list) -> str:
    secs = [r[2] for r in runs]
    ok = sum(1 for r in runs if r[1])
    to = sum(1 for r in runs if r[3])
    return (f"OBS 79.55 {label}: bindings run {len(runs)}, closed {ok}, failed {len(runs) - ok - to}, fuse {to}, wall sum {sum(secs):.1f} s, "
            f"max {max(secs) if secs else 0:.1f} s, mean {sum(secs) / max(1, len(secs)):.1f} s")


def main(argv: list) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    verb, game, name = argv[0], argv[1], argv[2]
    opt = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d
    hyps = hypotheses_of(opt("--hyps"))
    lemma = RL.load(game, name)
    bits = Bits(lemma, hyps)
    bs = bits.bindings()
    only = opt("--only")
    if only:
        want = tuple(only.split(","))
        bs = [b for b in bs if tuple(b[0]) + tuple(b[1]) == want]
    sample = int(opt("--sample", 0))
    jobs = int(opt("--jobs", 1))
    fuse = int(opt("--fuse", 600))
    t0 = time.time()
    sts = [(b, bits.statement(b)) for b in bs]
    todo = [b for b, s in sts if s.settled is None]
    settled = len(bs) - len(todo)
    ground = time.time() - t0
    print(f"OBS 79.55 plan {game} {name}: bindings {len(bs)}, settled by the program {settled}, to the solver {len(todo)}, expression nodes {len(bits.g.dag.nodes)}, facts {len(bits.g.facts)}, writer seconds {ground:.1f}")
    if verb == "plan":
        return 0
    if sample:
        todo = sample_of(todo, sample)
    if verb == "write":
        print(bits.write(path_of(lemma), todo))
        return 0
    if verb == "each":
        from catspace.io import paths
        d = Path(paths.experiment("rule_lemma_bits", game, name, "each.txt")).parent
        print(f"OBS 79.55 start-up of a Lean file with the solver's module imported: {baseline_secs():.1f} s")
        runs = run_each(bits, todo, d, jobs, fuse)
        print(summary(f"each {game} {name}", runs))
        slow = sorted(runs, key=lambda r: -r[2])[:5]
        for b, ok, secs, to, out in slow:
            print(f"OBS 79.55 slowest {bits.label_of(b).split(' ')[0]}: {secs:.1f} s ok {ok}")
        for b, ok, secs, to, out in runs:
            if not ok:
                print(f"NOT CLOSED {bits.label_of(b).split(' ')[0]} timeout {to}: {out[:300]!r}")
                print(f"  counterexample: {state_text(bits, counterexample(out))}")
        return 0 if all(r[1] for r in runs) else 1
    if verb == "whole":
        p = bits.write(path_of(lemma), todo)
        r = run_lean(p, fuse)
        print(f"OBS 79.55 whole {game} {name}: theorems {len(todo)}, ok {r['ok']}, timeout {r['timeout']}, wall {r['secs']:.1f} s, text {p.stat().st_size} bytes")
        if not r["ok"]:
            print(r["out"][:600])
        if r["ok"] and not sample and not only and "--record" in argv:
            rec = {"lemma": lemma.name, "game": game, "form": lemma.form, "kind": "bv_decide: a SAT solver's certificate, checked in Lean (the proof rests on the axiom Lean.ofReduceBool, which `#print axioms` lists)",
                   "lean_file": str(p.relative_to(ROOT)), "file_hash": file_hash(p), "bindings": len(bs), "settled": settled, "to_solver": len(todo), "theorems": len(todo),
                   "closed": len(todo), "failed": 0, "fuse": 0, "wall_secs": round(r["secs"], 1), "hypotheses": hyps,
                   "trust": "the statement over bits is made from the description by rule_lemma_bits.py (trusted, not tied in Lean to the description's relations); bindings the writer settles by constants are not checked by Lean"}
            rp = record_path(lemma)
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
            print(f"record written {rp}: {standing(lemma)}")
        return 0 if r["ok"] else 1
    if verb == "agree":
        from catspace.incubator.analyst.discovery import rule_lemma_certify as RC
        games = int(opt("--games", 2))
        every = bs if "--all" in argv else todo
        res = agree(bits, RC.corpus_states(lemma, games=games), every, int(opt("--states", 0)) or None)
        print(f"OBS 79.55 agree {game} {name}: states {res['states']}, bindings {res['bindings']}, comparisons {res['compared']}, disagreements {res['disagree']}, "
              f"left true {res['left_true']}, hypotheses broken at a state {res['hyp_broken']}, examples {res['examples']}")
        return 0 if res["disagree"] == 0 else 1
    print(f"unknown verb {verb}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
