"""plan 76.0 (c) GDL -> LEAN, BY CODE: the clauses of rules/chess.gdl translated to Lean definitions over the rule kernel's types (chess-lean's `Chess.Position`,
`Chess.Color`, `Chess.PieceKind`; squares as their file and rank, `Fin 8` each). Datalog's least fixpoint IS an inductive definition: every relation becomes an
INDUCTIVE PREDICATE, one constructor per clause (its body literals the constructor's premises, its body-only variables the constructor's implicit arguments: the
existentials). Chosen over decidable functions because the translation is then one rule per clause shape with no termination argument (a recursive relation such as
`slide` is an inductive with a recursive premise, and Lean checks its positivity), and a proof about it is an induction on the derivation -- the shape of the Datalog
evaluation itself. Decidability is not needed by the proofs; the cross-play is where the spec is run.

THE CLAUSE SHAPES the translator handles (each counted in `--survey`): (1) a ground fact -> a constructor with no premise; (2) a rule with positive literals -> premises;
(3) recursion through the relation itself -> a recursive premise (the arguments every clause passes through unchanged become the inductive's PARAMETERS, a uniform
prefix); (4) `(not R ...)` of a relation of a lower stratum -> a `¬` premise; (5) `(distinct x y)` -> `x ≠ y` (ground: decided here, the clause kept or dropped); (6)
constants in a head -> the indices' constants; (7) SPECIALISATION: a call with a constant at an argument where some head of the callee has a constant (the callee is
indexed there: `attack_by` by the unit kind) calls the callee specialised to that constant (`attack_by rook` -> `attackByRook`: the clauses whose head unifies,
the constant substituted); (8) a state fluent `(true (f ...))` through the BINDING table (`cell` -> `p.board (f, r) = some ⟨c, k⟩`, `control` -> `p.turn = c`).

PLAN 76.1 (b) THE NAMED EXTENSIONS (rules/chess.gdl's header): (1) SORTS -- every argument's type is its slot's DECLARED sort (`Program.signatures`; the
spec type-checks), a constant read by its slot's sort (the rank 1 is Fin 8's 0, the clock's 1 is the Nat 1): no inference, no clash; (2) THE BINDING TABLE is the
ontology shelf's (`bind` of its entries: the sorts' Lean types and constants, the fluents' templates -- the e.p. square, the castling rights as the kernel's set,
the saturating clock); (3) MOVE TERMS -- a compound term of the `move` sort is the shelf's `Ideas.Spec.Binding.MoveTerm` (in a head: `legal`'s index; in `does`:
the joint move); (9) THE INPUTS -- `does` through `Ideas.Spec.Binding.Does p mv` and `record` through `Ideas.Spec.Binding.seen rec p`: a relation that reads them
(directly or through what it calls) takes the parameters `(mv : MoveTerm)` / `(rec : Record ..)` after the position; (4) THE GAME LAYER -- `init.*`, `legal`,
`next.*`, `terminal`, `goal` translate like any relation, and `applyMoveSpec p mv q` (q is the position the `next` facts describe, through the fluents' bindings) is
generated from the fluent table -- its functionality and its equality with the kernel's `applyMove` are proved in Ideas.Spec.NextEquiv.

  .venv/bin/python scripts/gdl_to_lean.py rook [--out lean/ideas/Ideas/Spec/Rook.lean]      # the rook's clauses (its attack, its move, its capture and what they read)
  .venv/bin/python scripts/gdl_to_lean.py spec [--out lean/ideas/Ideas/Spec/Chess.lean]     # plan 76.1 (b): the whole spec, and applyMoveSpec
  .venv/bin/python scripts/gdl_to_lean.py survey [--lean <file>]                            # every relation of the spec: translated, or why not (OBS lines)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.incubator.analyst.discovery import ontology as ON
from catspace.io import paths

SPEC = paths.REPO_ROOT / "rules" / "chess.gdl"
OUT = paths.REPO_ROOT / "lean" / "ideas" / "Ideas" / "Spec" / "Rook.lean"
OUT_SPEC = paths.REPO_ROOT / "lean" / "ideas" / "Ideas" / "Spec" / "Chess.lean"
ROOK_ROOTS = [("attack_by", ("rook", None, None, None, None, None)), ("move_by", ("rook", None, None, None, None, None)),
              ("capture", ("rook", None, None, None, None, None))]
PARAMS = {"p": "(p : Chess.Position)", "mv": "(mv : Ideas.Spec.Binding.MoveTerm)", "rec": "(rec : Ideas.Game.Record Chess.Position Chess.RepetitionKey)"}
PARAM_ORDER = ("p", "mv", "rec")


class Untranslatable(Exception):
    pass


RESERVED = {"p", "mv", "rec", "q", "at", "by", "do", "then", "else", "if", "end", "let", "have", "show", "from", "fun", "match", "with", "in", "Type", "Prop", "open"}


def lean_var(v: str) -> str:
    """A GDL variable's Lean name (`?f1` -> `f1`; a name Lean or this file reserves gets a prime: `p` is the position)."""
    n = camel(v[1:])
    return n + "'" if n in RESERVED else n


def camel(s: str, upper: bool = False) -> str:
    parts = [p for p in s.replace(".", "_").split("_") if p]
    out = "".join(p[:1].upper() + p[1:] for p in parts)
    return out if upper else out[:1].lower() + out[1:]


def binding_table(store: "dict | None" = None) -> tuple:
    """THE BINDING TABLE, read from the ontology shelf (76.1 (b) (2)): (sort -> its bind: a Lean type and the constants' terms, or a generated enumeration's name,
    or a term sort's functor templates), (fluent or input relation -> (template, the parameters it reads))."""
    store = store if store is not None else ON.load()
    sorts, rels = {}, {}
    for e in store.values():
        b = e.get("bind")
        if not b:
            continue
        if e["kind"] == "OBJECT" and isinstance(e["sort"], str):
            sorts[e["sort"]] = b
        elif e["kind"] == "RELATION":
            g = e["gdl"]
            rel = ("true." + g["fluent"]) if "fluent" in g else g["relation"]
            rels[rel] = (b["template"], tuple(b["params"]))
    return sorts, rels


class Translator:
    def __init__(self, text: str, namespace: str = "Ideas.Spec.Rook", store: "dict | None" = None):
        self.prog = GDL.Program(text, "spec")
        bad = self.prog.typecheck()
        if bad:
            raise Untranslatable(f"the spec does not type-check: {bad[:3]}")
        self.namespace = namespace
        self.clauses = self.prog.clauses
        self.sort_bind, self.rel_bind = binding_table(store)
        self.by_head = defaultdict(list)
        for c in self.clauses:
            self.by_head[c.head[0]].append(c)
        # a relation is INDEXED at a position when some head of it has a constant there
        self.indexed = defaultdict(set)
        for rel, cs in self.by_head.items():
            for c in cs:
                for i, a in enumerate(c.head[1]):
                    if isinstance(a, str) and not GDL.is_var(a):
                        self.indexed[rel].add(i)
        self.items = {}                     # (rel, spec) -> dict(name, clauses (head args, body), ...)
        self.order = []
        self.shapes = defaultdict(int)
        self.rename = {}
        self.enums_used = set()
        self.uses_binding = False

    # -------------------------------------------------------------------------------------------- sorts
    def slot_sorts(self, rel: str, spec: tuple) -> tuple:
        sig = self.prog.signature(rel)
        if sig is None:
            raise Untranslatable(f"no signature: {rel}")
        return tuple(s for s, c in zip(sig, spec) if c is None)

    def lean_type(self, sort: str) -> str:
        b = self.sort_bind.get(sort)
        if b is None:
            raise Untranslatable(f"a sort with no binding in the shelf: {sort}")
        if "enum" in b:
            self.enums_used.add(sort)
            return b["enum"]
        if b["type"].startswith("Ideas.Spec.Binding."):
            self.uses_binding = True
        return b["type"]

    def lean_const(self, c: str, sort: str) -> str:
        b = self.sort_bind.get(sort)
        if b is None:
            raise Untranslatable(f"a sort with no binding in the shelf: {sort}")
        if "enum" in b:
            self.enums_used.add(sort)
            return f"{b['enum']}.{camel(c)}"
        if "constants" in b:
            if b["type"].startswith("Ideas.Spec.Binding."):
                self.uses_binding = True
            return b["constants"][c]
        if "terms" in b:
            self.uses_binding = True
            return b["terms"][c]
        if b["type"] == "Nat":
            return c
        raise Untranslatable(f"no Lean term for the constant {c} of the sort {sort}")

    def var_sorts(self, head_rel: str, head_spec: tuple, head: tuple, body: list) -> dict:
        """The sort of every variable of one clause, by the slots it fills (the spec type-checks, so one sort each)."""
        env: dict = {}

        def walk(t, sort):
            if GDL.is_var(t):
                env.setdefault(t, sort)
            elif isinstance(t, tuple):
                pat = next(m for m in self.prog.sorts[sort] if isinstance(m, tuple) and m[0] == t[0] and len(m) == len(t))
                for a, s in zip(t[1:], pat[1:]):
                    walk(a, s)
        for a, s in zip(head, self.slot_sorts(head_rel, head_spec)):
            walk(a, s)
        for lit in body:
            if lit[0] == "distinct":
                continue
            kind, r2, s2, a2 = lit
            for a, s in zip(a2, self.slot_sorts(r2, s2) if s2 is not None else self.prog.signature(r2)):
                walk(a, s)
        for lit in body:
            if lit[0] == "distinct":
                x, y = lit[1], lit[2]
                if GDL.is_var(x) and x not in env and y in env:
                    env[x] = env[y]
                if GDL.is_var(y) and y not in env and x in env:
                    env[y] = env[x]
        return env

    def render(self, t, sort: str, ci_key, env: dict) -> str:
        if GDL.is_var(t):
            return self.rename.get(ci_key, {}).get(t) or lean_var(t)
        if isinstance(t, str):
            return self.lean_const(t, sort)
        b = self.sort_bind.get(sort) or {}
        if "terms" not in b:
            raise Untranslatable(f"a compound term of a sort with no term binding: {GDL.term_text(t)}")
        pat = next(m for m in self.prog.sorts[sort] if isinstance(m, tuple) and m[0] == t[0] and len(m) == len(t))
        self.uses_binding = True
        return b["terms"][t[0]].format(*[self.render(a, s, ci_key, env) for a, s in zip(t[1:], pat[1:])])

    # -------------------------------------------------------------------------------------------- specialisation
    def spec_of_call(self, rel: str, args: tuple) -> tuple:
        return tuple(a if (i in self.indexed[rel] and isinstance(a, str) and not GDL.is_var(a)) else None for i, a in enumerate(args))

    def name(self, rel: str, spec: tuple) -> str:
        return camel(rel) + "".join(camel(c, upper=True) for c in spec if c is not None)

    def request(self, rel: str, spec: tuple) -> None:
        key = (rel, spec)
        if key in self.items:
            return
        if rel in self.rel_bind:
            raise Untranslatable(f"a bound relation is read through its binding, not generated: {rel}")
        if rel.startswith("true."):
            raise Untranslatable(f"a fluent with no binding: {rel}")
        self.items[key] = None
        out = []
        for c in self.by_head.get(rel, ()):
            sub = {}
            ok = True
            for a, s in zip(c.head[1], spec):
                if s is None:
                    continue
                if GDL.is_var(a):
                    if a in sub and sub[a] != s:
                        ok = False
                    sub[a] = s
                elif a != s:
                    ok = False
            if not ok:
                continue
            head = tuple(_subst(a, sub) for i, a in enumerate(c.head[1]) if spec[i] is None)
            body = []
            for lit in c.body:
                if lit[0] == "distinct":
                    x, y = _subst(lit[1], sub), _subst(lit[2], sub)
                    if GDL.is_ground(x) and GDL.is_ground(y):
                        self.shapes["distinct decided here"] += 1
                        if x == y:
                            ok = False
                            break
                        continue
                    body.append(("distinct", x, y))
                    continue
                kind, r2, a2 = lit
                a2 = tuple(_subst(a, sub) for a in a2)
                if r2 in self.rel_bind:
                    body.append((kind, r2, None, a2))
                    continue
                s2 = self.spec_of_call(r2, a2)
                self.request(r2, s2)
                body.append((kind, r2, s2, tuple(a for i, a in enumerate(a2) if s2[i] is None)))
            if ok:
                out.append((head, body, c.text))
        self.items[key] = {"rel": rel, "spec": spec, "name": self.name(rel, spec), "clauses": out}
        self.order.append(key)

    # -------------------------------------------------------------------------------------------- what a relation reads
    def params_of(self) -> dict:
        """key -> the parameters its Lean definition takes (p: it reads the state; mv: the joint move; rec: the game record), through what it calls."""
        dep = {}
        for key in self.order:
            need = set()
            for _h, body, _t in self.items[key]["clauses"]:
                for lit in body:
                    if lit[0] == "distinct":
                        continue
                    if lit[1] in self.rel_bind:
                        need |= set(self.rel_bind[lit[1]][1])
                    else:
                        need |= dep.get((lit[1], lit[2]), set())
            dep[key] = need
        return dep

    # -------------------------------------------------------------------------------------------- emitting
    def emit(self, title: str, header: "list | None" = None) -> str:
        self.params = self.params_of()
        body_lines = []
        for key in self.order:
            body_lines += self._emit_rel(key)
        need_binding = self.uses_binding or any("mv" in v or "rec" in v for v in self.params.values())
        lines = ["import Chess.Basic"] + (["import Ideas.Spec.Binding"] if need_binding else []) + ["", "/-!", f"# {title}", ""]
        lines += header if header is not None else [
            "GENERATED by scripts/gdl_to_lean.py from rules/chess.gdl (ANALYST_PLAN 76.0 (c)) -- do not edit by hand; regenerate. Every relation is an inductive",
            "predicate, one constructor per GDL clause (the clause is the constructor's comment). A relation that reads the state takes the kernel's `Chess.Position`.",
            "The proofs about these definitions are in `Ideas.Spec.RookEquiv` (hand-written: rule-level proofs).",
        ]
        lines += ["-/", "", f"namespace {self.namespace}", ""]
        for sort in sorted(self.enums_used, key=lambda s: self.sort_bind[s]["enum"]):
            nm = self.sort_bind[sort]["enum"]
            consts = sorted(c for c in self.prog.sorts[sort] if isinstance(c, str))
            lines.append(f"/-- The GDL constants of one argument sort: {', '.join(consts)}. -/")
            lines.append(f"inductive {nm} where")
            for c in consts:
                lines.append(f"  | {camel(c)}")
            lines.append("  deriving DecidableEq, Repr")
            lines.append("")
        lines += body_lines
        return lines

    def finish(self, lines: list) -> str:
        return "\n".join(lines + [f"end {self.namespace}"]) + "\n"

    def _params(self, key) -> int:
        """The uniform prefix: positions where every clause's head has a variable (distinct ones) and every recursive premise passes the same variable."""
        it = self.items[key]
        arity = len(it["clauses"][0][0]) if it["clauses"] else 0
        rec = [any(lit[0] == "pos" and (lit[1], lit[2]) == key for lit in body) for _h, body, _t in it["clauses"]]
        if not any(rec):
            return 0
        k = 0
        for i in range(arity):
            good = True
            for (head, body, _t) in it["clauses"]:
                a = head[i]
                if not GDL.is_var(a) or a in head[:i] or a in head[i + 1:]:
                    good = False
                    break
                for lit in body:
                    if lit[0] == "pos" and (lit[1], lit[2]) == key and lit[3][i] != a:
                        good = False
            if not good:
                break
            k = i + 1
        return k

    def _pfx(self, key) -> str:
        need = self.params[key]
        return "".join(PARAMS[x] + " " for x in PARAM_ORDER if x in need)

    def _emit_rel(self, key) -> list:
        it = self.items[key]
        nm = it["name"]
        sorts = self.slot_sorts(it["rel"], it["spec"])
        npar = self._params(key)
        pfx = self._pfx(key)
        par_names = []
        if npar:
            head0 = it["clauses"][0][0]
            par_names = [lean_var(head0[i]) for i in range(npar)]
            for ci, (head, _b, _t) in enumerate(it["clauses"]):
                self.rename[(key, ci)] = {head[i]: par_names[i] for i in range(npar)}
        params = "".join(f"({par_names[i]} : {self.lean_type(sorts[i])}) " for i in range(npar))
        idx_types = " → ".join([self.lean_type(sorts[i]) for i in range(npar, len(sorts))] + ["Prop"])
        gdl_name = it["rel"] + "".join(f" {c}" for c in it["spec"] if c is not None)
        out = [f"/-- GDL `{gdl_name}`. -/", f"inductive {nm} {pfx}{params}: {idx_types} where"]
        for ci, (head, body, text) in enumerate(it["clauses"]):
            env = self.var_sorts(it["rel"], it["spec"], head, body)
            vs = []

            def collect(t):
                if GDL.is_var(t):
                    if t not in vs:
                        vs.append(t)
                elif isinstance(t, tuple):
                    for a in t[1:]:
                        collect(a)
            for a in list(head) + [x for lit in body for x in (lit[1:3] if lit[0] == "distinct" else lit[3])]:
                collect(a)
            bind_vars = [v for v in vs if not (npar and v in [head[i] for i in range(npar)])]
            binders = " ".join(f"{{{lean_var(v)} : {self.lean_type(env[v])}}}" for v in bind_vars)
            prem = [self._lit((key, ci), lit, env) for lit in body]
            concl = self._call(key, [self.render(a, s, (key, ci), env) for a, s in zip(head, sorts)])
            ty = " → ".join(prem + [concl])
            self.shapes["fact" if not body else "rule"] += 1
            if any(lit[0] == "pos" and (lit[1], lit[2]) == key for lit in body):
                self.shapes["recursive rule"] += 1
            if any(lit[0] == "neg" for lit in body):
                self.shapes["negation"] += 1
            if any(lit[0] == "distinct" for lit in body):
                self.shapes["distinct kept"] += 1
            if any(lit[0] != "distinct" and lit[1] in self.rel_bind for lit in body):
                self.shapes["fluent or input through the binding"] += 1
            if any(lit[0] != "distinct" and lit[1] not in self.rel_bind and lit[2] is not None and any(lit[2]) for lit in body):
                self.shapes["specialised call"] += 1
            if any(isinstance(a, tuple) for a in head) or any(lit[0] != "distinct" and any(isinstance(a, tuple) for a in lit[3]) for lit in body):
                self.shapes["move term"] += 1
            out.append(f"  /-- `{text}` -/")
            out.append(f"  | c{ci + 1} {binders + ' : ' if binders else ': '}{ty}")
        out.append("")
        return out

    def _call(self, key, args: list) -> str:
        it = self.items[key]
        p = [x for x in PARAM_ORDER if x in self.params[key]]
        return " ".join([it["name"]] + p + args) if (p or args) else it["name"]

    def _lit(self, ci_key, lit, env) -> str:
        if lit[0] == "distinct":
            x, y = lit[1], lit[2]
            s = env.get(x) if GDL.is_var(x) else env.get(y)
            return f"{self.render(x, s, ci_key, env)} ≠ {self.render(y, s, ci_key, env)}"
        kind, r2, s2, a2 = lit
        if r2 in self.rel_bind:
            tpl, _ps = self.rel_bind[r2]
            sig = self.prog.signature(r2)
            if any(x in tpl for x in ("Ideas.Spec.Binding.",)):
                self.uses_binding = True
            txt = tpl.format(*[self.render(a, s, ci_key, env) for a, s in zip(a2, sig)], p="p")
        else:
            txt = self._call((r2, s2), [self.render(a, s, ci_key, env) for a, s in zip(a2, self.slot_sorts(r2, s2))])
        if kind == "neg":
            return f"¬ ({txt})"
        return txt

    # -------------------------------------------------------------------------------------------- the next state as one predicate (76.1 (b) (4))
    def apply_move_spec(self) -> list:
        """`applyMoveSpec p mv q`: q is a position the spec's `next` facts describe -- for every fluent, its next relation holds of exactly the arguments the fluent's
        binding reads off q (the binding's template with q for the position)."""
        lines = ["/-- THE SPEC'S NEXT STATE (76.1 (b) (4), generated from the fluent table): `q` is a position the `next` facts after the joint move `mv` describe --",
                 "every fluent's next relation holds of exactly what the fluent's binding reads off `q`. Its functionality and its equality with the kernel's",
                 "`Chess.applyMove` are proved in `Ideas.Spec.NextEquiv`. -/",
                 "def applyMoveSpec (p : Chess.Position) (mv : Ideas.Spec.Binding.MoveTerm) (q : Chess.Position) : Prop :="]
        parts = []
        fluents = [f for f in self.prog.signatures if ("next." + f) in self.by_head]
        for f in fluents:
            key = ("next." + f, (None,) * len(self.prog.signature(f)))
            if key not in self.items:
                self.request(*key)
                self.params = self.params_of()
            sig = self.prog.signature(f)
            vs = [f"x{i}" for i in range(len(sig))]
            tpl, _ps = self.rel_bind["true." + f]
            call = self._call(key, vs)
            bound = " ".join(f"({v} : {self.lean_type(s)})" for v, s in zip(vs, sig))
            parts.append(f"  (∀ {bound}, {call} ↔ {tpl.format(*vs, p='q')})" if vs else f"  ({call} ↔ {tpl.format(p='q')})")
        lines.append(" ∧\n".join(parts))
        lines.append("")
        return lines


def _subst(a, sub):
    return sub.get(a, a) if isinstance(a, str) else a


def rook(out: Path) -> None:
    t = Translator(SPEC.read_text())
    for rel, spec in ROOK_ROOTS:
        t.request(rel, spec)
    text = t.finish(t.emit("THE ROOK, GENERATED FROM THE RULE SPEC: its attack, its move, its capture, and the relations they read"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    n_cl = sum(len(t.items[k]["clauses"]) for k in t.order)
    print(f"OBS gdl_to_lean rook: relations {len(t.order)} ({', '.join(t.items[k]['name'] for k in t.order)}), clauses {n_cl}, "
          f"generated sorts {sorted(t.sort_bind[s]['enum'] for s in t.enums_used)}; shapes " + ", ".join(f"{k} {v}" for k, v in sorted(t.shapes.items())) + f"; written {out}")


SPEC_HEADER = [
    "GENERATED by scripts/gdl_to_lean.py from rules/chess.gdl (ANALYST_PLAN 76.1 (b)) -- do not edit by hand; regenerate. EVERY relation of the spec is an",
    "inductive predicate, one constructor per GDL clause (the clause is the constructor's comment), its arguments typed by the spec's declared sorts (extension (1)).",
    "A relation that reads the state takes the kernel's `Chess.Position` `p`; one that reads the joint move takes `mv : Ideas.Spec.Binding.MoveTerm` (the side to",
    "move plays it, the other noop: `Ideas.Spec.Binding.Does`); one that reads the game record takes `rec` (extension (5): `Ideas.Spec.Binding.seen`). The fluents",
    "are read through the ontology shelf's bindings (extension (2)): `cell` the board, `control` the turn, `castling_right` the kernel's set through",
    "`Ideas.Spec.Binding.right`, `ep_square` the e.p. square, `halfmove` the clock saturated at 100 (`Ideas.Spec.Binding.clock`). `applyMoveSpec` (last) is the",
    "next state as one predicate; the proofs about it (functional; the kernel's `applyMove` on every pseudo-legal move) are in `Ideas.Spec.NextEquiv`.",
]


def spec(out: Path) -> None:
    t = Translator(SPEC.read_text(), namespace="Ideas.Spec.Chess")
    for r in sorted(t.by_head):
        t.request(r, (None,) * len(t.by_head[r][0].head[1]))
    lines = t.emit("THE RULES OF CHESS, GENERATED FROM THE RULE SPEC: every relation of rules/chess.gdl, and the next state as one predicate", SPEC_HEADER)
    lines += t.apply_move_spec()
    text = t.finish(lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    n_cl = sum(len(t.items[k]["clauses"]) for k in t.order)
    print(f"OBS gdl_to_lean spec: relations {len(t.order)} (of the spec's {len(t.by_head)} heads, with their specialisations), clauses {n_cl}, generated sorts "
          f"{sorted(t.sort_bind[s]['enum'] for s in t.enums_used)}; shapes " + ", ".join(f"{k} {v}" for k, v in sorted(t.shapes.items())) + f"; written {out}")


def survey(lean_out: "Path | None" = None) -> int:
    """Every relation of the spec translated on its own (unspecialised): OK, or the first reason it is not (the relation, or the dependency, that stops it).
    `lean_out`: every translatable relation written to one Lean file (to be checked with `lake env lean`). The inputs and the fluents (read through their
    bindings) are counted apart: they are bindings, not definitions."""
    text = SPEC.read_text()
    probe = Translator(text)
    rels = sorted(r for r in probe.by_head)
    bound = sorted(r for r in probe.prog.nodes if r in probe.rel_bind)
    unbound_inputs = sorted(r for r in probe.prog.nodes if r not in probe.by_head and r not in probe.rel_bind)
    ok, bad = [], defaultdict(list)
    for r in rels:
        t = Translator(text)
        try:
            t.request(r, (None,) * len(t.by_head[r][0].head[1]))
            t.emit("survey")
            ok.append(r)
        except Untranslatable as x:
            bad[str(x)].append(r)
    print(f"OBS gdl_to_lean survey: relations {len(rels)} (the spec's heads; of its {len(probe.prog.nodes)} relations, {len(bound)} are fluents and inputs read through "
          f"their bindings: {', '.join(bound)}{'; with no binding ' + ', '.join(unbound_inputs) if unbound_inputs else ''}), translated with their dependencies "
          f"{len(ok)}, not translated {len(rels) - len(ok)}")
    by_reason = defaultdict(list)
    for why, rs in bad.items():
        by_reason[why.split(":", 1)[0]].append((why.split(":", 1)[-1].strip(), rs))
    for reason, items in sorted(by_reason.items(), key=lambda x: -sum(len(rs) for _w, rs in x[1])):
        print(f"OBS gdl_to_lean not translated ({sum(len(rs) for _w, rs in items)}): {reason} -- " +
              "; ".join(f"{w} stops {', '.join(rs)}" for w, rs in sorted(items)))
    if lean_out is not None:
        t = Translator(text, namespace="Ideas.Spec.Survey")
        for r in ok:
            t.request(r, (None,) * len(t.by_head[r][0].head[1]))
        out = t.finish(t.emit("SURVEY: every relation of the spec the translator handles (not part of the library)"))
        lean_out.write_text(out)
        n_cl = sum(len(t.items[k]["clauses"]) for k in t.order)
        print(f"OBS gdl_to_lean survey file: relations {len(t.order)}, clauses {n_cl}; shapes " + ", ".join(f"{k} {v}" for k, v in sorted(t.shapes.items()))
              + f"; written {lean_out}")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "rook":
        rook(Path(a[a.index("--out") + 1]) if "--out" in a else OUT)
    elif a[0] == "spec":
        spec(Path(a[a.index("--out") + 1]) if "--out" in a else OUT_SPEC)
    elif a[0] == "survey":
        sys.exit(survey(Path(a[a.index("--lean") + 1]) if "--lean" in a else None))
