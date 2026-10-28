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
the constant substituted); (8) a state fluent `(true (f ...))` through the BINDING table (`cell` -> `p.board (f, r) = some ⟨c, k⟩`, `control` -> `p.turn = c`). The
SORTS of the arguments are inferred (union-find over argument slots and variables), each class bound by the constants it holds: files and ranks -> `Fin 8`, colours,
unit kinds -> the kernel's types; a class of unbound constants -> a generated enumeration (named by `SORT_NAMES`).

  .venv/bin/python scripts/gdl_to_lean.py rook [--out lean/ideas/Ideas/Spec/Rook.lean]      # the rook's clauses (its attack, its move, its capture and what they read)
  .venv/bin/python scripts/gdl_to_lean.py survey                                            # every relation of the spec: translated, or why not (OBS lines)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.io import paths

SPEC = paths.REPO_ROOT / "rules" / "chess.gdl"
OUT = paths.REPO_ROOT / "lean" / "ideas" / "Ideas" / "Spec" / "Rook.lean"
ROOK_ROOTS = [("attack_by", ("rook", None, None, None, None, None)), ("move_by", ("rook", None, None, None, None, None)),
              ("capture", ("rook", None, None, None, None, None))]

# THE BINDING (the ontology's Lean column for the spec's base vocabulary): constants by sort, the sorts' Lean types, the state fluents' readings
CONST = {
    "file": {f: str(i) for i, f in enumerate("abcdefgh")},
    "rank": {str(i + 1): str(i) for i in range(8)},
    "colour": {"white": "Chess.Color.white", "black": "Chess.Color.black"},
    "kind": {k: f"Chess.PieceKind.{k}" for k in ("king", "queen", "rook", "bishop", "knight", "pawn")},
}
SORT_TYPE = {"file": "Fin 8", "rank": "Fin 8", "colour": "Chess.Color", "kind": "Chess.PieceKind"}
FLUENTS = {
    "true.cell": (("file", "rank", "colour", "kind"), "p.board ({0}, {1}) = some ⟨{2}, {3}⟩"),
    "true.control": (("colour",), "p.turn = {0}"),
}
SORT_NAMES = {"north": "Dir", "kingside": "Side", "even": "Parity", "pawn": "Kind"}      # a generated enumeration's name, by a constant it holds
KEYWORDS = {"does", "legal", "goal", "terminal"}                                        # the game layer's relations (moves and outcomes: no Position predicate)


class Untranslatable(Exception):
    pass


RESERVED = {"p", "at", "by", "do", "then", "else", "if", "end", "let", "have", "show", "from", "fun", "match", "with", "in", "Type", "Prop", "open"}


def lean_var(v: str) -> str:
    """A GDL variable's Lean name (`?f1` -> `f1`; a name Lean or this file reserves gets a prime: `p` is the position)."""
    n = camel(v[1:])
    return n + "'" if n in RESERVED else n


def camel(s: str, upper: bool = False) -> str:
    parts = [p for p in s.replace(".", "_").split("_") if p]
    out = "".join(p[:1].upper() + p[1:] for p in parts)
    return out if upper else out[:1].lower() + out[1:]


class Translator:
    def __init__(self, text: str):
        self.clauses = GDL.clauses(GDL.parse(text))
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
        self.state_dep = {}
        self.items = {}                     # (rel, spec) -> dict(name, clauses (head args, body), ...)
        self.order = []
        self.shapes = defaultdict(int)
        self.rename = {}

    # -------------------------------------------------------------------------------------------- specialisation
    def spec_of_call(self, rel: str, args: tuple) -> tuple:
        return tuple(a if (i in self.indexed[rel] and isinstance(a, str) and not GDL.is_var(a)) else None for i, a in enumerate(args))

    def name(self, rel: str, spec: tuple) -> str:
        return camel(rel) + "".join(camel(c, upper=True) for c in spec if c is not None)

    def request(self, rel: str, spec: tuple) -> None:
        key = (rel, spec)
        if key in self.items:
            return
        if rel in KEYWORDS or rel.startswith(("next.", "init.")):
            raise Untranslatable(f"a game-layer relation (moves / outcomes / the next state), no Position predicate: {rel}")
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
                if any(isinstance(a, tuple) for a in a2):
                    raise Untranslatable(f"a compound term (a move term needs a Move binding): in {r2}")
                if r2 in FLUENTS:
                    body.append((kind, r2, None, a2))
                    continue
                s2 = self.spec_of_call(r2, a2)
                self.request(r2, s2)
                body.append((kind, r2, s2, tuple(a for i, a in enumerate(a2) if s2[i] is None)))
            if ok:
                out.append((head, body, c.text))
        self.items[key] = {"rel": rel, "spec": spec, "name": self.name(rel, spec), "clauses": out}
        self.order.append(key)

    # -------------------------------------------------------------------------------------------- sorts
    def infer_sorts(self) -> None:
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for key in self.order:
            it = self.items[key]
            for ci, (head, body, _t) in enumerate(it["clauses"]):
                def node(a):
                    return ("var", key, ci, a) if GDL.is_var(a) else ("const", a)
                for i, a in enumerate(head):
                    union(("slot", key, i), node(a))
                for lit in body:
                    if lit[0] == "distinct":
                        union(node(lit[1]), node(lit[2]))
                        continue
                    kind, r2, s2, a2 = lit
                    if r2 in FLUENTS:
                        sorts, _tpl = FLUENTS[r2]
                        for a, srt in zip(a2, sorts):
                            union(node(a), ("sort", srt))
                        continue
                    for j, a in enumerate(a2):
                        union(("slot", (r2, s2), j), node(a))
        for srt, consts in CONST.items():
            for c in consts:
                if ("const", c) in parent:
                    union(("const", c), ("sort", srt))
        classes = defaultdict(set)
        for x in list(parent):
            classes[find(x)].add(x)
        self.sort_of = {}
        self.enums = {}
        for root, members in classes.items():
            bound = {m[1] for m in members if m[0] == "sort"}
            if len(bound) > 1:
                raise Untranslatable(f"one argument class takes constants of the sorts: {sorted(bound)}")
            if bound:
                srt = bound.pop()
                lean_t, cmap = SORT_TYPE[srt], CONST[srt]
                for m in members:
                    if m[0] == "const" and m[1] not in cmap:
                        raise Untranslatable(f"a constant outside its sort's binding (GDL constants are untyped: the clock's numbers meet the ranks): {m[1]} in a {srt} argument")
            else:
                consts = sorted(m[1] for m in members if m[0] == "const")
                if not consts:
                    raise Untranslatable(f"an argument class with no constant and no binding: {sorted(map(str, members))[:3]}")
                if all(c.isdigit() for c in consts):              # numerals with no other binding: the naturals
                    srt, lean_t, cmap = "count", "Nat", {c: c for c in consts}
                    for m in members:
                        self.sort_of[m] = (srt, lean_t, cmap)
                    continue
                nm = next((SORT_NAMES[c] for c in consts if c in SORT_NAMES), camel(consts[0], upper=True) + "Sort")
                self.enums[nm] = consts
                lean_t, cmap = nm, {c: f"{nm}.{camel(c)}" for c in consts}
                srt = nm
            for m in members:
                self.sort_of[m] = (srt, lean_t, cmap)

    def term(self, key, ci, a) -> str:
        if GDL.is_var(a):
            return self.rename.get((key, ci), {}).get(a) or lean_var(a)
        return self.sort_of[("const", a)][2][a]

    def var_type(self, key, ci, a) -> str:
        return self.sort_of[("var", key, ci, a)][1]

    def slot_type(self, key, i) -> str:
        return self.sort_of[("slot", key, i)][1]

    # -------------------------------------------------------------------------------------------- emitting
    def emit(self, title: str) -> str:
        self.infer_sorts()
        # what reads the state
        dep = {}
        for key in self.order:
            it = self.items[key]
            dep[key] = any(lit[0] != "distinct" and (lit[1] in FLUENTS or dep.get((lit[1], lit[2]), False))
                           for _h, body, _t in it["clauses"] for lit in body)
        self.state_dep = dep
        lines = [
            "import Chess.Basic",
            "",
            "/-!",
            f"# {title}",
            "",
            "GENERATED by scripts/gdl_to_lean.py from rules/chess.gdl (ANALYST_PLAN 76.0 (c)) -- do not edit by hand; regenerate. Every relation is an inductive",
            "predicate, one constructor per GDL clause (the clause is the constructor's comment). A relation that reads the state takes the kernel's `Chess.Position`.",
            "The proofs about these definitions are in `Ideas.Spec.RookEquiv` (hand-written: rule-level proofs).",
            "-/",
            "",
            "namespace Ideas.Spec.Rook",
            "",
        ]
        for nm, consts in sorted(self.enums.items()):
            lines.append(f"/-- The GDL constants of one argument sort: {', '.join(consts)}. -/")
            lines.append(f"inductive {nm} where")
            for c in consts:
                lines.append(f"  | {camel(c)}")
            lines.append("  deriving DecidableEq, Repr")
            lines.append("")
        for key in self.order:
            lines += self._emit_rel(key)
        lines.append("end Ideas.Spec.Rook")
        return "\n".join(lines) + "\n"

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

    def _emit_rel(self, key) -> list:
        it = self.items[key]
        nm = it["name"]
        arity = len(it["clauses"][0][0]) if it["clauses"] else len([s for s in it["spec"] if s is None])
        npar = self._params(key)
        pfx = "(p : Chess.Position) " if self.state_dep[key] else ""
        par_names = []
        if npar:
            head0 = it["clauses"][0][0]
            par_names = [lean_var(head0[i]) for i in range(npar)]
            for ci, (head, _b, _t) in enumerate(it["clauses"]):
                self.rename[(key, ci)] = {head[i]: par_names[i] for i in range(npar)}
        params = "".join(f"({par_names[i]} : {self.slot_type(key, i)}) " for i in range(npar))
        idx_types = " → ".join([self.slot_type(key, i) for i in range(npar, arity)] + ["Prop"])
        gdl_name = it["rel"] + "".join(f" {c}" for c in it["spec"] if c is not None)
        out = [f"/-- GDL `{gdl_name}`. -/", f"inductive {nm} {pfx}{params}: {idx_types} where"]
        for ci, (head, body, text) in enumerate(it["clauses"]):
            vs = []
            for a in list(head) + [x for lit in body for x in (lit[1:3] if lit[0] == "distinct" else lit[3])]:
                if GDL.is_var(a) and a not in vs:
                    vs.append(a)
            bind_vars = [v for v in vs if not (npar and v in [head[i] for i in range(npar)])]
            binders = " ".join(f"{{{lean_var(v)} : {self.var_type(key, ci, v)}}}" for v in bind_vars)
            prem = []
            for lit in body:
                prem.append(self._lit(key, ci, lit))
            concl = self._call(key, [self.term(key, ci, a) for a in head])
            ty = " → ".join(prem + [concl])
            self.shapes["fact" if not body else "rule"] += 1
            if any(lit[0] == "pos" and (lit[1], lit[2]) == key for lit in body):
                self.shapes["recursive rule"] += 1
            if any(lit[0] == "neg" for lit in body):
                self.shapes["negation"] += 1
            if any(lit[0] == "distinct" for lit in body):
                self.shapes["distinct kept"] += 1
            if any(lit[0] != "distinct" and lit[1] in FLUENTS for lit in body):
                self.shapes["fluent through the binding"] += 1
            if any(lit[0] != "distinct" and lit[1] not in FLUENTS and any(lit[2]) for lit in body):
                self.shapes["specialised call"] += 1
            out.append(f"  /-- `{text}` -/")
            out.append(f"  | c{ci + 1} {binders + ' : ' if binders else ': '}{ty}")
        out.append("")
        return out

    def _call(self, key, args: list) -> str:
        it = self.items[key]
        p = ["p"] if self.state_dep[key] else []
        return " ".join([it["name"]] + p + args) if (p or args) else it["name"]

    def _lit(self, key, ci, lit) -> str:
        if lit[0] == "distinct":
            return f"{self.term(key, ci, lit[1])} ≠ {self.term(key, ci, lit[2])}"
        kind, r2, s2, a2 = lit
        if r2 in FLUENTS:
            txt = FLUENTS[r2][1].format(*[self.term(key, ci, a) for a in a2])
        else:
            txt = self._call((r2, s2), [self.term(key, ci, a) for a in a2])
        if kind == "neg":
            return f"¬ ({txt})"
        return txt


def _subst(a, sub):
    return sub.get(a, a) if isinstance(a, str) else a


def rook(out: Path) -> None:
    t = Translator(SPEC.read_text())
    for rel, spec in ROOK_ROOTS:
        t.request(rel, spec)
    text = t.emit("THE ROOK, GENERATED FROM THE RULE SPEC: its attack, its move, its capture, and the relations they read")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    n_cl = sum(len(t.items[k]["clauses"]) for k in t.order)
    print(f"OBS gdl_to_lean rook: relations {len(t.order)} ({', '.join(t.items[k]['name'] for k in t.order)}), clauses {n_cl}, "
          f"generated sorts {sorted(t.enums)}; shapes " + ", ".join(f"{k} {v}" for k, v in sorted(t.shapes.items())) + f"; written {out}")


def survey(lean_out: "Path | None" = None) -> int:
    """Every relation of the spec translated on its own (unspecialised): OK, or the first reason it is not (the relation, or the dependency, that stops it).
    `lean_out`: every translatable relation written to one Lean file (to be checked with `lake env lean`)."""
    text = SPEC.read_text()
    probe = Translator(text)
    rels = sorted(r for r in probe.by_head)
    ok, bad = [], defaultdict(list)
    for r in rels:
        t = Translator(text)
        try:
            t.request(r, (None,) * len(t.by_head[r][0].head[1]))
            t.emit("survey")
            ok.append(r)
        except Untranslatable as x:
            bad[str(x)].append(r)
    print(f"OBS gdl_to_lean survey: relations {len(rels)}, translated with their dependencies {len(ok)}, not translated {len(rels) - len(ok)}")
    by_reason = defaultdict(list)
    for why, rs in bad.items():
        by_reason[why.split(":", 1)[0]].append((why.split(":", 1)[-1].strip(), rs))
    for reason, items in sorted(by_reason.items(), key=lambda x: -sum(len(rs) for _w, rs in x[1])):
        print(f"OBS gdl_to_lean not translated ({sum(len(rs) for _w, rs in items)}): {reason} -- " +
              "; ".join(f"{w} stops {', '.join(rs)}" for w, rs in sorted(items)))
    if lean_out is not None:
        t = Translator(text)
        for r in ok:
            t.request(r, (None,) * len(t.by_head[r][0].head[1]))
        out = t.emit("SURVEY: every relation of the spec the translator handles (not part of the library)")
        lean_out.write_text(out.replace("namespace Ideas.Spec.Rook", "namespace Ideas.Spec.Survey").replace("end Ideas.Spec.Rook", "end Ideas.Spec.Survey"))
        n_cl = sum(len(t.items[k]["clauses"]) for k in t.order)
        print(f"OBS gdl_to_lean survey file: relations {len(t.order)}, clauses {n_cl}; shapes " + ", ".join(f"{k} {v}" for k, v in sorted(t.shapes.items()))
              + f"; written {lean_out}")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "rook":
        rook(Path(a[a.index("--out") + 1]) if "--out" in a else OUT)
    elif a[0] == "survey":
        sys.exit(survey(Path(a[a.index("--lean") + 1]) if "--lean" in a else None))
