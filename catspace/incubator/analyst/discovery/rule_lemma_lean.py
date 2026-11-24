"""PLAN 79.44 PART C: THE LEAN OBLIGATION OF A LEMMA (SYSTEM_SPEC 11 "One lemma machinery, used twice"; AGENTS.md rule 11: Lean is the source of truth; ledger rows 49, 33).

THIS TEXT IS THE INTERFACE, written before the body. Game-free: it reads a lemma through `rule_lemma.py` and writes Lean through the existing generator (scripts/gdl_to_lean.py, its
`Translator`, loaded by path: the script is no package); a game's Lean module, namespace and state type come from the lemma document's `lean` entry, never from this source.

WHAT IT WRITES. From a lemma, one Lean document under lean/ideas/Ideas/Mined/ (where the mined statements are put): the delta's relations as inductive predicates by the generator's own
conventions (one constructor a clause; the description's relations are the generated ones of the module the lemma names, opened, not written again), and the lemma's statement
over them:
  SAME     theorem <name>_same    (p : <state>) (hcond : <the condition>) (<places>) : L p <places> <-> M p <places>
  IMPLIES  theorem <name>_implies (p : <state>) (hcond : <the condition>) (<places>) : L p <places> -> M p <places>
the condition is the existence of a binding of its literals (a variable is bound existentially, a negated literal is a negation, a fluent is read through the generator's binding).
THE STANDING: the statement is written OPEN (`sorry`, the convention of the mined statements: lean_index lists it as OPEN) unless the document carries a proof (`lean.proof`, lines of
Lean, and `lean.uses`, the theorems of the library it rests on): the writer then puts those lines in place of `sorry` and Lean accepts the text only if they prove the statement. No proof is
written by this plan for a rule lemma; one is written for a lemma about play, by the library's theorem. The standing PROVED is held by `rule_lemma.standing_of` only while the text exists,
holds no `sorry`, states the theorem the document names, names each theorem of `lean.uses` (each in docs/LEAN_INDEX.md and not OPEN), and the named theorem is in the index and not OPEN.
INTERFACE (who calls it: the tests; `python -m catspace.incubator.analyst.discovery.rule_lemma_lean write <game> [<name>]`; the index is rebuilt afterwards with `lean_index build`):
  `obligation(lemma) -> str`                the Lean text
  `path_of(lemma) -> Path`                   lean/ideas/Ideas/Mined/rule_lemma_<game>_<name>.lean
  `write(lemma) -> Path`
Refusals (`Untranslatable`, the generator's own): a relation the description's generated module does not hold under the name the generator gives it (the generated module is older than
the description it was made from), a sort with no binding in the ontology shelf, a relation that reads the joint choice or the game record.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.incubator.analyst.discovery import rule_lemma as RL

ROOT = RL.ROOT
MINED = ROOT / "lean" / "ideas" / "Ideas" / "Mined"
LEAN_ROOT = ROOT / "lean" / "ideas"


def generator():
    """scripts/gdl_to_lean.py as a module (loaded by path: `scripts` is no package)."""
    name = "gdl_to_lean_for_lemmas"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "gdl_to_lean.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def ident(lemma: RL.Lemma) -> str:
    return RL.obligation_path(lemma).stem


def path_of(lemma: RL.Lemma) -> Path:
    return RL.obligation_path(lemma)


def module_path(module: str) -> Path:
    return LEAN_ROOT / (module.replace(".", "/") + ".lean")


class _Writer:
    """The generator's `Translator` over the composed description, asked for the delta's relations, the lemma's two relations and what the condition reads."""

    def __init__(self, lemma: RL.Lemma):
        G = generator()
        self.G = G
        self.lemma = lemma
        self.t = G.Translator(RL.composed_text(lemma), namespace=lemma.lean["namespace"])
        self.prog = self.t.prog
        base_heads = RL._heads(GDL.parse(RL.description_text(lemma)))
        self.mine = RL._heads([GDL.parse(c)[0] for c in lemma.delta])
        self.base_heads = base_heads
        self.cond_vars: list = []
        self.cond_text = self._condition()
        for rel in (lemma.left["relation"], lemma.right["relation"]):
            self.t.request(rel, (None,) * lemma.left["places"])
        for h in sorted(self.mine):
            if h in self.t.by_head:
                self.t.request(h, (None,) * len(self.prog.signature(h)))
        self.t.params = self.t.params_of()

    def _condition(self) -> str:
        t, prog = self.t, self.prog
        env: dict = {}
        parts = []
        for l in self.lemma.condition:
            if l["relation"] == RL.DIFFER:
                continue
            rel = RL.fluent_or_relation(prog, l["relation"])
            for a, s in zip(l["args"], prog.signature(rel)):
                if a.startswith("?"):
                    env.setdefault(a, s)
                    if a not in self.cond_vars:
                        self.cond_vars.append(a)
        for l in self.lemma.condition:
            if l["relation"] == RL.DIFFER:
                x, y = l["args"]
                s = env.get(x) if x.startswith("?") else env.get(y)
                parts.append(f"{t.render(x, s, None, env)} ≠ {t.render(y, s, None, env)}")
                continue
            rel = RL.fluent_or_relation(prog, l["relation"])
            sig = prog.signature(rel)
            args = tuple(a for a in l["args"])
            if rel in t.rel_bind:
                tpl, _ps = t.rel_bind[rel]
                txt = tpl.format(*[t.render(a, s, None, env) for a, s in zip(args, sig)], p="p")
            else:
                spec = t.spec_of_call(rel, args)
                t.request(rel, spec)
                t.params = t.params_of()
                key = (rel, spec)
                txt = t._call(key, [t.render(a, s, None, env) for a, s in zip(args, t.slot_sorts(rel, spec))])
            parts.append(f"¬ ({txt})" if l.get("neg") else txt)
        self.cond_env = env
        body = " ∧ ".join(parts) if parts else "True"
        if self.cond_vars:
            binders = " ".join(f"({self.G.lean_var(v)} : {t.lean_type(env[v])})" for v in self.cond_vars)
            return f"∃ {binders}, {body}"
        return body

    def refuse_missing(self) -> None:
        """Every description relation the delta calls is held by the generated module under the name the generator gives it."""
        t = self.t
        mod = module_path(self.lemma.lean["module"])
        text = mod.read_text() if mod.is_file() else ""
        for key in list(t.order):
            if key[0] not in self.mine:
                continue
            for _h, body, _txt in t.items[key]["clauses"]:
                for lit in body:
                    if lit[0] == "distinct" or lit[1] in t.rel_bind:
                        continue
                    callee = (lit[1], lit[2])
                    if lit[1] in self.mine:
                        continue
                    name = t.items[callee]["name"]
                    if not re.search(rf"^inductive {re.escape(name)}\b", text, re.M):
                        raise self.G.Untranslatable(f"{self.lemma.name}: the generated module {self.lemma.lean['module']} holds no relation {name} (for {lit[1]}): it is older than the description")

    def text(self) -> str:
        t, lemma = self.t, self.lemma
        self.refuse_missing()
        for key in t.order:
            need = t.params[key]
            if key[0] in self.mine and ("mv" in need or "rec" in need):
                raise self.G.Untranslatable(f"{lemma.name}: {key[0]} reads the joint choice or the game record")
        body = []
        for key in t.order:
            if key[0] in self.mine:
                body += t._emit_rel(key)
        sig = self.prog.signature(lemma.left["relation"])
        names = [f"a{i + 1}" for i in range(len(sig))]
        binders = " ".join(f"({n} : {t.lean_type(s)})" for n, s in zip(names, sig))
        lk, rk = (lemma.left["relation"], (None,) * len(sig)), (lemma.right["relation"], (None,) * len(sig))
        left, right = t._call(lk, names), t._call(rk, names)
        joint = "↔" if lemma.form == "SAME" else "→"
        thm = f"{lemma.name}_{lemma.form.lower()}"
        st = lemma.standing
        proof_lines = lemma.lean.get("proof") or []
        proof = "\n".join("  " + ln for ln in proof_lines) if proof_lines else "  sorry"
        need_binding = t.uses_binding
        imports = [lemma.lean["module"]] + [m for m in lemma.lean.get("imports", []) if m != lemma.lean["module"]] + (["Ideas.Spec.Binding"] if need_binding else [])
        lines = [f"import {m}" for m in imports] + ["",
                 f"/-! AUTO-GENERATED by catspace/incubator/analyst/discovery/rule_lemma_lean.py from the lemma `{lemma.name}` of the game `{lemma.game}` (rules/lemmas/{lemma.game}/{lemma.name}.json) -- do not edit by hand.",
                 f"PLAN 79.44 PART C: THE OBLIGATION of a lemma of origin {lemma.origin} and form {lemma.form}: under its condition, `{lemma.left['relation']}` {'has the same rows as' if lemma.form == 'SAME' else 'has its rows among those of'} `{lemma.right['relation']}`.",
                 "The delta's relations are generated as the rest of the description is (one constructor a clause; the clause is the comment); the description's own relations are those of "
                 f"`{lemma.lean['module']}`. The lemma, in words: {lemma.says}",
                 ("The statement is OPEN (`sorry`): no proof is written; lean_index lists it as OPEN."
                  if not proof_lines else "The proof is the document's own, by reasoning from the theorems it names (`lean.uses`): " + ", ".join(lemma.lean.get("uses", [])) + ". Lean accepts the text only if it is a proof of the statement."),
                 "-/", "", "set_option linter.unusedVariables false", "", f"namespace Ideas.Mined.{ident(lemma)}", "", f"open {lemma.lean['namespace']}", ""]
        lines += body
        lines += [f"/-- The lemma `{lemma.name}` ({lemma.form}, origin {lemma.origin}), generated from its document. -/",
                  f"theorem {thm} (p : {lemma.lean['position']}) (hcond : {self.cond_text}) {binders} :",
                  f"    {left} {joint} {right} := by", proof, "", f"end Ideas.Mined.{ident(lemma)}", ""]
        return "\n".join(lines)


def obligation(lemma: RL.Lemma) -> str:
    return _Writer(lemma).text()


def write(lemma: RL.Lemma) -> Path:
    p = path_of(lemma)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(obligation(lemma))
    return p


def main(argv: list) -> int:
    if len(argv) < 2 or argv[0] != "write":
        print(__doc__)
        return 2
    for lemma in RL.load_all(argv[1]):
        if len(argv) > 2 and lemma.name not in argv[2:]:
            continue
        print(f"OBS lemma obligation {lemma.game} {lemma.name} ({lemma.form}, origin {lemma.origin}, standing claimed {lemma.standing['is']}): written {write(lemma).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
