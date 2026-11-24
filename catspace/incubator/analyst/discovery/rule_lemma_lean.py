"""PLAN 79.44 PART C: THE LEAN OBLIGATION OF A LEMMA (SYSTEM_SPEC 11 "One lemma machinery, used twice"; AGENTS.md rule 11: Lean is the source of truth; ledger rows 49, 33).

THIS TEXT IS THE INTERFACE, written before the body. Game-free: it reads a lemma through `rule_lemma.py` and writes Lean through the existing generator (scripts/gdl_to_lean.py, its
`Translator`, loaded by path: the script is no package); a game's Lean module, namespace and state type come from the lemma document's `lean` entry, never from this source.

WHAT IT WRITES. From a lemma, one Lean file under lean/ideas/Ideas/Mined/ (where the mined statements go): the delta's relations as inductive predicates by the generator's own
conventions (one constructor a clause; the description's relations are the generated ones of the module the lemma names, opened, not written again), and the lemma's statement
over them:
  SAME     theorem <name>_same    (p : <state>) (hcond : <the condition>) (<places>) : L p <places> <-> M p <places>
  IMPLIES  theorem <name>_implies (p : <state>) (hcond : <the condition>) (<places>) : L p <places> -> M p <places>
the condition is the existence of a binding of its literals (a variable is bound existentially, a negated literal is a negation, a fluent is read through the generator's binding).
THE STANDING: the statement is written OPEN (`sorry`, the convention of the mined statements: lean_index lists it as OPEN) -- this plan proves no lemma by hand. A lemma whose document
says PROVED has its statement closed by `exact <theorem>`, the theorem named in the document; Lean then accepts it only if that theorem is the statement (up to the definitions): the
standing PROVED is held by `rule_lemma.standing_of` only while this file exists, holds no `sorry` and names that theorem, and the theorem is in docs/LEAN_INDEX.md and not OPEN.
INTERFACE (who calls it: the tests; `python -m catspace.incubator.analyst.discovery.rule_lemma_lean write <game> [<name>]`; the index is rebuilt after with `lean_index build`):
  `obligation(lemma) -> str`                the Lean text
  `path_of(lemma) -> Path`                   lean/ideas/Ideas/Mined/rule_lemma_<game>_<name>.lean
  `write(lemma) -> Path`
Refusals (`Untranslatable`, the generator's own): a relation the description's generated module does not hold under the name the generator gives it (the generated module is older than
the description it was made from), a sort with no binding in the ontology shelf, a relation that reads the joint move or the game record.
"""
