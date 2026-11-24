"""PLAN 79.44 PART A: THE LEMMA FORM AND ITS READER (SYSTEM_SPEC 11 "One lemma machinery, used twice"; pieces: rules, and the long-term memory's statements; ledger rows 49, 44).

THIS TEXT IS THE INTERFACE, written before any body (docs/AGENT_BRIEF.md section 0). The module is one piece of the machinery that holds, certifies and states a lemma. It
names no game and no relation of any description: a game, its relations and its constants occur only in the lemma documents themselves, in their fixtures, and in the
descriptions under rules/. Its three neighbours (the certificate: `rule_lemma_certify.py`; the proof obligation: `rule_lemma_lean.py`; the statement: `statement_of` here) read
a lemma only through the functions below.

WHAT A LEMMA IS. One form, for a lemma about the rules (a cheaper condition that gives the same rows as a relation of the description) and for a lemma about play (a
relation that is contained in another). Two forms of claim:
  SAME     "under the condition, relation L has the same rows as relation M"
  IMPLIES  "under the condition, every row of L is a row of M"
L and M are named by relation and number of places; each exists in the description composed with the lemma's delta, with that many places (its declared signature).
The CONDITION is a list of literals in the form the learned lemmas already use ({"relation", "args", "neg"}, rust/engine/rep/src/lemma.rs; an argument that starts with
`?` is a variable): a conjunction read on the state; empty means always. A relation named in a literal that the description declares as a state fluent is read as the fluent.
The DELTA is a list of clauses in the description's own language (a clause or a signature declaration each), ADDITIVE ONLY: it defines new relations (and their helpers),
retracts nothing, redefines no relation the description defines, may read any relation of the description, and with the description is still stratified, safe and well sorted
by the reader's own check (discovery/gdl.py). It defines at least one of L and M. The ORIGIN is HAND (knowledge given, printed wherever the lemma is) or MINED (a later row).
The STANDING a document CLAIMS is CONJECTURE, CERTIFIED (with the certificate's key) or PROVED (with the name of a Lean theorem); the standing a lemma HAS is computed
(`standing_of`): a CERTIFIED claim holds only while the certificate beside it is fresh, a PROVED claim only while the theorem is in docs/LEAN_INDEX.md and not OPEN; otherwise it has
CONJECTURE.

WHERE LEMMAS LIVE (the folder): `rules/lemmas/<game>/`, beside the descriptions (rules/<description>.gdl, rules/variants/, rules/concepts/); not rules/learned, not data/derived.
  rules/lemmas/<game>/<name>.json              the lemma (one document)
  rules/lemmas/<game>/<name>.certificate.json  its certificate, written by `rule_lemma_certify.run`
  rules/lemmas/<game>/states.txt               the fixtures: one state a line, as the fluents of the state without `true` -- `(f a b) (g c) ...`; a line that starts with `;` is a note
  lean/ideas/Ideas/Mined/<...>.lean            the Lean obligation, written by `rule_lemma_lean.write` (an OPEN statement unless PROVED)

THE DOCUMENT (JSON, keys in this order; `delta` one clause a string):
  {"name": "<slug>", "game": "<game>", "variant": "<variant of the game's vocabulary>", "form": "SAME" | "IMPLIES", "says": "<the lemma in plain words>",
   "left": {"relation": "<L>", "places": n}, "right": {"relation": "<M>", "places": n}, "condition": [{"relation": "r", "args": ["?x", "c"], "neg": false}, ...],
   "delta": ["(signature ...)", "(<= (head ...) ...)", ...], "origin": "HAND", "standing": {"kind": "CONJECTURE"} | {"kind": "CERTIFIED", "key": "<16 hex>"} | {"kind": "PROVED", "theorem": "<Lean name>"},
   "lean": {"module": "<Lean module of the generated description>", "namespace": "<its namespace>", "position": "<the Lean type of a state>"}}

WHAT THE MODULE TAKES AND RETURNS (who calls it: the certificate module and the Lean writer here; later the translator, which reads a SAME lemma as a rewrite of L by M, and the
layers above, which read the statements):
  `lemma_dir(game) -> Path`                 the folder above (game-free: a game is a name)
  `load(game, name) -> Lemma`               one document, read and checked; raises `LemmaError` listing every problem
  `load_all(game) -> [Lemma]`               every document of the folder, by name
  `check(doc: dict, base_text: str) -> [str]`   the problems of a document read against a description text (empty: well formed); the checks are named in the body
  `description_text(lemma) -> str`          the description the lemma is about (the game's composition of its base, variant and concept deltas: game_store.spec_texts)
  `composed_text(lemma) -> str`             the description with the delta composed on (cs_gdl compose, which the variants use)
  `program(lemma) -> gdl.Program`           the composed text read by the reader; its evaluator is the oracle of the certificate
  `condition_rows(lemma, db) -> [binding]`  the bindings of the condition on an evaluated state (one empty binding when the condition is empty)
  `standing_of(lemma, fresh: bool) -> (word, detail)`  the standing the lemma HAS, given whether its certificate is fresh (the certificate module tells)
  `statement_of(lemma, fresh: bool) -> dict`  PART D: the lemma as the JSON of a `Statement` (rust/engine/rep/src/statement.rs): id the name, pattern the condition, says Holds, holder
                                            ours, `also` the keys the statement stands on (its form, its two relations, the key of the text it was certified against), standing
                                            Proved(theorem) for PROVED and Conjecture(<the sample or the certificate>) for the others. No reader of it exists in the wake yet.
Failure modes: a document with a problem is refused whole (`LemmaError`), never half read; a standing that its evidence does not hold is `CONJECTURE`, printed with the cause.
"""
