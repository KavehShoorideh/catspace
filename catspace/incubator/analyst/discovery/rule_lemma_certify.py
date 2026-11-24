"""PLAN 79.44 PART B: THE CERTIFICATE OF A LEMMA (SYSTEM_SPEC 11 "One lemma machinery, used twice"; pieces: rules, and the long-term memory's statements; ledger rows 49, 44).

THIS TEXT IS THE INTERFACE, written before the body. Game-free: the module reads a lemma through `rule_lemma.py` and the description through the reader (discovery/gdl.py); it names
no game and no relation of any description.

WHAT IT DOES. A lemma says that two relations of a description (with the lemma's delta composed on) have the same rows (SAME) or that one's rows are among the other's (IMPLIES),
under a condition. The certificate asks the reader's own evaluator (the oracle: the description is the source of the rules) at every state of a corpus: it evaluates both relations
and the condition, and where the condition holds it compares their rows. Nothing is weakened to make a lemma pass: a disagreement is kept with its state and its rows, and a lemma with
one is not certified.
THE CORPUS (the fixtures and the seeded games the kernel certificates use, by the description's own moves):
  the FIXTURES  rules/lemmas/<game>/states.txt, one state a line (the format is in rule_lemma.py);
  the GAMES     `GAMES` seeded games, seed `SEED` (the seed of discovery/certify.py), each played from the description's initial state by choosing, with a seeded generator, one
                legal move of each role at each turn (the description's `legal` and `next`), to the description's `terminal` or to the fuse `PLY_FUSE` (a named fuse: random play
                is not promised to end).
WHAT IS COUNTED AND PRINTED (OBS lines): states read; states where the condition held; rows of each relation on those states; rows compared (SAME: the union; IMPLIES: the left
rows); disagreements (states, and rows: SAME the rows of one and not the other, IMPLIES the left rows not on the right) with the first `EXAMPLES` states and rows. A lemma whose
condition never held, or whose left relation never had a row where it held, is VACUOUS and is not certified.
THE CERTIFICATE (rules/lemmas/<game>/<name>.certificate.json, the record of discovery/certify.py: binding, ingests, oracle, corpus, positions, disagreements, examples, key, sources,
date, seconds, and for a lemma the counts above): its KEY is a hash of what the answer depends on: the composed description text, the delta, the condition, the form and the two
relations, the reader's source (discovery/gdl.py), the fixtures, the corpus parameters, and the sources of this module and of rule_lemma.py. A change to any of them makes the
certificate STALE until `run` is repeated. The certificates of the old bindings (discovery/certify.py BINDINGS, their twelve) are not touched and not counted here.
INTERFACE (who calls it: `rule_lemma.standing_of` through `status`; the tests; `python -m catspace.incubator.analyst.discovery.rule_lemma_certify run <game> [<name>] [--jobs N]`):
  `key_of(lemma) -> str`                        the 16 hex digits of the key
  `corpus_states(lemma, games=GAMES, seed=SEED) -> iterator of (label, state)`   (for tests; `certify` reads it by chunks)
  `read_state(lemma, prog, state) -> dict`      {"held": bool, "left": set(rows), "right": set(rows)} at one state
  `certify(lemma, jobs=1, games=GAMES, fixtures=None) -> dict`   the certificate (not written); `fixtures`: a list of state texts that replaces the folder's (a test's own)
  `write(lemma, cert)`, `load(lemma) -> dict | None`, `path_of(lemma) -> Path`
  `status(lemma) -> "fresh" | "stale" | "failing" | "vacuous" | "missing"`   fresh: it exists, its key is the current one, it found no disagreement and was not vacuous
  `standing(lemma) -> (word, detail)`           `rule_lemma.standing_of` with the freshness read here
  `run(game, names=None, jobs=1, games=GAMES) -> int`   certify and write each lemma, print the OBS lines; the exit status 1 when one is not certified
"""
