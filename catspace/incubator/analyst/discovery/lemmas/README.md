# The lemma store (plan 73)

One JSON file per DERIVED WORD: what the agenda may say beyond the base (movement and attack, tempo, material), where it came from, what it rests on,
whether it is proved. The fact store holds claims about positions, the policy holds order, Lean holds proofs; this directory holds the words. Loaded and
checked by `catspace/incubator/analyst/discovery/lemma_store.py`; the file is named by the entry's `name` (lower case, every run of characters other than
letters, digits and `_` one `-`).

| field | what it holds |
|---|---|
| `name` | the string the engine prints (a claim's lemma, a kill or GAP reason); a template writes each slot (a move, a square, a number, another word) as `...` |
| `kind` | CLOSING (closes a line or a question: a claim, a closer, a refutation) / DEFECT_FIX (why a goal fails, what mends it) / FORCING (why a move forces) / TECHNIQUE (a goal with a measure and a terminal theorem) / STOP (a GAP, a cut, a floor: never a kill) / TABLE (the five-men oracle) |
| `reads` | the base words it is built from, a non-empty subset of `movement`, `attack`, `tempo`, `material` |
| `terms` | the vocabulary s-expressions it compiles (may be empty); their own reads must be within `reads` |
| `status` | PROVED (every name in `lean` is a proved theorem of docs/LEAN_INDEX.md) / CONJECTURE (no theorem proves it as stated) / REFUTED / HEURISTIC (a stop or a tie-break: no proof obligation) |
| `lean` | the theorem names its implementation's docstrings cite, qualified to their full name in the index; non-empty if and only if PROVED |
| `origin` | BASE (the game's rule) / HAND (we wrote it) / MINED (a generator produced it) |
| `plan`, `journal` | the ANALYST_PLAN section(s) and the date |
| `measure` | TECHNIQUE only: `{quantity, floor, terminal}`, and for a technique the agenda ASKS (plan 73.5b) also `class` (the material signature, e.g. `KRvK`), `depth` (d: our moves before the measure must be lower), `n` (the floor's mate search) and `obligation` (the Lean statement that would prove it); `null` otherwise |
| `evidence` | `{uses: {}, confirmed: [], refuted: []}`: filled from runs later, never during one |
| `implementation` | where it runs, `path:function` from the repo root (validated: the file defines the name) |
| `note` | one plain sentence |

A printed string is CATALOGUED when it is an entry's name, or one of `lemma_store.PATTERNS` builds it from entry names (a prefix such as `after Qxd5: ` or
`inherited: `, a suffix, a composition such as `chase: step_after+step_least over X, Y`, a template), or it begins with an entry's name followed by its detail
(`: `, ` (`, `, `, `; `, ` -- `). The agenda passes every claim lemma, kill reason and GAP reason through `lemma_store.name(s)`, which returns `s` unchanged and
counts it; an uncatalogued string is counted (`ctx.uncatalogued`, a `GAP: uncatalogued word ...` line in the trace) and fails `validate`.

    python -m catspace.incubator.analyst.discovery.lemma_store report
    python -m catspace.incubator.analyst.discovery.lemma_store validate [<trace guard snapshot dir> ...]
    python -m catspace.incubator.analyst.discovery.lemma_store uses <trace file | question log .jsonl>
