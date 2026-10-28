# The lemma store (plan 73)

One JSON file per DERIVED WORD: what the agenda may say beyond the base (movement and attack, tempo, material), where it came from, what it rests on,
whether it is proved. The fact store holds claims about positions, the policy holds order, Lean holds proofs; this directory holds the words. Loaded and
checked by `catspace/incubator/analyst/discovery/lemma_store.py`; the file is named by the entry's `name` (lower case, every run of characters other than
letters, digits and `_` one `-`).

| field | what it holds |
|---|---|
| `name` | the string the engine prints (a claim's lemma, a kill or GAP reason); a template writes each slot (a move, a square, a number, another word) as `...` |
| `kind` | CLOSING (closes a line or a question: a claim, a closer, a refutation) / DEFECT_FIX (why a goal fails, what mends it) / FORCING (why a move forces) / TECHNIQUE (a goal with a measure and a terminal theorem) / STOP (a GAP, a cut, a floor: never a kill) / TABLE (the five-men oracle) / GOAL (plan 75.2: a goal kind the agenda pursues toward the outcome -- its questions' word, its stake, its role) / VALUATION (plan 75.2: a count the stakes are in -- the unit worths) |
| `reads` | the base words it is built from, a non-empty subset of `movement`, `attack`, `tempo`, `material` |
| `terms` | the vocabulary s-expressions it compiles (may be empty); their own reads must be within `reads` |
| `status` | PROVED (every name in `lean` is a proved theorem of docs/LEAN_INDEX.md) / CONJECTURE (no theorem proves it as stated) / REFUTED / HEURISTIC (a stop or a tie-break: no proof obligation) |
| `lean` | the theorem names its implementation's docstrings cite, qualified to their full name in the index; non-empty if and only if PROVED |
| `origin` | BASE (the game's rule) / HAND (we wrote it) / MINED (a generator produced it) |
| `plan`, `journal` | the ANALYST_PLAN section(s) and the date |
| `measure` | TECHNIQUE only: `{quantity, floor, terminal}`, and for a technique the agenda ASKS (plan 73.5b) also `class` (the material signature, e.g. `KRvK`), `depth` (d: our moves before the measure must be lower), `n` (the floor's mate search) and `obligation` (the Lean statement that would prove it); `null` otherwise |
| `evidence` | `{uses: {}, confirmed: [], refuted: []}`: filled from runs later, never during one |
| `implementation` | where it runs, `path:function` from the repo root (validated: the file defines the name; a Lean `theorem` counts -- plan 73.6: a mined sub-lemma is implemented by its Lean file) |
| `note` | one plain sentence |
| `params` | optional (plan 75.2): what the agenda READS of the entry at run time. A GOAL has exactly `question` (the goal word its questions and claims carry), `basis` (RULE: bound to a kernel predicate, named in `rule`; VALUATION: a count of the `valuation` entry), `rule`, `valuation`, `stake` (a number; `{unit, less}`: the valuation's worth of the unit it places less the one it replaces; null: the question's own, its target's worth), `ends_game`, `keep` (don't lose: chased with keep semantics), `about` (the question's pattern key naming its location), `chase`, `reads_on_kill`, `near` (how near a unit must be for a root: null, every one). A VALUATION has exactly `worth` (unit kind -> a whole number), `unit` (the least gain), `royal` (the unit kind of no worth: its loss is the terminal). The store has exactly one GOAL that ends the game, one that places a unit, one gain, one keep, all counting in one valuation (`lemma_store.goal_kinds`). A STOP (plan 75.2 (b)) may carry its SHAPE here: `context` (root: the root's question budget; chase: the chase's stops; judge: the cost judge's cut) and its counts and goal lists (e.g. the budget's `per_unit`, the chase clock's `moves`, the walk of checks' `from_depth`, the follow list's depths), read at run time through `lemma_store.stops_for(context)`. A FORCING word (plan 75.2 (c)) may say `forced` (its replies are forced: the chase follows it below the root, the judge may let it decide; a printed route word that begins with a forced word is forced), `rank` / `rank_any` (the chase's route order when the word IS / begins with the entry's name) and `walked` (the plain check the walk of checks tests); a CLOSING word `reads_every_reply` (its claim is no fact of another position), `chase` (made by the chase), `leaf_read` (the d1 families read to their leaves), `window` (a gain within the window), `draw` (a repetition); plan 75.2 (d): a FORCING word `at_king` (its threat is on their royal unit: the king's zone -- its square and the squares next to it -- joins its route's frame), a CLOSING word `refutes` (a route killed by it is refuted: the kill proves a loss), `units` (the unit kinds of ours, beside the royal unit and the pawns, a measure holds for: `reasoned: the measure they hold`), `order` (a tie-break key's components, each `[name, sign]`, sign 1 fewer first, -1 more first: `the shape key`); a DEFECT_FIX `ahead` (the moves the side to move has in hand over the count its arithmetic compares: `no stopper in time`, the rule of the square's one move). No other kind has params. |

A printed string is CATALOGUED when it is an entry's name, or one of `lemma_store.PATTERNS` builds it from entry names (a prefix such as `after Qxd5: ` or
`inherited: `, a suffix, a composition such as `chase: step_after+step_least over X, Y`, a template), or it begins with an entry's name followed by its detail
(`: `, ` (`, `, `, `; `, ` -- `). The agenda passes every claim lemma, kill reason and GAP reason through `lemma_store.name(s)`, which returns `s` unchanged and
counts it; an uncatalogued string is counted (`ctx.uncatalogued`, a `GAP: uncatalogued word ...` line in the trace) and fails `validate`.

    python -m catspace.incubator.analyst.discovery.lemma_store report
    python -m catspace.incubator.analyst.discovery.lemma_store validate [<trace guard snapshot dir> ...]
    python -m catspace.incubator.analyst.discovery.lemma_store uses <trace file | question log .jsonl>
