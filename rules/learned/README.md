# What the engine has learned about a game (plan 78.14)

One directory per game: `rules/learned/<game>/`.

| file | what it holds |
|---|---|
| `vocabulary.json` | the game's FACT VOCABULARY: its description (and, per variant, the delta that composes it), the derived relations that belong to the fact layer beside the state's own fluents, the relations a move's effect is read in (`delta`), how a state holds its units (`units`, for a game whose description declares them) |
| `lemmas/<name>.json` | one learned lemma per file: its body and its response in the lemma language, what verification measured (`stats`), where it came from (`meta`) |

Snapshots built from these are published under `data/derived/snapshots/<game>/<id>.json` (not committed: regenerate with `python -m catspace.incubator.analyst.discovery.game_store publish <game>`).
A snapshot is named by the first 16 hex digits of the SHA-256 of its bytes and the engine refuses a file that does not hash to its name.

## The lemma language

A lemma is "if these facts hold, a move of this kind is among the best". It is READ, never run: the engine binds its body to the position's facts and reads first the moves its response
names. It is a prior over the order of reading; it never says a move is best or a line is lost (AGENTS.md rule 7), so a wrong lemma costs reading time and nothing else.

```json
{"name": "...",
 "body": [{"relation": "attack", "args": ["us", "?a", "?b", "?c", "?d"], "neg": false}, ...],
 "response": {"removed": [{"relation": "cell", "args": ["?c", "?d", "them", "?k"], "neg": false}],
              "added":   [{"relation": "cell", "args": ["?c", "?d", "us", "?m"], "neg": false}]},
 "stats": {"status": "SAMPLED", "dev": {...}, "held": {...}, "depth": 5},
 "meta": {"origin": "sleep", "surprise": {...}, "steps": ["lift ...", "drop ..."], "teacher": false}}
```

* An argument starting with `?` is a variable, anything else a constant. `us` is the side to move and `them` the other side (with more than two roles `r1`, `r2` ...): every lemma is written
  in the mover's perspective, so one lemma serves both colours.
* The body is a conjunction of positive literals over the vocabulary, and negated ones whose variables the positive ones bind. A response variable the body does not bind is free.
* The response is the move's effect on the facts: the facts it removes and the facts it adds, over the `delta` relations.
* `stats.status` is `SAMPLED`: the claim "an optimal move of this kind exists" was checked EXACTLY (a search of stated depth, the plain minimax of the judge) on a sample of positions
  of games the lemma was not learned from, with no refutation; it is not proved. Lean stays the source of truth for what is proved (AGENTS.md rule 11).
* `meta.teacher` is true if a teacher's label went into the lemma; the snapshot then says `teacher_used`.

## How a lemma gets here

`python -m catspace.incubator.analyst.discovery.sleep <game> --log <wake log> --write` (see that module's docstring for the six steps): surprises are read from the wake's log, replayed
exactly, the saving moves become ground seeds, each seed is generalized by exact-checked lifts and drops on the development games' positions and then verified on the held-out games'
positions; only a lemma that passes every gate is filed here. The gates, the funnel and every rejection are reported.
