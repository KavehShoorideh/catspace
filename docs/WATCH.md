# The surprise watcher

Kaveh, 2026-10-05: "we want our player to be able to predict the next set of moves without search ... this pawn push
is going to result in losing the pawn, which is fine, as long as I know ... When I get surprised, that's the problem."
And: the tactical layer keeps hypotheses "over the language of Lean" that predict with simple rules and probabilities,
short-term with high certainty, long-term with low certainty; we are "not playing per se, but watching somebody play";
on a surprise we backtrack to the missing piece and add it, judged by a moving average of surprise.

## What it does

The watcher reads Lichess games already on disk (January 2019 records, `game_id % 97 == 7`, anybody against anybody).
After every move it forecasts, then the game reveals what happened:

| Horizon | Forecast | Revealed |
|---|---|---|
| short | the mover's material change once the exchanges settle (the next two moves quiet, at most 12 plies) | when that window closes |
| long | the game's result for the mover: loss, draw, win (games lost on time are censored) | when the game ends |

The short forecast is the sum of one forecast per FRAME (a square holding a piece the other side attacks, with its
attackers and defenders) and one for ELSEWHERE (captures on squares that were not frames), combined by convolution.
SURPRISE is minus log2 of the probability given to what happened, in bits.

## Rules

A rule has a kind (frame, elsewhere, result), a condition in a small JSON language over attributes, a parent (an
exception refines its parent: a Ripple-Down tree), its cornerstone positions and an English text; optionally a
reference attribute, so it forecasts an offset from it ("the exchange goes as counted" is one rule for every piece
value). Its forecast is the count of what happened each time its condition held, smoothed toward its parent's forecast
with weight ALPHA = 4 observations, so a new rule starts by saying what its parent says. A game's result counts as ONE
observation, shared by the positions that saw it.

When several rules apply (Kaveh, 2026-10-05): the more specific supersedes the less specific and the older. A learned
prior over rules comes later; `rules.POLICIES` is its slot, and every applicable rule's probability of what happened
is logged as its training data.

The rule base started EMPTY (only the three roots, the base rates). After each batch of 20 games Claude, in session,
reads the largest surprises (`show`), backtracks each (what was forecast, by which rule, what happened, which frame was
missing) and writes the rule or the exception, with the position as its cornerstone.

## Commands and files

```
python -m catspace.incubator.analyst.watch.watcher batch [--games 20]     # the next batch of the stream
python -m catspace.incubator.analyst.watch.watcher show [--batch N] [--top 20]
python -m catspace.incubator.analyst.watch.watcher add --kind frame --parent F0 --cond '{...}' --text "..." --cornerstone G:P:SQ [--ref '{"attr": "see_mover"}']
python -m catspace.incubator.analyst.watch.watcher replay                 # rebuild the state with today's code
```

- `catspace/incubator/analyst/watch/` -- frames.py (frames, attributes, the observation), rules.py (the language, the
  tree, counts, selection), watcher.py (the loop).
- `artifacts/watch/rules.jsonl` -- the rules, append-only (committed). `artifacts/watch/batches.jsonl` -- one summary
  per batch (committed).
- `data/derived/watch/` -- the state (counts, the stream's position) and every batch's cases (git-ignored; `replay`
  rebuilds them).
- `tests/test_analyst_watch.py`.

## Judging

The health metric is the moving average of surprise per position, by horizon, over the batches. A rule's credit is the
bits it saved against the root and against its parent on the cases it decided (both probabilities are logged), not
the batch-to-batch change, which mixes games: one drawn game moves the long average by itself.
