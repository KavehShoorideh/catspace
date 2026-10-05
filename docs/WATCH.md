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

## Memory: a move at a time (Kaveh 2026-10-05)

The watcher keeps a memory of the game (frames.FrameMemory). A FRAME (a square and its occupant) persists across
plies while the occupant stays and the square stays contested, with its `age` (plies since it became a frame) and
`declined` (the times the attacker could win it by exchange and played something else). The EXCHANGE IN PROGRESS is
kept at the position: `exchange_on` (the last move captured and a recapture is available) and `settled` (the material
once the forecast exchanges settle: material plus the forecast's expected change), which the result rules read
instead of the material mid-exchange. INCREMENTAL: a move can change a frame's attackers, defenders, pins or
exchange only if it touches a line that reaches the square (up to the pieces standing on it, past sliders) or its
knight jumps, or such a line from a piece attacking or defending it, before or after the move; every other frame
keeps its heavy attributes (a test checks carried == recomputed on real games). OBS: about 11% of frames are carried
-- moves happen where the contested squares are.

## Rules

A rule has a kind (frame, elsewhere, result), a condition in a small JSON language over attributes, a parent (an
exception refines its parent: a Ripple-Down tree), its cornerstone positions and an English text; optionally a
reference attribute, so it forecasts an offset from it ("the exchange goes as counted" is one rule for every piece
value). Its forecast is the count of what happened each time its condition held, smoothed toward its parent's forecast
with weight ALPHA = 4 observations, so a new rule starts by saying what its parent says. A game's result counts as ONE
observation, shared by the positions that saw it.

When several rules apply (Kaveh, 2026-10-05): the more specific supersedes the less specific and the older. A young
rule is BLENDED with the rule it supersedes, trusted n/(n+20) by its own cases n (`rules.forecast`, policy
evidence_blend: a simple prior until a learned one; `rules.POLICIES` is the slot, and every applicable rule's
probability of what happened is logged as its training data). A new rule's counts start from the stored past cases
its condition covers (`watcher.backfill`; results once per game).

The rule base started EMPTY (only the three roots, the base rates). After each batch of 20 games Claude, in session,
reads the largest surprises (`show`), backtracks each (what was forecast, by which rule, what happened, which frame was
missing) and writes the rule or the exception, with the position as its cornerstone.

## Commands and files

```
python -m catspace.incubator.analyst.watch.watcher batch [--games 20]     # the next batch of the stream
python -m catspace.incubator.analyst.watch.watcher show [--batch N] [--top 20]
python -m catspace.incubator.analyst.watch.watcher add --kind frame --parent F0 --cond '{...}' --text "..." --cornerstone G:P:SQ [--ref '{"attr": "see_mover"}']
python -m catspace.incubator.analyst.watch.watcher replay                 # rebuild the state with today's code
python -m catspace.incubator.analyst.watch.watcher credit [--batch N]      # each rule's bits saved, vs the root and its parent
python -m catspace.incubator.analyst.watch.watcher health                  # per batch: bits saved per position vs the base rates
python -m catspace.incubator.analyst.watch.watcher export --batches 4 5    # the viewer's data (data/derived/watch/viewer_games.json)
```

THE VIEWER (`catspace/incubator/analyst/watch/viewer.html`, published as a private artifact with the exported games
beside it as games.json): pick a watched game and step through it; at every position the board shows the frames
(ringed by what their rule forecasts for the mover), each frame's rule, its forecast, its memory (age, captures
declined, a young rule's blend) and, once revealed, what happened and the surprise in bits; a surprise trace under
the board jumps to any move; "Think, then reveal" makes Next show the forecast before the outcome. It replays the
records the watcher wrote while watching (prequential); watching a new game live would need the local server.

Vocabulary today. Frame attributes (frames.attributes): `occ`, `occ_value`, `occ_mover` (the occupant is the
mover's), `attacker_to_move`, `n_att`, `n_def`, `min_att`, `min_def`, `att_lower`, `hanging`, `outnumbered`,
`see_att` (what the attackers win by exchange, each side free to stop), `see_mover` (that, in the mover's material),
`just_moved`, `recapture`, `pinned`, `in_check`, and from memory `age`, `declined`. Position attributes (watcher.position_attrs): `material`, `n_frames`,
`in_check`, `phase`, `elo_diff`, `mover_elo`, `ply`, `mate1_tomove`, `promo_tomove`, `promo_mover`, and for result
rules `settled` and `exchange_on`. A new word is a
plain function of the board with a test.

- `catspace/incubator/analyst/watch/` -- frames.py (frames, attributes, the observation), rules.py (the language, the
  tree, counts, selection), watcher.py (the loop).
- `artifacts/watch/rules.jsonl` -- the rules, append-only (committed). `artifacts/watch/batches.jsonl` -- one summary
  per batch (committed).
- `data/derived/watch/` -- the state (counts, the stream's position) and every batch's cases (git-ignored; `replay`
  rebuilds them).
- `tests/test_analyst_watch.py`.

## Judging (and Kaveh's decisions)

Asked and answered on 2026-10-05: forecasts are material balance (short) and result (long); the observation is the
watched game itself; rules attach to frames; more specific then newer supersedes; the rule base starts empty; reviews
every 20 games; a frame is counted at every ply it is seen (each ply is a new chance); a game's result counts once.

The health metric is the moving average of surprise per position, by horizon, over the batches. A rule's credit is the
bits it saved against the root and against its parent on the cases it decided (both probabilities are logged), not
the batch-to-batch change, which mixes games: one drawn game moves the long average by itself.
