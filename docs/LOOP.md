# The idea loop

Kaveh, 2026-10-05: "build the engineering loop first, and then we slot in different research pieces to see which one
works ... plus some level of human in the loop testing that decides whether the loop is going off the rails or the
loop is going fine." The plan was written by Fable and executed by Opus the same day.

The loop turns the player's mistakes into ideas, gives each idea a tier by proof or by evidence, wires it into the
player by what that tier permits, and measures what changed against frozen baselines. Every round ends in a one-page
ledger and a decision by Kaveh. The framework runs end to end with today's pieces in every slot; research pieces
replace stages and are judged by the same ledger.

## A round

```
python -m catspace.incubator.analyst.loop.round --round N           # needs the gate of round N-1
python -m catspace.incubator.analyst.loop.gate --round N --decision continue|rollback|swap [--stage NAME] --note "..."
python -m catspace.incubator.analyst.loop.dig --round N --set pawnendings  # before the gate: the puzzles the round moved
```

| Stage | Reads | Writes | Day-one piece | Research slot |
|---|---|---|---|---|
| frame | the player, the basis | `Framed`: where the player went wrong | dev puzzle misses (rows i % 97 == 5) and game positions where the frames said safe and the game took material | other framers |
| mine | framed positions | `Guard`: relation literals, claim, held-out evidence | `lemma_gen run` on the batch, `prove` re-test on held-out positions | R4 vocabulary from failed proofs |
| translate | guards, the registry | `Candidate`: hypotheses, footprint, bounded claim, sentence | Python candidates; the three Lean lemmas from `registry.py` | R2 guard-to-Lean translator |
| prove | candidates | `Promoted`: tier and evidence | Lean axioms check + bridges against exhaustive search; held-out evidence | R3 automatic proofs |
| use | promoted ideas | `UseRecord`s and `active.json` | closure (proved only), order (tie-break), sentence | R5 explainer, R6 budget redirect |
| measure | `active.json` | `Measurement`s and fire scans | the player on three frozen held-out sets, paired by position | — |

The hard rule (in `stages/use.py`): an idea may close search nodes only with tier unconditional or forcing, a Lean
theorem whose axioms are exactly `propext`, `Classical.choice`, `Quot.sound`, and a bridge that checked at least one
claim and found none wrong. A conjecture-tier idea may only break ties among moves the tactics value equally, and
speak in a sentence.

## The ledger and the gate

`artifacts/loop/round_NNN/LEDGER.md`: inputs, mined, promoted, fired, changed, controls, proposed next, decision.
"Changed" pairs the round's player by position against the frozen plain baseline and against the previous kept
round's arm (the player before this round's ideas), so a round's effect is attributed to its own ideas.
The four controls:

- **NOTHING FIRES** — no newly promoted idea has an instance at the solver's positions of the held-out sets.
- **RESULTS FALL** — a held-out set's expected score fell against the plain baseline or the previous kept round (for
  judgment, not a verdict). The expected score is the number of puzzles a player picking at random among its tied top
  moves solves. "Solution among the tied top moves" is not used: a tie-break only shrinks tied sets, so it can only
  fall when one acts (round 1's misreading, docs/BUGS.md 2026-10-05).
- **UNSOUND** (red) — an idea with a Lean statement whose axioms or bridge failed. A red round cannot be gated continue.
- **WALL TIME** — seconds per solver step against the baseline's.

Gate decisions: `continue` keeps the round's ideas; `rollback` retires them (the basis is append-only, so retire
lines are appended and nothing is deleted); `swap STAGE` makes the next round refuse to run until its config names a
different module for that stage (`--config` with `{"stages": {"mine": "catspace.incubator.analyst.loop.stages.mine_v2"}}`).

## Where things live

- `catspace/incubator/analyst/loop/` — contracts, basis, stages, ledger, gate, driver, wiring, registry, bridges.
- `data/derived/loop/round_NNN/` — every stage output, logs, `active.json` (git-ignored).
- `artifacts/loop/round_NNN/` — the ledger, inputs, promoted ideas, uses, measurements, controls, the gate (committed).
- `artifacts/loop/basis.jsonl` — the basis, hash-chained (committed; the plan put it under `data/derived`, which git
  ignores, so it moved to where a round's commit can carry it).
- `lean/ideas/scripts/check_axioms.sh` — the axioms check used by the prove stage.
- `tests/test_analyst_loop.py` — contracts, basis, use rules, a deliberately false idea, the driver's refusals, the ledger.

## Held-out sets (frozen in round 0)

| Set | Rows | Baseline dump (the plain player) |
|---|---|---|
| backrank | 300 back-rank puzzles, i % 97 == 17 | `frame_puzzles_lichess_zone_candidates_off17_backRankMate_lean1004.json` |
| pawnendings | 300 pawn-ending puzzles, i % 97 == 17 | `frame_puzzles_lichess_zone_candidates_off17_pawnEndgame_lean1005.json` |
| general | 300 puzzles, stride 997, offset 300 | `frame_puzzles_lichess_zone_candidates_off300_lean1004.json` |

Round 0 re-runs the three baselines once (they must reproduce the frozen dumps to the node) to time them.

## Proposals for Kaveh's call (2026-10-05)

### How the miner works today (rounds 0-1)

FRAME picks positions where the player went wrong: development puzzles it missed, and game moves its frames rated
safe (one-ply frame value >= 0) after which the game lost 2 pawns or more within 4 plies. MINE keeps only the
positions; the move played and the move missed are dropped. In those positions lemma_gen enumerates frames of one or
two pieces ("Q" is a queen of either colour, "bQ" a black one), fits each frame a claim ("moving this piece keeps /
loses material", measured by a two-ply episode inside the frame, relative to passing) and an exception guard (an OR
of ANDs of relation literals), and re-tests every lemma on 6,000 positions of Lichess games it was not fitted on.
PROVED there means the guard caught at least 95% of the exceptions and at least 80% of the guard's firings were real
exceptions (its "purity": the precision of the guard, not the rule's accuracy). So the rules learn the raw outcome,
not the player's error, and they restate what the frames already compute.

### Proposal 1: default rules, corrected by surprise (Kaveh's design)

Kaveh, 2026-10-05: "we want our player to be able to predict the next set of moves without search ... this pawn push
is going to result in losing the pawn, which is fine, as long as I know ... When I get surprised, that's the
problem." And: "you want to start with more generic rules ... if your queen goes to the middle of the board, you
lose. And then suddenly in one game you see the queen go to the middle of the board, but you don't lose ... surprise
... that knowledge is being corrected with every example that comes ... not ... a gradient ... an exception written
in Lean or in something else ... written up by the LLM investigating the position ... we just keep growing based on
what we see and the surprise."

- RULE: a default claim and a tree of exceptions; each exception may have its own. Every rule applies wherever it
  matches, in every game.
- FORECAST: what the rules predict for a move within k plies. A predicted loss is fine; it is the plan.
- SURPRISE: the forecast is contradicted along the game's continuation (or the engine's line): the rule said "the
  queen is lost" and it was not, or "safe" and the piece fell.
- EXCEPTION: the LLM looks at the surprising position, the rule and the continuation, and writes the exception in the
  rule language, so it is executable and checkable. The position that created it is kept as its CORNERSTONE. An
  exception is accepted only if it fires on its cornerstone, leaves the verdicts on earlier cornerstones unchanged,
  and on held-out games removes more surprises than it creates.
- GROWTH: by surprise, never by frame size; an exception may name any number of pieces.
- THE LOOP'S HEALTH METRIC: the surprise rate on held-out games, round over round. Puzzles stay a held-out control.
- GAMES: Maia-1500 and Maia-1800 against Stockfish (human-like mistakes, punished), and Lichess games. Maia is the
  body until the player can plan.

Already in the repo: lemma_gen's `--mode counter` ("believe until a counterexample, add what separates it") is this
loop with a search over the relation vocabulary in place of the LLM; the concept-bank loop (2026-09-20) had LLM
proposals checked by rules; lemma_apply applies guarded rules everywhere; the loop's frame, prove, use and measure
stages carry over, and MINE becomes "explain the surprise".

Prior art: Ripple-Down Rules (Compton and Jansen, from maintaining the GARVAN-ES1 expert system, late 1980s): a wrong
conclusion is patched by an exception added in the context of the case, which is stored as the rule's cornerstone
case. FOLD-R++ (Wang and Gupta, 2022) learns default rules with exceptions from data and tells exceptions from noise.

### Proposal 2: one rule language, executed in Rust, stated in Lean

- The rules and the LLM's exceptions are written in one small language over board relations (today's vocabulary:
  attacks, defends, attacked by a cheaper piece, outnumbered, gives check, stays on a line, and the after/then/some
  lifts; the LLM may add relations, each one a function with a test).
- Rust evaluates it: per position the relation tables are computed once on bitboards, then every rule's frame and
  exceptions are evaluated as boolean expressions over those tables, for all rules at once; Python orchestrates.
  rust/analyst_core already computes lemma_gen's outcomes and base literals.
- Lean states it: the same rule text translates to a Lean predicate, so an exception that carries weight can be
  proved (the R2 slot, the guard-to-Lean translator).
- Why now: "knowledge applied everywhere" means every rule at every position of every game. Today's Python Applier
  costs about 0.04 s per solver step for 100 rules (pawn endings, round 1: 0.06 s a step against 0.02 s); a grown
  rule base will hold thousands.

### Proposal 3: reason to a mate (strategy lemmas)

A forced mate can be proved once and then played without search: a STRATEGY (which move to play), an INVARIANT (what
stays true: the king cut off, the rooks safe) and a PROGRESS MEASURE that falls whatever the defender does (the ranks
left to the king). The proof is an induction on the measure, as the passer's was (`forcedWithinB`,
lean/ideas/Ideas/Passer.lean). In play the player checks the hypotheses and plays the strategy's move; "mate within
N" needs no search, and it predicts the whole continuation: no surprise. Two rooks against a king (the ladder): one
rook holds a rank, the other checks on the next, the king loses a rank every two plies; a rook the king attacks runs
to the far end of its rank, a tempo that keeps the cut.

Prior art: Huberman (1968), "A program to play chess end games", Stanford CS-106: "better" and "worse" predicates
from the chess books, and a proof that the program mates from any starting position in three endgames. Bratko (1978),
"Proving correctness of strategies in the AL1 assertional language", Information Processing Letters 7(5). Sadikov and
Bratko (2006), "Learning long-term chess strategies from databases", Machine Learning.

Plan, if chosen: prototype the ladder in Python, check it on every two-rooks-against-king position against the
tablebase, then prove it in Lean. It will rarely fire in real games; its value is the template (rook against king,
queen against king) and the first mate reached by reasoning rather than search.

### Suggested order

1. The rule language and its Rust evaluator (everything after it applies rules everywhere).
2. The surprise loop on Maia-against-Stockfish games, the LLM writing exceptions, cornerstones kept.
3. The two-rook strategy lemma in parallel, as the Lean showcase of reasoning to a mate.
