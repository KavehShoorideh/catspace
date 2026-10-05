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
