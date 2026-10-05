# Idea loop — round 000 (2026-10-05 11:16 PDT)

Code 4844bd3f; basis before the round 0 ideas (head 000000000000), after 3 ideas (head cf90a07d1e40). Stages: frame 0s, mine 0s, translate 0s, prove 23s, use 0s, measure 902s; total 15.4 min.

## Inputs
- No framing this round (baseline round).

## Mined
- Nothing mined (baseline round).

## Promoted
- **lean:backRankMate** — unconditional; Ideas.backRankMate_sound; axioms propext, Classical.choice, Quot.sound; bridge 624 claims checked, 0 wrong.
- **lean:unstoppablePasser** — unconditional; Ideas.Passer.unstoppablePasser_sound'; axioms propext, Classical.choice, Quot.sound; bridge 972 claims checked, 0 wrong.
- **lean:unstoppablePasser2** — unconditional; Ideas.Passer2.unstoppablePasser2_sound; axioms propext, Classical.choice, Quot.sound; bridge 1351 claims checked, 0 wrong.
- Not promoted: 0.

## Fired (static scan at the solver's positions of the held-out sets)
| idea | backrank | pawnendings | general |
|---|---|---|---|
| lean:unstoppablePasser | 658/658, proved 0, one-away 1 | 851/851, proved 0, one-away 221 | 661/670, proved 3, one-away 27 |
| lean:unstoppablePasser2 | 658/658, proved 0, one-away 0 | 851/851, proved 62, one-away 62 | 661/670, proved 16, one-away 9 |
| lean:backRankMate | 656/658, proved 297, one-away 332 | 0/851, proved 0, one-away 0 | 335/670, proved 21, one-away 88 |

## Changed (the player with this round's wiring, paired by position against the frozen plain baselines)
| set | baseline tied / single | round tied / single | gained | lost | nodes on the same positions | wall / step |
|---|---|---|---|---|---|---|
| backrank | 300/300 / 289 | 300/300 / 289 | 0 | 0 | 1,063,544 → 836,385 (0.786; fewer 632, more 1) | not timed |
| pawnendings | 249/300 / 48 | 251/300 / 53 | 2 | 0 | 1,882,225 → 1,926,848 (1.024; fewer 44, more 175) | not timed |
| general | 279/300 / 227 | 279/300 / 227 | 0 | 0 | 20,427,992 → 20,152,828 (0.987; fewer 318, more 3) | not timed |

## Controls
- NOTHING FIRES: ok — newly promoted ideas that fire: 3 of 3
- RESULTS FALL: ok — no held-out set lost more puzzles than it gained
- UNSOUND: ok — every Lean statement has standard axioms and a clean bridge
- WALL TIME: backrank: n/a a step against 0.21s; pawnendings: n/a a step against 0.02s; general: n/a a step against 1.12s

## Proposed next
- Continue: no control flagged.

## Decision
- pending: `python -m catspace.incubator.analyst.loop.gate --round 0 --decision continue|rollback|swap [--stage NAME] --note "..."`
