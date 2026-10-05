# Idea loop — round 000 (2026-10-05 13:03 PDT)

Code 4844bd3f (ledger rendered by b6d04ded+dirty); basis before the round 0 ideas (head 000000000000), after 3 ideas (head c4d3652e6e23). Stages: frame 0s, mine 0s, translate 0s, prove 23s, use 0s, measure 7s; total 0.5 min.
Stages ran under code frame 94eaffc1+dirty, mine 94eaffc1+dirty, translate 94eaffc1+dirty, prove 94eaffc1+dirty, use 94eaffc1+dirty, measure 94eaffc1+dirty.
Rendered from the stage outputs on disk (--render-only); the code of frame, mine, translate, prove, use, measure changed since they ran (or predates the source digest).

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
| lean:unstoppablePasser2 | 658/658, proved 0, one-away 0 | 851/851, proved 62, one-away 62 | 661/670, proved 16, one-away 9 |
| lean:unstoppablePasser | 658/658, proved 0, one-away 1 | 851/851, proved 0, one-away 221 | 661/670, proved 3, one-away 27 |
| lean:backRankMate | 656/658, proved 297, one-away 332 | 0/851, proved 0, one-away 0 | 335/670, proved 21, one-away 88 |
| mined:L-d7d626b1 | 658/658, speaks 658 | 851/851, speaks 851 | 670/670, speaks 670 |
| mined:L-16a354b6 | 649/658, speaks 622 | 652/851, speaks 449 | 626/670, speaks 565 |
| mined:L-ed25f52a | 623/658, speaks 565 | 500/851, speaks 321 | 584/670, speaks 528 |
| mined:L-1f5b02aa | 586/658, speaks 585 | 397/851, speaks 396 | 540/670, speaks 535 |
| mined:L-53e6d711 | 441/658, speaks 395 | 492/851, speaks 315 | 434/670, speaks 341 |
| mined:L-357b3596 | 325/658, speaks 324 | 388/851, speaks 386 | 431/670, speaks 423 |
| mined:L-b3306d88 | 228/658, speaks 228 | 1/851, speaks 1 | 334/670, speaks 331 |
| mined:L-29baffdc | 229/658, speaks 229 | 0/851, speaks 0 | 296/670, speaks 292 |
| mined:L-75ee9e46 | 181/658, speaks 166 | 0/851, speaks 0 | 278/670, speaks 264 |
| ... 91 more |  |  |  |

## Changed (the player with this round's wiring, paired by position: plain = the frozen baseline, previous = the previous kept round's arm)
| set | expected score: plain / previous / round | single top move: plain / previous / round | puzzles up / down vs previous | solution among tied (context): plain / previous / round | nodes vs plain on the same positions | wall / step |
|---|---|---|---|---|---|---|
| backrank | 300.0 / — / 300.0 | 289 / — / 289 | 0 / 0 (vs plain) | 300 / — / 300 of 300 | 1,063,544 → 836,385 (0.786; fewer 632, more 1) | not timed |
| pawnendings | 76.8 / — / 82.0 | 48 / — / 53 | 18 / 5 (vs plain) | 249 / — / 251 of 300 | 1,882,225 → 1,926,848 (1.024; fewer 44, more 175) | not timed |
| general | 238.3 / — / 239.8 | 227 / — / 228 | 5 / 1 (vs plain) | 279 / — / 280 of 300 | 20,427,992 → 20,149,612 (0.986; fewer 327, more 29) | not timed |

Expected score: the puzzles a player picking at random among its tied top moves solves. A tie-break can only remove moves from the tied set, so 'solution among tied' can only fall and 'single top move' only rise when one acts; the expected score moves either way and is what RESULTS FALL reads.

## Controls
- NOTHING FIRES: ok — newly promoted ideas that fire: 3 of 3
- RESULTS FALL: ok — no held-out set's expected score fell against the plain baseline or the previous round
- UNSOUND: ok — every Lean statement has standard axioms and a clean bridge
- WALL TIME: backrank: n/a a step against 0.21s; pawnendings: n/a a step against 0.02s; general: n/a a step against 1.12s

## Proposed next
- Continue: no control flagged.

## Decision
- pending: `python -m catspace.incubator.analyst.loop.gate --round 0 --decision continue|rollback|swap [--stage NAME] --note "..."`
