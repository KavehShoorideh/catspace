# Idea loop — round 001 (2026-10-05 12:34 PDT)

Code 1a4ce48a (ledger rendered by 3cfc2f65); basis before the round 3 ideas (head cf90a07d1e40), after 103 ideas (head ea66b6a9affa). Stages: frame 632s, mine 79s, translate 0s, prove 27s, use 0s, measure 946s; total 28.1 min.
Rendered from the stage outputs on disk (--render-only); the code of frame, mine, translate, prove, use, measure changed since they ran (or predates the source digest).

## Inputs
- Framed: 6 puzzle misses on development rows (i % 97 == 5) and 150 game positions where the frames rated the move safe and the game took material within the next plies (development games). Ids frozen in inputs_framed.json.

## Mined
- 199 lemmas from lemma_gen on the framed positions; held-out re-test: PROVED 100, conjecture 99.
  - [* *] last_attacker(0,1) :: slot 0 acting loses (support 2137, held-out PROVED, purity 0.82)
  - [* P] last_attacker(0,1) :: slot 0 acting loses (support 2048, held-out PROVED, purity 0.83)
  - [* bP] last_attacker(0,1) :: slot 0 acting loses (support 1440, held-out conjecture, purity 0.86)
  - [* Q] last_attacker(1,0) :: slot 1 acting loses (support 1442, held-out conjecture, purity 0.83)
  - [* Q] defends(1,0) :: slot 1 acting keeps (support 1165, held-out PROVED, purity 0.95)

## Promoted
- **lean:backRankMate** — unconditional; Ideas.backRankMate_sound; axioms propext, Classical.choice, Quot.sound; bridge 624 claims checked, 0 wrong.
- **lean:unstoppablePasser** — unconditional; Ideas.Passer.unstoppablePasser_sound'; axioms propext, Classical.choice, Quot.sound; bridge 972 claims checked, 0 wrong.
- **lean:unstoppablePasser2** — unconditional; Ideas.Passer2.unstoppablePasser2_sound; axioms propext, Classical.choice, Quot.sound; bridge 1351 claims checked, 0 wrong.
- Conjecture tier: 100 ideas (0 unsound). Allowed uses: order, sentence.
  - mined:L-d7d626b1: [*] moving the * keeps material, unless (lands_attacked() & not attacked(0))
  - mined:L-1f5b02aa: [* *] moving the * loses material, unless not lands_attacked() or attacked(0) or (worth_ge(1,0) & not after_some_captures_worth_ge(0,1))
  - mined:L-b3306d88: [* Q] moving the Q keeps material, unless (lands_attacked() & not attacked(1)) or (not attacked_by_lower(1) & lands_attacked())
- Not promoted: 99.

## Fired (static scan at the solver's positions of the held-out sets)
| idea | backrank | pawnendings | general |
|---|---|---|---|
| lean:unstoppablePasser | 658/658, proved 0, one-away 1 | 851/851, proved 0, one-away 221 | 661/670, proved 3, one-away 27 |
| lean:unstoppablePasser2 | 658/658, proved 0, one-away 0 | 851/851, proved 62, one-away 62 | 661/670, proved 16, one-away 9 |
| lean:backRankMate | 656/658, proved 297, one-away 332 | 0/851, proved 0, one-away 0 | 335/670, proved 21, one-away 88 |
| mined:L-d7d626b1 | 658/658, speaks 658 | 851/851, speaks 851 | 670/670, speaks 670 |
| mined:L-16a354b6 | 649/658, speaks 622 | 652/851, speaks 449 | 626/670, speaks 565 |
| mined:L-ed25f52a | 623/658, speaks 565 | 500/851, speaks 321 | 584/670, speaks 528 |
| mined:L-1f5b02aa | 586/658, speaks 585 | 397/851, speaks 396 | 540/670, speaks 535 |
| mined:L-53e6d711 | 441/658, speaks 395 | 492/851, speaks 315 | 434/670, speaks 341 |
| mined:L-357b3596 | 325/658, speaks 324 | 388/851, speaks 386 | 431/670, speaks 423 |
| mined:L-b3306d88 | 228/658, speaks 228 | 1/851, speaks 1 | 334/670, speaks 331 |
| mined:L-29baffdc | 229/658, speaks 229 | 0/851, speaks 0 | 296/670, speaks 292 |
| mined:L-73527c48 | 181/658, speaks 166 | 0/851, speaks 0 | 278/670, speaks 264 |
| ... 91 more |  |  |  |

## Changed (the player with this round's wiring, paired by position: plain = the frozen baseline, previous = the previous kept round's arm)
| set | expected score: plain / previous / round | single top move: plain / previous / round | puzzles up / down vs previous | solution among tied (context): plain / previous / round | nodes vs plain on the same positions | wall / step |
|---|---|---|---|---|---|---|
| backrank | 300.0 / 300.0 / 300.0 | 289 / 289 / 294 | 0 / 0 | 300 / 300 / 300 of 300 | 1,063,544 → 836,385 (0.786; fewer 632, more 1) | 0.20s |
| pawnendings | 76.8 / 82.0 / 92.9 | 48 / 53 / 72 | 60 / 82 | 249 / 251 / 169 of 300 | 1,601,503 → 1,639,414 (1.024; fewer 43, more 161) | 0.06s |
| general | 238.3 / 239.8 / 241.0 | 227 / 228 / 237 | 14 / 30 | 279 / 280 / 250 of 300 | 19,210,971 → 18,933,498 (0.986; fewer 310, more 20) | 1.23s |

Expected score: the puzzles a player picking at random among its tied top moves solves. A tie-break can only remove moves from the tied set, so 'solution among tied' can only fall and 'single top move' only rise when one acts; the expected score moves either way and is what RESULTS FALL reads.

## Controls
- NOTHING FIRES: ok — newly promoted ideas that fire: 100 of 100
- RESULTS FALL: ok — no held-out set's expected score fell against the plain baseline or the previous round
- UNSOUND: ok — every Lean statement has standard axioms and a clean bridge
- WALL TIME: backrank: 0.20s a step against 0.21s (0.94x); pawnendings: 0.06s a step against 0.02s (3.02x); general: 1.23s a step against 1.12s (1.10x)

## Proposed next
- All new ideas are conjecture tier: slot R2 (the guard-to-Lean translator) and R3 (automatic proofs) decide whether any can close search.
- More puzzles went down than up against the previous round on pawnendings, general, though the expected score did not fall: the gain sits in fewer puzzles; dig the down ids (MISS anatomy) before relying on the round's uses.
- Time per step above 1.5x the baseline on pawnendings: profile the round's uses before keeping them.

## Decision
- pending: `python -m catspace.incubator.analyst.loop.gate --round 1 --decision continue|rollback|swap [--stage NAME] --note "..."`
