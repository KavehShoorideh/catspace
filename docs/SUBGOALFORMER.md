# SubgoalFormer — transformer over subgoals, querying the geometry (Kaveh 2026-08-12)

Origin: the two-queens premove analysis. Kaveh's calculus, verbatim in structure:

> our subgoal is promotion. What can block promotion? his king. How many plies / what
> probability of success for the opponent subgoal that his king will reach in front of my
> pawn? never and zero, because my queen will block it and the king is too far anyway.
> also there's no way he can threaten us and cost us tempo (subgoal with me in check is
> unreachable). So we have calculated and we can modify the prob of promotion to 100%.
> so we can premove.
>
> this means we need a transformer over subgoals to select move, each time being able to
> query not with a matrix, but with the geometry.

## The insight, in our objects

Every step of that reasoning is a GEOMETRY QUERY the JQT stack exposes:

| natural-language step                        | query                                             |
|----------------------------------------------|---------------------------------------------------|
| "our subgoal is promotion"                   | candidate subgoal g, from selector / leverage     |
| "what can block it? his king"                | opponent counter-subgoal g' (blockade concept)    |
| "how many plies for his king to get there?"  | dA(s -> g' | opponent to act)  (CDA ruler)        |
| "what probability of success?"               | dB -> P(activate g' | resistance)  (CDB ruler)    |
| "never and zero (queen blocks, king far)"    | the race: dA(their g') > dA(our g) => P ~ 0       |
| "no check subgoal reachable, no tempo loss"  | P(activate check-concept) ~ 0                     |
| "so P(promotion) := 100%, premove"           | conditional promotion of the plan's probability    |

The plan's probability is not a static field readout -- it is REVISED by ruling out every
opponent counter-subgoal via the two rulers. That revision loop is what the transformer does
in one pass.

## Architecture

Tokens (per position, per side):
  * our candidate subgoals: top-k by pursuit value (leverage x reachability), from the
    selector / concept-leverage table
  * opponent counter-subgoals: the ARMED set (reply head dyn2) + leverage-harmful set
  * special tokens: current state summary phi(s); tempo/check concepts always included
    (the "can he cost us tempo" query is mandatory, per the premove calculus)

Token features = LIVE GEOMETRY QUERIES, recomputed each position (never a stored matrix):
  * dA(s -> anchor_g), dB-linked P(activate g)  -- BOTH sides' versions (z-conditioned two-
    evaluator principle: their reachability under our resistance, ours under theirs)
  * plies-to-activation vs the opponent's plies-to-counter (the race coordinates)
  * leverage swing (who the concept serves), base rate, armed flag
  * for the committed plan: premove/force_h from the forcing-search tree when available

Attention REPLACES the 512x512 interaction matrix: interference between subgoals is inferred
from their geometric coordinates in context (their blockade 6 plies away vs our promotion 3
= a race attention can read), so unseen subgoal pairings generalize -- the lift matrix only
knows pairs it has counted.

Heads:
  1. commitment: distribution over (pursue g | deny g') -- feeds the existing commitment
     protocol and move_for execution
  2. revised P(plan succeeds): the premove-confidence -- P(reach g) AFTER ruling out
     counter-subgoals; premove-safe iff ~1 and no counter-subgoal reachable
  3. delta-WDL per subgoal token (the subgoals-as-tokens plan, 2026-08-08), consistency-tied
     to the field

Training signals (all existing or streaming):
  * reach-events (24k+, growing) -- did the committed subgoal activate; calibrates head 2
  * game outcomes -- head 3
  * forcing-search premove labels -- head 2's premove-safety bit
  * selector RL rewards -- head 1 (warm-start from the current selector)

## Acceptance: the RACE BATTERY

Constructed promotion-race positions in three classes:
  a. unstoppable (king out of the square / path blocked): P(promote) ~ 1, premove-safe ON
  b. stoppable (king reaches the blockade in time): P drops, premove OFF, plan revised
  c. tempo-vulnerable (opponent has a check that wins the race by force): P drops via the
     check-subgoal token specifically -- the attention weight on the tempo token is the
     legibility check
Graded on calibration (P vs ground truth by TB/exhaustive search) and on ATTENTION
LEGIBILITY (does the model attend to the counter-subgoal that actually matters).

## Dependencies and order

1. reach_jqt2 (RUNNING): delivers the concept anchors + calibrated CDA/CDB rulers -- the
   query substrate. Gate suite first.
2. Refit concept stack on the jqt2 trunk (codes now persistence-shaped).
3. Race battery (buildable immediately after 1).
4. SubgoalFormer v1 (heads 1+2, ~4 layers), warm-started from the RL selector; head 3 after.
