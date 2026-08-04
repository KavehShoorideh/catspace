> **A repository restructure occurred on 2026-08-03.** File and directory paths
> mentioned below are historical and no longer valid. The content is left exactly
> as written — this is the research record, not documentation of the current tree.
> For the current layout see [`repo_structure.md`](../../../repo_structure.md); for what
> moved and why see [the refactor plan](../../../catspace/research/docs/2026-08-03-refactor-plan.md).

# Phase 2 report: control field + ascent cone validation gates

**Date**: 2026-08-02. **What was run**: `experiments/controlfield_gates.py`
(full scale: gate1 n=2000, gate2 n=200, gate3 all 3 hand-picked gambits),
27s wall-clock. Smoke run at n=100/30 confirmed representative before scaling
(rho +0.112 at n=100 vs +0.144 at n=2000 -- consistent, not a fluke either way).

**All three gates failed.** Per the spec ("if gate 3 fails, stop and surface
it; the failure is more informative than a workaround"), stopping here --
Phase 3 (training a GAB transformer on these labels) is NOT started.

## Gate 1 (sanity): FAIL, but marginally and with a clean signal

`Spearman rho +0.144 (p=1.01e-10), n=2000, gate >=0.15`

The correlation between `cone_size` and a shallow (depth-6) Stockfish eval
favoring the mover is real and highly significant (p=1e-10) -- this is not
noise -- but the effect size falls just short of the 0.15 bar. The metric
captures *some* real signal about who's better, just not strongly. Did not
retune tau/weights/target_mode to push this over the threshold, per the
spec's explicit instruction not to tune to force gates.

## Gate 2 (known tactics): FAIL, badly

`tried=200 skipped=0 hits=25 rate=12.5%, gate >=60%`

On mateIn2/kingsideAttack-themed Lichess puzzles, the puzzle's actual solving
move lies in the strict ascent cone (tau=0, king_zone target) only 12.5% of
the time -- a 4.8x shortfall from the 60% bar, not a near-miss.

**Working diagnosis, not yet tested further**: the ascent cone as specified is
a **one-ply, non-forcing** construct -- `damage(m)` is computed by simulating
the move `m` and reading the resulting control field directly, treating the
opponent's actual reply as irrelevant. But mate-in-2 and real kingside attacks
are inherently **forcing sequences**: the winning first move is frequently a
check or a threat that removes the opponent's freedom to punish whatever
"damage" it appears to cost elsewhere on the board. The current `damage(m)`
calculation has no way to know the position is forcing, so a genuinely correct
mating move that "loses" control of an irrelevant square (because the mover's
attention and pieces are elsewhere) gets penalized by `M(g)*min(D_m(g),0)`
exactly as if the opponent had a free move to exploit that loss -- which they
don't, if it's check. This is a plausible, falsifiable explanation for why the
one-ply field undershoots so badly on precisely the puzzle category that's
*defined* by forcing sequences; it has not been separately verified (e.g. by
checking whether puzzles where move 1 gives check fail this gate less often
than puzzles where it doesn't) and should not be treated as confirmed.

## Gate 3 (gambit case study, the project's central hypothesis): mixed, does
## NOT confirm cleanly

| gambit | White raw control-sum, accepted | declined | accepted higher? |
|---|---|---|---|
| Danish | +5.30 | -1.40 | **yes** |
| Evans | +0.70 | +3.60 | no |
| Benko | -0.10 | +2.70 | no |

1/3 gambits show the hypothesized pattern (more force-concentration for the
sacrificer immediately after the sacrifice, vs. the declined line). Reported
exactly as instructed -- no weight tuning attempted to improve this number.

Note on method: `cone_size`/`best_gain` as computed are always for whichever
side is TO MOVE in the given position; since both the accepted and declined
lines above end with an even ply count (Black to move), those two derived
scalars describe the *defender's* cone, not the sacrificer's -- not the
comparison the gate actually wants. The White-POV raw control-sum
(`weighted_attacker_field(board).sum()`, mover-independent) is the metric
actually reported above as the honest comparison; it is a cruder proxy than
what the spec's cone_size/best_gain scalars were meant to provide, and this
mismatch (needing a same-side-to-move comparison point, one ply later in one
of the two lines, or a mover-independent formulation of gain/damage) is itself
worth fixing before this gate can be evaluated as originally intended.

## Interpretation

The spec's own framing (section 5, Phase 4) already anticipated the
possibility that a one-ply, hand-defined attacker-count field is the wrong
level of description for tactical force concentration -- Phase 4 exists
specifically to ask "does a model trained on real chess already represent
this better than the hand-coded version does?" Given gates 1-3's results, that
question is now more load-bearing, not less: the hand-coded field captures a
weak-to-moderate amount of real signal (gate 1) but clearly misses the
forcing-sequence structure that mate-in-2/attacking puzzles are built from
(gate 2), and doesn't cleanly validate the project's own central hypothesis on
a small hand-picked case study (gate 3, 1/3).

Per the spec's non-goals and this gate's explicit stop instruction: NOT
proceeding to Phase 3 (GAB transformer training on these labels) until this is
discussed. Phase 4 (probe a frozen existing trunk for whether it already
encodes something like this, reusing tonight's already-built and validated
probing infrastructure -- linear/MLP probes with Hewitt & Liang control tasks,
positive controls, layer sweeps) is comparatively cheap and doesn't depend on
these gates passing; it's a reasonable next step regardless of what's decided
about gates 1-3.

## Addendum: gate 3 reframed as committor decay (Kaveh's correction, 2026-08-02)

Kaveh's correction to the static gate-3 design: compensation isn't a property
of the position right after the sacrifice, it's whether the winning
probability (committor) HOLDS UP over the following moves -- even a genuinely
sound sacrifice should see its committor decay if the attacker fails to find
the continuation, and hold (or grow) if they keep finding forcing/cone-
building moves. Built `experiments/controlfield_gate3_decay.py`: from each
gambit's accepted position, two branches for the sacrificer's next 8 own
moves -- PRESSING (Stockfish's own best move each turn) vs DRIFTING (a
deliberately passive, non-immediately-losing legal move each turn); opponent
plays SF-best in both. Tracked SF committor (WDL, sacrificer POV, depth 12)
and in-cone occupancy of the actual move played, at every sacrificer turn.

`VERDICT gate3-decay: 2/3 gambits show pressing-committor-trend > drifting-
committor-trend | VERDICT gate3-decay-cone-check: in-cone rate pressing=16.7%
drifting=0.0% -- cone tracks pressing`

- **Evans and Benko confirm the reframed hypothesis** -- Benko strongly so
  (pressing trend +0.025, drifting trend -0.267 -- the drifting branch
  collapses to a near-certain loss within 2 sacrificer moves, pressing holds
  roughly flat). Bonus: the ascent cone's in-cone rate is higher in the
  pressing branch (16.7%) than drifting (0.0%) across both re-runs (k=5 and
  k=8) -- small sample (12 sacrificer-turns total) but directionally
  consistent, not cherry-picked (same result both times).
- **Danish does NOT confirm** -- but for an informative reason, not a
  methodology failure: committor is floor-clamped near 0.000 in BOTH
  branches from the very first move, and the game ends in a loss for White
  within 4 sacrificer turns regardless of which branch is played. Reading:
  at Stockfish depth-12 strength, the Danish Gambit accepted line is simply
  losing with no real compensation to track the decay OF -- this matches
  established engine theory (the Danish Gambit is considered practically
  refuted against accurate defense; whatever compensation it offers is a
  human-level psychological/practical effect, not one Stockfish-strength
  play concedes). This is a genuine boundary condition for the hypothesis
  (it's about whether REAL compensation decays with bad follow-up, not
  whether any accepted gambit looks equal no matter what), not a
  counterexample to it.

**Net reading**: Kaveh's reframing produces a much more informative and
better-supported test than the original static gate 3. 2/3 (arguably 2/2
among the gambits that have real engine-level compensation to begin with)
confirm that committor decay tracks whether the attacker keeps finding
cone-building moves. Still a very small sample (3 gambits, hand-picked, one
engine depth) -- this is directional evidence worth taking seriously, not a
statistically powered gate. A real gate-3-replacement at scale would need
this same design run over many real sacrifices mined from game data (not
just 3 hand-picked openings), which hasn't been built.

## Fix applied: exploitability-gated damage (Kaveh, 2026-08-02)

Per the gate 2 working diagnosis above ("the ascent cone is a one-ply,
non-forcing construct"), implemented the fix: `damage(m)` now only counts a
lost square against the mover if the OPPONENT has at least one LEGAL reply
that actually lands on/captures that square in the post-move position
(`move_derivatives(..., exploitable=True)` returns `E`, a (n_legal, 64) bool
mask; `damage[i] = (M * min(D[i],0) * E[i]).sum()` in
`catspace/controlfield/derivative.py::ascent_cone`). This is one extra ply of
LEGAL-MOVE enumeration on the already-computed post-move position -- not a
search or eval, consistent with the spec's "no search" non-goal -- and
directly fixes the diagnosed blindness: a checking move that "loses" control
of a square the opponent is now forced (by check) not to touch was previously
penalized as if the opponent had a free reply to punish it; now it isn't.

Added a permanent regression test
(`tests/test_derivative.py::test_damage_ignores_squares_opponent_cannot_legally_reach`)
using a constructed discovered-check position (White Rd1+Bd3+Re4, Black
Kd8+Re8; Bb5+ drops the bishop's defense of e4, D_m(e4)=-0.9, M(e4)=1.0, but
Black's only legal replies are king moves -- Black cannot legally play Re4
even though the rook could reach it if not in check). 10/10 tests pass.

**Gates rerun at full scale after the fix**:

| gate | before | after | gate | verdict |
|---|---|---|---|---|
| 1 (sanity) | rho +0.144 | **rho +0.249** (p=1.5e-29) | >=0.15 | **PASS** |
| 2 (known tactics) | 12.5% | **87.5%** | >=60% | **PASS** |
| 3 (gambit, dynamic decay version) | 2/3 confirm | 2/3 confirm (unchanged direction; in-cone rates shifted pressing 16.7%/drifting 8.3%, was 16.7%/0.0% -- still directionally consistent) | -- | reported |

Gate 1 improved 73% in effect size (0.144 -> 0.249) and gate 2 improved 7x
(12.5% -> 87.5%) from a single, principled, well-diagnosed fix -- strong
confirmation that the one-ply/non-forcing blindness was the real, dominant
cause of the original gate failures, not a sign the underlying control-field
concept was unsound. Gates 1 and 2 now both clear their bars. Gate 3 (the
project's central hypothesis, tested via committor decay) was already mostly
confirming (2/3) before this fix and is unaffected by it in direction.

**Updated recommendation**: with gates 1 and 2 now passing and gate 3 (in its
corrected, dynamic form) mostly confirming, the case for proceeding toward
Phase 3 (training a GAB transformer on these labels) is now substantially
stronger than it was before this fix -- worth discussing with Kaveh as the
next decision point, not proceeding automatically (the spec's stop-and-report
instruction was about a genuine failure state; this is a genuine pass state,
but Phase 3 is still a large resource commitment that deserves an explicit
go-ahead, not silent continuation).
