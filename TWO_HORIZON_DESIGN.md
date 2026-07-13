# Two-horizon architecture — design spec

Status: **approved 2026-07-13** (Kaveh), pre-build. Authored on Opus after the
Fable→Opus handoff. This doc is the spec the build is checked against; update it
if the design changes during implementation.

## Motivation

The single measured finding across 18 rounds (see JOURNAL.md, and GLOSSARY.md
"Short-horizon vs long-horizon discrimination"): a single d=64 embedding must be
both a sharp **tactician** (rank the ~30 legal moves; don't hang a rook) and a
calibrated **strategist** (is this position on the 40-ply path to mate). These
compete — every lever that improved one taxed the other. The fix is structural:
give each job its own parameters instead of forcing one function to compromise.

Two observed failure modes, and the head that fixes each:
- **Hangs material** (blunders a rook, ACPL ~289) → fixed by `near` at the
  search-pruning stage.
- **Aimless shuffling** (draws tablebase-won endgames by repetition) → fixed by
  `far` at the leaf-evaluation stage.

## Architecture

Shared board-encoder trunk → **two heads**:
- `near`: `F_near(s)`, `B_near(g)` — sharp *local* discrimination.
- `far` : `F_far(s)`,  `B_far(g)`  — calibrated *long-range* distance-to-goal.

Approved choice: **start with a shared trunk** (generic board features — piece
placement, attack maps — are not the locus of the competition; the conflict is
head-level). Fallback if the fitness probe shows the heads still fighting through
the trunk: split the trunk deeper. Two separate goal banks / zgoal sets, one per
head (near-zgoal and far-zgoal), since the two spaces are distinct.

## Roles in search (when near, when far)

- **near = the steering wheel.** Beam selection + move ordering: at every ply,
  keep only the top-k moves by near-score, so tactical blunders are pruned
  before expansion. near is accurate over 1–8 plies — exactly the beam's need.
- **far = the leaf evaluator.** A search leaf's value = its far-score toward the
  goal region (calibrated distance-to-mate). Supplies the strategic gradient that
  converts won positions. far is accurate over the long leaf→goal span.
- **Terminals** (mate / draw sentinels) override both, unchanged.
- Approved choice: **pure-far leaves** for v1 (near-pruning should already
  guarantee tactical soundness at the leaf). Add a small near term to the leaf
  score only if leaves prove tactically loose in evaluation.

## Training

Same (s, g) Lichess pairs, **stratified by ply-gap** — the stratification *is*
the specialization:
- **near head:** contrastive InfoNCE on **short-gap** pairs (1–8 plies).
  Optimizes local ordering. Cosine / short-range; does NOT need the quasimetric.
- **far head:** quasimetric + ply-gap calibration (+ region/asymmetry structure
  later) on **long-gap** pairs (≥16 plies) plus state→goal pairs. Optimizes
  calibrated long-range distance.
- Losses summed; each head's gradient specializes; the trunk learns shared
  board features.
- Approved choice: **hard ply-gap cut** for v1 (near ≤ 8, far ≥ 16, a dead zone
  between to keep them distinct). Revisit soft (gap-weighted) stratification if
  the hard cut shows boundary artifacts. Crossover motivated by the retrieval
  probe: discrimination is strong to ~10 plies then falls off a cliff by 20–50.

Data source: the human 4GB shard (the promoted incumbent's recipe — no self-play
mix, which was the play drag). The pair sampler must expose the anchor and goal
plies so the trainer can route each pair to the right head; `LichessPairSource`
already carries `ply` and `ply_g`.

## Evaluation (pre-registered gate — "evaluate them soundly")

Sound evaluation is the whole point; nothing promotes without clearing all of:

**Structural (fitness probe, the combination one embedding never hit):**
- `near` k=1 horizon-retrieval ≥ 0.95 (short-horizon sharpness preserved).
- `far` nearest-exemplar Spearman ρ ≥ 0.30 (clears the ~0.25 single-embedding
  ceiling — real long-range calibration gain).
- Both **simultaneously** in one checkpoint.

**Play (vs the incumbent `lichess_fb_4gb_qm_plygap_only.pt`):**
- KRRvKBP n=60 conversion ≥ 0.567 (hold or beat the incumbent).
- ACPL n=400 ≤ 289 (hold or beat — no robbing one axis to pay the other).
- Both hold/improve. If only one improves at the other's expense, the two-horizon
  split has NOT resolved the competition and is rejected (or retuned, ≤2 tries).

Search budget for all play evals: the operating `max_nodes` chosen by the
node-budget sensitivity sweep (in flight; ceiling ~1600 = 10× below competitive
Leela).

## Build order

1. `TorchFB`: add `near`/`far` heads (+ config flag `two_horizon`), each with its
   own `score`/`distance`; keep single-head path byte-identical when off.
2. Loss: route short-gap pairs → near InfoNCE, long-gap pairs → far
   quasimetric+ply-gap; sum. Unit-test the routing + that off-mode is unchanged.
3. `FBSearchPolicy`: near for beam/ordering, far for leaf value (two-horizon
   mode); single-head path unchanged. Unit-test legality + mate short-circuit.
4. zgoals: build+store near-zgoal and far-zgoal at save.
5. Train on the 4GB human shard; run the pre-registered gate above.
