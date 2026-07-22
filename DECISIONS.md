# DECISIONS.md — settled decisions, data, flags, dims

Canonical catalogue of things we have **settled** so we stop re-deriving them. If you want to change one,
that's a **deliberate reversal** (say so, in the JOURNAL) — not a rediscovery. Last updated 2026-07-22.
See JOURNAL.md for the evidence behind each line.

---

## 1. Field & embedding

- **Embedding dim `d = 512`** for the real field (the nucleus / geometry field). `d=64` was the earlier
  lichess-QRL experiment size and is weaker — the incumbent going forward is **512**.
- **Field type: IQE quasimetric, trained by a QRL / L1-reachability-geometry objective — NOT InfoNCE.**
  IQE + InfoNCE collapses to loss=ln(N) (scale-blind, ignores the triangle inequality). Do not use InfoNCE.
- **Encoder:** 128 channels, 10 residual blocks, **GroupNorm** (NOT BatchNorm), spectral-norm ON,
  `iqe_components=32`, `iqe_leak_beta=10`, `tau=0.1`, `omega_free_field` (elo/clock conditioning).
- **`cert_base_full.pt` is DEAD** — a stale BatchNorm checkpoint the current GroupNorm code cannot load
  (`unexpected running_mean/var keys`). Do not use it; the play server and everything downstream must run on
  a GroupNorm field (e.g. `iqe_nucleus_*`).
- **Feature convention: BOARD_ONLY channels (18, 19) are ZEROED** in the geometry-training path (`bp()`).
  Any distance/embedding measurement must zero them too, or the numbers are off-convention.

## 2. Objective / training recipe (the "full" field)

The field must be trained with **all four** terms together — "sharp + aligned + non-collapsed + optimal":

1. **Geometry (sharpness):** `L_pos` = (d(F(s),B(s')) − 1)² on legal 1-ply edges; `L_hard` = push
   irreversible reverses ≥ 1+margin.
2. **DTM alignment (direction):** `L_dtm` = (d(F(s),MATE_W) − dtm/scale)² on won positions. Geometry gives
   sharpness; DTM gives the mate-pointing gradient **where human data is empty** (tablebase range).
3. **Repulsion (anti-collapse):** two-tier — EVERY random pair → floor 12, material-UNREACHABLE → floor 30.
   `w_repel = 4` (was 1 — too weak to beat L_pos's pull-to-1). **The cure for collapse is repulsion, not width.**
4. **Best-play edges:** consecutive plies of Stockfish/tablebase continuations added as optimal-successor
   `L_pos` edges (correct edges, not just legal ones).

**Collapse is the root failure mode.** The old objective's repulsion was under-powered (w=1) AND too sparse
(material-unreachable only) → the field collapsed to ~6 effective dims of 512, with d_step ≈ d_rand (a real
1-ply successor no closer than a random board). Effective-rank + d_rand/d_step ratio are health gates on every
field.

## 3. Planner / engine architecture (long/short)

- **Field = COARSE long-range navigator, NOT a fine move-prior.** Evidence: field-guided search reaches the
  near-mate region ~97% vs pure-search 77% at low budgets, but its per-move ranking is at chance.
- **Uniform prior beats the field prior** for local search; the field is a bad move-orderer.
- **Long/short with handoff:** field plans a SUBGOAL → **uniform-prior MCTS** navigates to it → **handoff to
  pure MCTS + mate_stop** when within ~5–6 plies of the goal (≤5 pieces) OR the planner stalls (coarse
  distance-to-mate stops improving). **EXECUTE phase uses PURE search — the field value HURTS near mate.**
- **The subgoal selector is the RL-swappable seam** (Kaveh: "eventually the planner is RL" → PlanSelector).
  The current composed-distance heuristic is a placeholder; keep it behind a clean `select_subgoal()`.
- Engine lives in `experiments/longshort_engine.py`; wired into `experiments/viz/play_server.py`
  (`--planner`, routes `/engine_move`). Still weak (0.12 mate vs pure 0.42) until the field collapse is fixed.

## 4. Established findings — do NOT re-litigate

- **Fields do NOT extrapolate reachability to an unseen regime.** DTM alignment decays off-material (nucleus:
  3p 0.70 → 6p 0.21) and proximity-in-embedding does NOT predict accuracy (corr ≈ 0). The literature agrees:
  quasimetric "stitching" is composition WITHIN transition support, not extrapolation.
- **Distillation DOES work** — the field learns an unseen regime from targets (6-piece DTM 0.19 → 0.53). Grow
  support outward via search-backed teachers (MCTS reaches the trained region, back up, distill).
- **Conversion is SEARCH-limited.** Pure MCTS + mate_stop mates ~0.42 of the KRRvKBP toy; field-guided
  conversion does NOT beat it (the field hurts the fine mating task). The field earns its keep in the coarse
  phase, not the finish.
- **No supervised probes drive moves** (Kaveh's rule). Concepts-as-CAV-directions do NOT transfer cross-regime.
- **Exact concept computations are DIAGNOSTIC ONLY** (Kaveh 2026-07-22): the king escape-volume flood-fill
  (ladder_mate.escape_volume) measured that the cornering concept guides search almost as well as the DTM
  oracle (0.75 vs 0.85 @400n) -- but it must NEVER be a play-time value. It is an instrument (like eff-rank)
  for judging whether the LEARNED field/heads have acquired the concept.
- Position-level DTM correlation (even 0.53) ≠ move-level ±1-ply discrimination — the field can't be a fine
  prior even when its value is decent.

## 5. Data we HAVE and MUST use for full trainings

| Data | Path | What / size | Used for |
|---|---|---|---|
| Geometry pool | `data/derived/geom_pool.npz` | 357,223 positions **+ DTM labels** (37M) | L_pos/L_hard positions; L_dtm (won) |
| Geometry edges | `data/derived/geom_pool_edges.npz` | 2,172,150 legal 1-ply edges, 9% irrev (433M) | L_pos / L_hard |
| Best-play continuations | `data/shards/sf_cont_endgame_v1/` | 4,924 conts / 148,714 pos → **143,790 edges** (Stockfish + tablebase-completed) | best-play L_pos edges |
| Tablebase DTM (endgame) | `data/derived/dtm_endgame.npz` | 24,000 KRRvKBP-tree pos, DTM 1–193, 3–6 pieces | DTM alignment, distillation, subgoal bank, eval |
| Syzygy tablebases | `data/syzygy/` | KRRvKBP, KRRvKB, KRRvKP, KRRvK, KRvK … | exact play/defense, DTM generation |
| Human lichess | `data/shards/lichess_db_standard_rated_2019-01.prefix{256mb,1gb,4gb}/` | human games; 2.54% ≤6-piece, 0.005% KRRvKBP | middlegame/human field, continuation seeds |
| KRRvKBP test set | `artifacts/experiments/krrkbp_test_n200.json` | 200 fixed KRRvKBP starts | the conversion eval set |
| Winning-region | `data/derived/stratified_perfect.npz` | White-mate / winning-simplification region | long/short goal region |

**Continuations are generated by** `experiments/gen_stockfish_continuations.py` with `--max-pieces 12`
(low-piece seeding) + tablebase completion once ≤ frontier pieces — human middlegame seeds barely reach the
endgame otherwise (2.8% vs 50% with low-piece seeding).

**Fields:** `iqe_nucleus_gn.pt` (d=512, DTM-anchored but COLLAPSED, rank 6) · `nucleus_distilled.pt` (DTM
distilled to 0.53@6p, still collapsed) · `iqe_nucleus_full.pt` (the full-recipe field: repulsion + DTM +
best-play — the intended incumbent once validated).

## 6. Full-training flags

**Field (geometry) — `experiments/train_geometry_l1.py`:**
```
--ckpt <resume field> --out <save> \
--data data/derived/geom_pool.npz --edges data/derived/geom_pool_edges.npz \
--bp-shards data/shards/sf_cont_endgame_v1 \
--steps 2500 --batch 256 --lr 3e-4 --device auto \
--w-pos 2 --w-hard 1 --hard-margin 15 \
--w-repel 4 --repel-floor 30 --repel-floor-all 12   # anti-collapse (was w-repel 1) \
--w-dtm 1 --dtm-scale 20                              # DTM alignment
```

**Lichess QRL field — `experiments/train_lichess_fb.py`:** `--l2-preset iqe-qrl` (NEVER infonce),
`--qrl-objective`, `--qrl-unreach-weight 8 --qrl-unreach-floor 30` (repulsion), `--selfplay-shards
<continuations> --selfplay-frac 0.35`, `--dtm-hinge data/derived/dtm_endgame.npz`, `--resume-lr-scale 0.1`
(guards resume-at-peak-lr collapse). `--struct-weight` (board-structure head) was **REJECTED** — no effect.

**Conversion eval — `experiments/conversion_field_mcts.py`** (tablebase-free) / `planner_longshort.py`
(tablebase frontier). Health check: `d_step` vs `d_rand` + effective rank (BOARD_ONLY zeroed).
</content>

---

## 7. DECOUPLED: field geometry vs DTM head (settled 2026-07-22)

**Do NOT overload the quasimetric distance `d`.** Repulsion fixes the near/far metric (successor-vs-random
8× separation) but the field stays ~1-D, and that single axis encodes MATERIAL-reachability -- which COMPETES
with mate-distance (DTM) for the one dim. Forcing `d` to be both broke DTM alignment (0.19->0.05) and did not
raise effective rank (repulsion needs only 1 axis to separate near/far; raising RANK would need a decorrelation
/covariance term, not more repulsion). Resolution:

- **`d` (the field's quasimetric) = REACHABILITY GEOMETRY ONLY** -- trained by geometry (L_pos/L_hard) +
  repulsion + best-play edges, **`--w-dtm 0`** (no DTM term). Used for subgoal reachability / navigation geometry.
  Incumbent geometry field: `iqe_geom_field.pt`.
- **DTM / mate-distance = a SEPARATE head/net.** Candidates, spearman(pred,DTM) held-out by piece count
  (3p / 4p / 6p):
  - plain CNN `data/derived/sep/dtm_cnn.pt` (`train_dtm_cnn.py`, board->DTM): **0.89 / 0.61 / 0.355**
  - distilled field `data/derived/sep/nucleus_distilled.pt` (d distilled to DTM): **0.88 / 0.71 / 0.53** -- the
    strongest DTM predictor, BUT its `d` is DTM-overloaded (not a clean geometry), so it is a value model, not
    the navigator.
  - overloaded geometry-`d` (DTM crammed into the quasimetric): **0.05 @6p** -- this is the failure the
    decoupling avoids.
  A DTM head on the frozen *geometry* encoder is TBD (may beat the plain CNN by reusing richer 128ch/10-block
  features). Whichever we pick, it is SEPARATE from the geometry field's `d`.
- The planner reads **`d` for reachability**, **the DTM head for mate-distance** -- whichever it needs. This
  matches train_geometry_l1's original intent ("NO DTM -- that moves to the L2 head").
- **VALIDATED 2026-07-22** (`experiments/validate_decoupled.py`, printed verdicts): the clean geometry field
  `iqe_geom_field.pt` (--w-dtm 0) hits **ratio 33.0x** (was 8.2x with DTM crammed in) and **eff-rank 15.1/512**
  (was 1.3-5.5). **The DTM term was the rank-crusher** -- same repulsion/steps, removing L_dtm alone raised
  rank ~10x. Decoupling confirmed: spearman(d,DTM) ~ -0.04..-0.10. DTM-head bake-off: frozen-trunk head
  0.72/0.52/0.27 (3p/4p/6p) LOSES to the plain CNN (0.89/0.61/0.36) and the distilled field (0.88/0.71/0.53)
  -> **nucleus_distilled.pt stays the DTM/value head** (a separate value model, not the navigator).
- **Open regression:** asym INVERTED on the clean field (irr 5.6 / rev 9.1 = 0.61x; was 2.03x correct on
  full2). Cause: repel-floor-all treats reversible reverses (turn parity -> never 1-ply edges) as random
  pairs and inflates them; L_hard unconverged at 2500 steps. Candidate fixes: exempt known reverse pairs
  from the all-pairs floor, and/or longer training / higher w_hard. Not yet applied.
- Rank note: the old "needs a decorrelation term" hypothesis is WEAKENED -- rank recovered to 15 just by
  removing the DTM term; revisit decorrelation only if 15 proves insufficient.

## 8. Search allocation: subgoals bias the PRIOR, never the VALUE (Kaveh 2026-07-23)

- **Principle (settled):** a subgoal ("save the bishop", "trap his rook", "corner the king") concentrates
  the search's PRIOR -- focused, human-like, few-line search -- while the leaf VALUE stays the GLOBAL
  objective (distance-to-mate/win). Humans' sacrifice-blindness = the subgoal contaminating the value;
  humans' threat-strength = a superb prior. Keep the strength, drop the blindness: the value never adopts
  the subgoal, so sacrifice lines that beat the subgoal remain discoverable through the value term.
- **Implementation sketch:** pi = alpha * pi_subgoal + (1-alpha) * pi_global in MCTS's policy_fn (the
  sockets already exist: policy_fn vs value_fn). alpha = the focus dial (1 = blind human, 0 = priorless
  global search); candidate: gate alpha on the spread/certainty field (plan-alignment prior, commit b2626fe).
- **Concept family:** trap-a-piece == constrain(piece) -- the cornering/escape-volume operator applied to
  ANY target piece, not only the king. Mate = constrain(king). Train/probe it as ONE parameterized
  relational primitive.
- **Planned instruments:** (a) threat-perception probe (is-piece-hanging linear probe with in-stratum
  controls) on every field; (b) sacrifice-required positions mined from tb (won, every material-preserving
  move raises DTM, some sacrifice lowers it) + alpha-sweep of mate-rate/search-cost = the blindness curve.

## 9. NORTH STAR (Kaveh 2026-07-23): the strength-per-node frontier

Concepts + tiny search ~= Stockfish strength provably EXISTS: humans do it at ~10-100 imagined
lines (single-digit-ish ACPL), AlphaZero did it machine-verified at ~80k sims vs Stockfish's ~70M
nodes at parity. Every engine result gets framed on this frontier (strength at fixed node budget /
nodes-to-convert). First measured instance (KRRvK ladder, 2026-07-22): pure 0.12 -> one concept
0.75@400n (~1.5k nodes) -> oracle ~1.2k -> Stockfish ~54k. The goal is the human corner.

## 10. Checkpoint policy (Kaveh 2026-07-23): ALL training runs keep step-suffixed ladders

Every trainer saves `<out>_step{N}.pt` at --ckpt-every intervals PLUS the rolling latest at --out.
(train_lichess_fb always did; train_geometry_l1 and train_dtm_cnn fixed 2026-07-23 — geometry
previously overwrote one file.) Ladders enable early acceptance readings mid-run (e.g. the
in-stratum probe at step 5000 of the 30k human-field run) and post-hoc best-step selection.
`iqe_geom_field.pt` predates the fix (final-only) — validated, kept.

## 11. TRAINING_STANDARDS.md is the standing do's/don'ts for ALL trainings (Kaveh 2026-07-23)

Checkpoint ladders; ONE richest input format everywhere (REVERSES the BOARD_ONLY zeroing of sec 1,
from the next run onward); never overwrite + metadata-in-checkpoint; MLflow via catspace/tracking.py
(no hand-rolled tracking); smoke-first; gate battery; fixed eval sets; watchdogs. See the file for
the full list with scars.

## 12. Layered engine package + module map (refactor 2026-07-23)

- **`catspace/engine/`** is the layered engine (Kaveh: "layered, try different models in each layer"):
  Protocols in `interfaces.py` (ValueModel / MovePrior / SubgoalSelector + Region + SearchOutcome);
  `fields.py` FieldModel (convention-aware ckpt wrapper -- resolves zeroed-vs-full planes from stored
  args; ALL embed/distance calls chunked); `values.py` (Constant / TablebaseValue[DIAGNOSTIC] / DTMCNN /
  FieldGoalDistance); `priors.py` (Uniform / MixturePrior alpha-dial); `search.py` MCTSSearch (returns
  evals_used); `engine.py` LayeredEngine (injected layers, plan->execute handoff). All 268 tests pass.
- **Canonical utility homes:** `catspace/tb.py` (TB, tb_best_move, white_pov_value, rollout, rollout_dtm --
  old experiments/ locations re-export); `catspace/diagnostics.py` (escape_volume, mate_pattern,
  mate_labels, material_count, eff_rank -- DIAGNOSTIC-ONLY rule in the docstring).
- **`catspace/incubator/`** = the dedicated home for code without a home yet (rules in its __init__).
- **MLflow "registry" experiment** mirrors the incumbent models + datasets (13 entries; update via
  `experiments/register_incumbents.py`). DECISIONS sec 5 stays the prose source of truth.
- Predating layers to FOLD IN next: experiments/compute_layer.py (uncertainty-carrying tool layer) +
  experiments/catspace_engine.py (coded policy) + catspace/planner/ + goal_bank.py.
