# Design Notes — agreed research designs (2026-07-10)

These are the designs agreed during plan review. The refactor
(docs/IMPLEMENTATION_PLAN.md) builds only their *structures*; the research
itself is phased after the refactor stabilizes. Kept here so the rationale
survives context loss.

## 1. Eval head (embedding-space position eval)

Purpose: search in the embedding should be guided by position strength as
well as reachability — but the eval must be a function of the position IN
THE FIELD (`F(s)`), never the raw board. It doubles as a scientific probe:
does reachability geometry encode value?

- **Two training modes, in order**: (1) FROZEN PROBE — train FB, freeze,
  fit the head on top (labels cannot corrupt the planner's geometry; clean
  measurement of what the geometry already knows). (2) JOINT multi-task —
  auxiliary loss shapes the embedding; adopt only if the A/B harness shows
  it helps *planning* metrics, not just eval metrics.
- **Two heads**:
  - *Normative*: labels = Lichess `%eval` (Stockfish-flavored, ~6% of
    games) + local Stockfish backfill (UCI, depth 12–16, cached once at
    shard-build time, never overwriting real labels). Same-flavor backfill
    keeps the label column one distribution (lc0 would mix two).
  - *Descriptive*: labels = actual game Results per Elo bin — free
    ground-truth on every position; estimates the ω-conditional value.
  - Their divergence map = "trap regions" (normative≈equal, descriptive
    says humans at this level lose) — a discovery output, roadmap E1.
- **Target scale**: win-probability-like. v1: `tanh(cp/400)`, mate → ±1.
  (A 3-way WDL softmax is the designated upgrade if the draw channel is
  needed.)
- **Success metric is NOT MSE to shallow labels**: audit on a small
  stratified held-out sample against SF depth ~30 ranking correlation,
  lc0 WDL (audit only — too slow and wrong flavor for bulk labels), and
  arena Monte-Carlo outcomes under a fixed opponent. If the head disagrees
  with shallow labels but agrees with the deep audit, that is the
  positive finding ("better than the label"), certifiable by the e-test
  harness.
- **Guardrail**: eval must not become the primary search signal (that
  collapses the thesis into a plain value network). Reach stays primary;
  eval is for goal *selection* and far-field tie-breaking.

## 2. Hierarchical planning (M1.5 research phase)

Seed idea: when the readout rejects a move (e.g., a capture refuted by a
defender), the refutation is data — it names the obstacle. Find what would
need to change for the SAME move to work, express it as a vector/region in
the embedding, navigate there first. Inverse of the killer-move heuristic.

- **Enabling sets**: E(m, s) = states where move-pattern m applies and its
  current refutation (the B_opt reply) no longer works. Exactly enumerable
  at toy scale (GoalSpec with z_E = ΣB(g)); learned precondition head
  (F(s), move-embedding) → z_E at full board. Contrast sets (same move
  identity succeeds vs fails) give the precondition direction Δ; quantizing
  Δs (ConceptQuantizer) yields a discovered vocabulary of TECHNIQUES.
- **MoveIdentity** (pluggable, registry): syntactic (piece+target) /
  region_pair (quantizer token → token) / displacement (clusters of
  ΔF = F(s′) − F(s); same vector space as precondition Δs — composable).
  Semantic identity plans; syntactic executes at the leaf.
- **PlanSelector (RL)**: plans are candidate z-vectors, so plan choice is
  z-selection over a small discrete space + meta-actions {pursue,
  decompose, abandon, act-greedy, stop-decomposing}. Ladder: GreedyReach →
  ThresholdRules → LearnedSelector (options/semi-MDP; termination = replan
  events; REINFORCE with Δreach/Δeval shaping + outcome) → PlanMCTS (tree
  over intentions; cone = rollout policy; eval head = leaf evaluator).
  Guard: audit selected plans against exact DTM at toy scale
  (reward-hacking check on the eval-head shaping).
- **Recursive decomposition**: hop (a→b) is EXECUTABLE iff MC rollouts
  under the current field reach b from a within horizon h with prob ≥ p
  (verified, not estimated). Subgoal generators in preference order:
  enabling sets; region-graph pathfinding (tokens + observed transitions);
  geodesic midpoints argmax_m min(reach(a→m), reach(m→b)).
- **Give-up rules** (decomposition must stop when no path is likely):
  (1) no midpoint improves the bottleneck (hop is hard, not long);
  (2) unlikely territory — below calibrated availability threshold τ
  (WIN/DRAW-frontier calibration, audited against dtm=∞ ground truth) or
  below an eval-head floor; (3) dry-out — two successive decompositions
  fail to improve the bottleneck; (4) depth/compute caps (anytime: best
  executable prefix or INFEASIBLE).
- **BlockReason + wake triggers** ("remember WHY infeasible; listen for
  moves that might change it"): block records {rule, bottleneck hop,
  feasibility, refutation key, enabling direction Δ} — the why and the fix
  are the same object. Two-tier wake: continuous drift watcher
  ⟨F(s_now) − F(s_blocked), Δ⟩ ≥ θ (catches the opponent fixing our problem)
  + discrete event index keyed on MoveIdentity keys / region entry /
  stratum crossing / material delta. Waking re-checks only the bottleneck
  hop; woken plans join the selector's candidates. Anti-thrash: hysteresis
  + doubling cooldown. Prior-art note: BDI suspended intentions, but with
  conditions in the LEARNED embedding. Legibility: viewer shows blocked
  plans with "waiting for: X".
- **Pre-registered M1.5 gates**: (a) KRkn conditional-capture audit —
  discovered Δ aligns with the exact defended-predicate; conditional plans
  beat the greedy baseline (e-tested); (b) improvement on the open KRkn
  deep-conversion frontier (DTM 21–43 vs optimal defense; flat field ≈3%);
  numeric target set after a pilot.

## 3. Known internal risks (tracked)

- Reach = on-policy successor measure, not optimal goal-reaching distance:
  midpoint decomposition on a fixed field is conservative; re-run
  decomposition after each PI round rather than once.
- Embedding/quantizer nonstationarity: plans, block reasons, and triggers
  reference embedding-version-specific objects — version the
  FittedMap/quantizer; invalidate or re-map plan memory on version bumps.
- Descriptive-head label correlation: all positions of a game share one
  result — split train/val BY GAME, weight later plies higher.
- Smoothness assumption in the drift watcher: sharp tactical changes may be
  small in F-space; backstop = periodic full re-check + always verify a
  woken plan with the shallow exact-rules readout before executing.
- Selector RL sample efficiency: plan-level decisions are sparse — warm
  start supervised from a DTM-computable oracle selector at toy scale.
- Arena fairness: compare policies at equal compute budgets; report
  think-time/node budgets alongside Elo-like results.
