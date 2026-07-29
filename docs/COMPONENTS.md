# COMPONENTS — what exists, where, and its status

*Rewritten 2026-07-28 for the current (frozen-trunk, two-evaluator) architecture; the pre-rebuild
component map is preserved at `docs/archive/COMPONENTS.md`. Status tags: **live** (in the current
line), **instrument** (analysis/eval only), **legacy** (pre-rebuild; kept, not maintained).
Design: `THESIS.md`. Plan: `../MILESTONES.md`.*

## Substrate (borrowed, frozen)

- **Board embedding φ(s)** — frozen Leela-family trunk (T1-256x10 distillate) via
  `lc0 leela2onnx` → `lczerolens` torch module; 64-d trunk features + WDL + moves-left heads.
  Batched forward ~277 pos/s CPU, faster on MPS. **live**
- **Human move prior** — Maia-2 (rating-conditioned, both players' Elos), batched ONNX
  inference (~10 ms/pos batched); the universal rating prior (μ=0). **live**
- **Referee** — Stockfish, weighted by the SF reliability map (~97% vs tablebase; errors on the
  win/draw boundary). Committor/crossing labels are SF-refereed by locked decision. **live**
- **Endgame truth** — syzygy ≤7p via `catspace/tb.py` (persistent sqlite probe cache,
  `journal_mode=DELETE` — WAL is banned, it filled the disk twice). **live**

## Player model — M2 (built, DoDs met)

- `catspace/style/model.py` — `StyleResidual`: z ∈ R¹⁶ residual over the frozen Maia-2 base,
  move-logit linear in z (convex MAP recovery; Laplace posterior). **live**
- `catspace/style/recover.py` — `recover_delta` (weighted MAP, recency weights), `score_nll`.
  **live**
- `catspace/style/estimator.py` — `OpponentEstimator`: the online (Elo, z) filter; history-prior
  + live moves; Elo-from-moves via Maia's 11 rating buckets; infer-then-condition retrieval
  (k≈50, Elo-banded, band widens with Elo uncertainty). Feeds play-time z and the field's
  causal ẑ_opp(t). **live**
- `catspace/style/dataio.py` — cache loader (legacy single-file + resumable shard layouts).
  **live**
- z artifacts: `artifacts/experiments/m2b_style_3k.pt` (2,975 training styles = `delta.weight`).

## Transition / crossing risk — M2a + primitive

- `catspace/transition.py` — the crossing-risk primitive: expected SF-refereed committor swing
  under any move model; opponent model in → exploit flux, our model in → self-blunder/denial.
  **live**
- `experiments/train_iqe_head.py`, `T_phi_rating_clock_*` checkpoints — the fast T predictor
  (context-conditioned crossing rate; clock-aware variants). **live**
- `experiments/m3_flux_gate.py` / `m3_strength_gradient.py`-family — M3 groundwork verdicts
  (location is position-driven, magnitude strength-driven). **instrument**

## Reachability field — the measure evaluator (v1 smoke passed)

- `experiments/build_reach_data.py` — goal bank (k-means on train-φ) + first-hit / censored-
  plies labels; balance audits gate the write. **live**
- `experiments/build_opp_positions.py` — opponent decision reconstruction by exact one-ply
  replay (for the causal ẑ_opp(t) pipeline). **live**
- `experiments/train_reach_head.py` — `ReachHead`: factored ⟨φ_r(s,z,elos), ψ_r(g)⟩, z-free
  goal tower, first-hit BCE + censored-plies head, pre-registered acceptance instrument
  (paired z-lift CIs, wrong-z placebo, calibration bins, bootstrap eff-rank). **live**
- Data: `data/derived/reach/reach_v1.npz`, `data/derived/m2b/cache_dense{,_opp}/` (DVC).

## Basins / committor — M0 instruments (in production use)

- `experiments/msm_basins.py` — MSM + PCCA metastable basins on real games. **live**
- `experiments/committor_by_material.py`, `sf_wdl_by_material.py`, `sf_vs_human_bands.py`,
  `engine_vs_human_basins.py`, `transition_map/bands/time` — the M0 evidence suite; figures in
  `docs/figures/`. **instrument**
- `experiments/sf_reliability_map.py` — SF-vs-tablebase calibration. **live**

## Goal bank & retrieval

- `catspace/goal_bank.py` — harvest/embed goal exemplars. **live**
- `catspace/vectordb.py` — bank sync + k-NN query over field embeddings. **live**
- `catspace/memory/retrieval.py` — `d_to_goal` / `d_to_region` retrieval primitives (the
  memory-index half of "index points from memory"). **live**
- `catspace/memory/plan_store.py` — the engine's live memory (sqlite, DELETE mode): plan ledger
  (intent vs realization — feeds the M4 steering verdict), per-opponent (Elo,z) persistence,
  reserved M7 armed-tactics table keyed on protective SAE atoms. **live, M4 substrate**

## Planner & search — the existence evaluator + navigation (M4/M5 pending)

- `catspace/planner/optionality.py` — the optionality portfolio: soft subgoal sets both sides,
  multipurpose-move prior, opportunistic re-selection + hysteresis; field-agnostic
  (distance_fn seam). Built, 16/16 tests. **live, awaiting M3 subgoal generator**
- `catspace/planner/probe.py`, `catspace/planner/cascade.py` — certified probe bounds ([lo,hi]
  from game-truth only) + LUCB decision cascade. **live**
- Forced-object prover (ε-support df-pn with GHI handling + ∏(1−δᵢ) certificates) — **designed
  (THESIS §3), not yet built**.
- `catspace/engine/orchestrator.py` — Ray single-flight probe memoization/coalescing. **live**
- `catspace/uci.py`, `experiments/uci_engine.py` — UCI wrapper for arena/gauntlet play. **live**

## Evaluation & statistics

- `catspace/stats.py` — `paired_nll_ci` (player-clustered bootstrap; the standard lift CI),
  e-value/anytime-valid helpers. **live**
- `catspace/diagnostics.py` — `eff_rank` (the collapse gate) + board instruments. **live**
- A/B harnesses: `playout_ab.py` (paired playouts), `move_ab.py`, `ab_test.py` (anytime-valid),
  `arena_real.py`, `gauntlet.sh` (fastchess SPRT), `experiment_report.py` +
  `catspace/audit.py` (Stockfish-leakage gate). **live**
- `experiments/play_vs_maia.py`, `playout_ab.py --search expectimax` — play-level checks vs the
  Maia ladder. **live**

## Training infrastructure

- `catspace/train/scaffold.py` — `standard_train`: MLflow (via `catspace/tracking.py` no-fail
  wrapper) + checkpoint ladders with provenance + health gates + Ray **Tune** sweeps (Ray Train
  is broken on py3.14 — use Tune). **live**
- `experiments/losses.py` — canonical unit-tested loss terms; no loss trains without a passing
  test here. **live**
- `experiments/m2b_cache.py` — the crash-safe resumable shard pattern (atomic temp-then-rename,
  disk pre-flight, skip-completed-shards resume) for every expensive precompute. **live**

## Data

- Records: `data/records/player_games_rapid` (6k individual + 40k provisional players,
  single-TC rapid, name-masked ids); full-month 2019-01 identity records (19.35M games) DVC'd.
- Shards: `data/shards/lichess_db_standard_rated_2019-01.full/` (~10 GB) — ingested once,
  reuse, never re-ingest.
- Derived: `data/derived/m2b/*` (feature caches), `data/derived/reach/*`,
  `transition_data_labeled.npz` (M2a SF-committor labels). All DVC-pointered.

## Legacy (pre-rebuild; kept for provenance, not maintained)

The FB two-tower field, sharpness/competence maps, two-horizon heads, plan-persistence policies,
self-play curriculum, play-atlas viz server: `catspace/nn/fb.py`, `catspace/competence.py`,
`catspace/two_field.py`, `catspace/chain.py`, `experiments/train_lichess_fb.py`,
`experiments/viz/` and friends — mapped in `docs/archive/COMPONENTS.md` and RUNBOOK §archive.
A code-level prune of `catspace/`'s unused modules is a scoped follow-up (needs an import/usage
audit), not part of the doc unification.
