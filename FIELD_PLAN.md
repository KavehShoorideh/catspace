# FIELD_PLAN — certainty geometry + two-timescale, two-perspective fields
*2026-07-14. The rigorous build plan (stages, gates, instruments) for the design
agreed in JOURNAL 2026-07-14: d = plies + λ(−ln P), slow/fast fields, mine/theirs
perspectives, learned (not hand-coded) fallibility. Every stage has: deliverable,
tests WITH statistics, instruments, and a GATE that must pass before the next
stage may depend on it.*

## Non-negotiable methodology (lessons already paid for)
- **No point estimates.** Every comparison: paired, bootstrap CI (or e-process for
  sequential looks); "wins" only when CI excludes zero; sweeps are *exploratory by
  declaration*; the selected config gets ONE pre-registered confirmatory run on an
  untouched frozen set.
- **Regime-aware play evals.** Report BOTH 200n (shallow — the thesis metric) and
  800n (saturation — embedding-quality metric). A change can help one and not the other.
- **Held-out discipline registry.** `artifacts/experiments/data_registry.json`
  lists every FEN set and its role (train / eval / confirmatory-frozen). Leakage
  is auditable, not remembered.
- **Failure-aware monitors.** Every background watcher triggers on success OR
  failure OR process-exit. Every run ends with a machine-readable `VERDICT` line.
- **Field-health panel on every new checkpoint** (regression guard, cheap):
  VAL_TOP1 (global retrieval, human holdout) · toy Spearman-vs-certainty ·
  retrieval MAE · playout 200n & 800n · qm_fitness (calibration/asymmetry drift).
  Any global metric regressing beyond its CI = stop, diagnose.
- Time every run; journal + commit at every gate.

## Stage 0 — radial certainty distillation, toy (RUNNING)
**Deliverable:** `certainty_full.pt` (6k steps; tb+ε scaffold table).
**Tests:** (a) held-out Spearman(d, plies+λ(−lnP̂)) w/ CI — short run already flipped
−0.099→+0.170, disjoint CIs; (b) **MONEY:** paired 200n playout vs incumbent on
held-out `test_n200`, CI-excluding-zero; (c) confirmatory: fresh never-touched
tablebase-verified starts, one run, pre-registered.
**Instruments:** gates.jsonl; field-health panel.
**GATE 0:** money CI > 0 AND confirmatory agrees AND no global regression.
*(If money fails but Spearman holds: geometry improved, readout didn't — diagnose
with the drill-down viewer before adding anything on top.)*

## Stage 1 — de-scaffold: own-play estimator
Swap tb+ε for the (now-tuned) model's own ε-noised play in `certainty_rollouts`.
**Tests:** P̂ distribution has usable spread (frac{0<P̂<1} ≥ 0.3, ESS per state ≥ 4);
then re-run all Stage-0 gates with the own-play table.
**Instruments:** P̂-distribution panel per table (mean, frac 0/1, n histogram);
**drift meter** between successive tables (MAE on shared states) = measured
"landscape shift" per generation — the non-stationarity monitor.
**GATE 1:** gates hold with zero oracle involvement.

## Stage 2 — the closed loop
Iterate: own rollouts → table → fast field → distill → stronger policy → rollouts.
**Tests:** per-round paired playout vs round-0 (200n), anytime-valid e-process
(we peek every round — sequential statistics, not repeated t-tests); Spearman gate
per round; P̂ mean should RISE (policy converts more).
**Stopping rule:** 2 consecutive rounds without CI-improvement = plateau; stop, analyze.
**Instruments:** round-trajectory jsonl + plot; frozen eval starts across all rounds.
**GATE 2:** ≥2 rounds of CI-real improvement (the loop actually compounds).

## Stage 3 — two-perspective runtime (mine / theirs)
Dual MemoryFields (my goal / their goal) over the one slow embedding; planner
objective mixes d_mine vs d_theirs (steer into their high-p_var, my low-p_var).
**Tests:** paired playout, planner-with-opponent-field vs without, matched compute,
vs a FALLIBLE (ε-noised) defender — vs an optimal defender an opponent model is
useless by construction, so the test opponent must have exploitable variance.
**Instruments:** two-field trace viewer: per-move (d_mine, d_theirs, p_var_mine,
p_var_theirs) along played games (extends the decision viewer).
**GATE 3:** opponent-field planner wins CI-real at matched compute.

## Stage 4 — learned fallibility: population prior (no hand-coding)
Measure blunder-rate vs (Elo bin, clock bucket) directly from the Lichess corpus
(move-quality proxy: eval drop where eval_cp exists — offline analysis only, never
a training label; audit gate applies). Fit the prior table; ε in rollouts comes
from it, not from a hand-coded curve.
**Tests:** calibration on held-out games — reliability diagram, ECE w/ CI;
per-Elo-band curves must differ (the signal exists) or the conditioning is dropped.
**GATE 4:** ECE ≤ threshold on held-out; prior beats global-constant ε on
log-likelihood of held-out blunders, CI-real.

## Stage 5 — per-opponent online update (calm-under-pressure)
Opponent's ε-by-clock as per-game Bayesian parameters, prior = Stage 4, updated
from observed move quality; re-prices their field's −ln P̂ at query time.
**Tests (toy, parameter recovery):** simulated opponents with KNOWN ε profiles
(calm / panicky / average) — does the posterior recover the profile within
coverage? Then: regret test — value lost acting on wrong-vs-recovered model.
**Instruments:** per-game posterior trace; flagging-plan EV logged before/after
opponent-model updates (did the update change the plan when it should?).
**GATE 5:** correct profile recovery + CI-real regret reduction vs prior-only.

## Explicitly deferred (documented, not forgotten)
Clock-ω plumbing (dual clocks through data/features/model — zero-init migration
designed in JOURNAL); pairwise (A→B) certainty targets from the same rollout data;
tactic-potential payloads in MemoryField; concept-axes expansion; from-scratch
retrain with certainty in the base objective (only if distill-on-incumbent
saturates at a gate).
