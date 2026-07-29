# TESTING — how claims are made, verified, and retracted

*Unified 2026-07-28 from TRAINING_STANDARDS.md + PROCESS.md (both in `docs/archive/`). Every
rule exists because we were burned without it — the scar is cited. Deviations are deliberate
reversals: say so in JOURNAL.md.*

## 1. Claims discipline

- **No number without a printed VERDICT.** No metric enters JOURNAL.md, a commit message, or a
  doc unless a repo script printed it; figures come from repo scripts, not heredocs. *(Scar: the
  +0.86 retraction.)*
- **Grade every claim** — proven / provable / plausible; literature claims carry verification
  grades ([V]/[Q]/[K], see THESIS §6). No sycophancy; retract loose claims explicitly and loudly.
- **Controls decide.** A raw effect needs its control before it's a claim (the in-stratum
  lesson: raw cohesion said "patterns cluster"; the controlled number said "phase shells").
- **Diagnose before concluding.** Never call a plateau "fundamental" before ruling out the loss
  (unit-test it) and the data (inspect the target distribution). *(Scar: margin_ranking with
  y=sign(0)=0 on 39% tied pairs — a bug misread as a wall.)*
- **Define identifications.** "X is a Y" must state Y's criteria and verify X meets them.
- **Search when stuck** >15 min on something that should work: stop tuning, read the paper /
  reference implementation. *(Found the IQE direction bug.)*
- **Exploitation claims come only from realized game outcomes** — opponent-model search values
  are structurally inflated above minimax (PrOM Thm 3); a "certified" probability requires
  sound stopping rules, not vanilla value-iteration convergence.

## 2. Training standards (every run)

1. **Checkpoint ladders**: `<out>_step{N}.pt` at intervals + rolling latest; never only-final.
   *(Scar: train_geometry_l1 overwrote one file.)*
2. **One input format, the richest available**, for every model flavor; checkpoints record their
   convention. *(Scar: zeroed planes made two fields numerically incomparable exactly when we
   needed their difference.)*
3. **Never overwrite; every artifact carries provenance** (full args + git commit embedded in
   the payload; append-only logs; distinct `--out` per run). *(Scar: cert_base_full.pt
   archaeology.)*
4. **Don't hand-roll infra**: MLflow via `catspace/tracking.py` (no-fail wrapper), scaffold via
   `catspace/train/scaffold.py` (ladders + gates + Ray Tune; Ray Train is broken on py3.14).
5. **Smoke before full** — short run first: trains, saves, loads, gates move. Then launch long.
6. **Health gates from the standard battery**, logged not eyeballed: effective rank
   (bootstrapped; collapse is a first-class failure mode — cure is repulsion, not width),
   d_step/d_rand sharpness, asymmetry. *(Scar: the field collapsed to ~1–6 of 512 dims twice.)*
7. **Fixed eval sets, fixed seeds** — never re-sample eval data per run; seeds in stored args.
8. **Watch long runs early and often** — first check ~1 min, then ~5-min cadence; watchdog on
   output growth (CPU busy ≠ progress). *(Scar: the 80-min silent MCTS hang.)* Kill orphaned
   workers before long runs; watch swap. *(Scar: 2026-07-27 swap-thrash stall.)*
9. **Don't hand-code concepts into play** — exact concept computations are diagnostics and
   data-generation labels only.
10. **Don't overload one representation with competing objectives** — separate heads for
    separate quantities; ordering losses over regression where possible. *(Scar: DTM-into-d
    crushed rank 15 → ~1.)*
11. **One disk-heavy job at a time.** *(Scar: the 2026-07-20 I/O starvation.)*
12. **Bundle well-justified changes** (no-one-lever rule) and record the bundle's contents;
    **rejections are conditional** on the field version — re-test the shelf after promotions.
13. **A smoke must smoke the CLAIM**, not just the machinery: every run's smoke includes a
    miniature of the acceptance instrument, which must be **written before launch**
    (pre-registered). *(Scar: the 25k veto-gate miss — AUC 0.37 was visible at any checkpoint;
    the measuring script was written 5h into the run.)*
14. **Audit data balance before training on a contrast** — support overlap on the probe region
    is the precondition for the comparison to mean anything. *(Scar: 0.13 phase overlap.)*
15. **Materialize expensive derived labels; cache expensive probes** (tablebase probes go
    through `catspace/tb.py`'s sqlite cache — `journal_mode=DELETE`, never WAL).
16. **DVC-track every dataset** (pointer in git, bytes outside); **optimize before long runs**
    (parallelize; remove unnecessary compute; fuse passes); **always run latest** (kill stale
    runs on engine updates — resume makes it cheap).

## 3. Acceptance instruments (the pattern)

Every trained artifact ships with a pre-registered evaluation, in the training script, printed
as VERDICT lines. The standard battery, by claim type:

- **Lift claims**: paired per-position NLL/BCE vs the ablated baseline, bootstrap CI
  **clustered by player** (`catspace.stats.paired_nll_ci` — resample players, never positions),
  plus the **placebo**: the wrong-entity control matched on rating (wrong-z within ±100 Elo;
  for the field's opponent slot, permute ẑ_opp trajectories at matched n_obs). A lift that
  survives the ablation but not the placebo is capacity, not signal.
- **Probability heads**: reliability bins (count-weighted ECE — max-gap alone is dominated by
  empty bins) on held-out AND out-of-population splits (the z=0 fallback must calibrate too).
- **Time/regression heads**: MAE vs the trivial baseline (global median).
- **Representations**: bootstrapped `eff_rank` (≥3 draws).
- **Play claims**: SPRT / anytime-valid e-values, n pre-registered, node budgets recorded,
  PGNs kept, MLflow-logged; Stockfish-leakage audit (`catspace/audit.py`) on any candidate that
  could have seen the reference's answers.

## 4. Test suites & self-tests

- `pytest tests/ -q` — the suite (~4.5 min).
- `python experiments/losses.py` — canonical loss terms with executable invariant tests; **no
  new loss enters a run without a passing test here, and terms are imported, never
  re-implemented.**
- `python experiments/endgame_handover.py` — tablebase handover self-test.
- Module self-tests: `catspace/style/estimator.py` (6/6), `catspace/planner/optionality.py`
  (16/16), `catspace/train/scaffold.py`.

## 5. Scope note

The rigor budget applies to OUR novel claims (exploitation dividend, planner effects, field
quality) — not to re-validating the community's solved rankings (MILESTONES decision 8:
adopt-before-build; trunk choice by fiat).
