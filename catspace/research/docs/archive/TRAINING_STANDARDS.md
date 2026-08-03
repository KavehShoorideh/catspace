# TRAINING_STANDARDS.md — do's and don'ts for ALL trainings (Kaveh 2026-07-23)

Standing engineering discipline; applies to every training run in this repo. Each rule exists
because we were burned without it (the scar is cited). Deviations are deliberate reversals —
say so in JOURNAL.md.

## DO

1. **Checkpoint everything, as ladders.** Every run saves `<out>_step{N}.pt` at regular
   intervals PLUS a rolling latest. Never only-final. *(Scar: train_geometry_l1 silently
   overwrote one file — no mid-run readings, no post-hoc best-step selection. Fixed 2026-07-23.
   Payoff same day: the step-5000 early in-stratum reading on the 30k human-field run.)*

2. **One input format, the richest available, for every model flavor.** All 20 feature planes
   (12 pieces + stm + 4 castling + ep + halfmove + repetition), everywhere. No per-flavor
   zeroing/trimming. *(Scar: the geometry path zeroed planes 18/19 while the lichess path kept
   them -- the two fields became numerically incomparable exactly when we needed to subtract
   them (the cooperative-vs-human veto gap). This REVERSES the old BOARD_ONLY convention
   [DECISIONS sec 1] from the next run onward; checkpoints record which convention they were
   trained under in their stored args.)*

3. **Never overwrite past results; every artifact carries its run metadata.** Checkpoints embed
   the full argparse namespace (+ what they resumed from) in their payload (`provenance` /
   `args`); logs are append-only under `artifacts/experiments/`; distinct runs use distinct
   `--out` names. *(Scar: cert_base_full.pt became unloadable and its training args had to be
   archaeologized from JOURNAL prose.)*

4. **Use existing tooling for training ops — don't hand-roll.** MLflow (local `./mlruns`,
   `mlflow ui` to browse) via the thin no-fail wrapper `catspace/tracking.py`: params = full
   args, metrics = per-step losses/gates, tags = final VERDICTs. Same spirit as the standing
   import-don't-reinvent rule (PyTorch, dictionary_learning, captum...). Tracking must never
   kill a run — the wrapper degrades to no-op.

5. **Smoke run before full run.** Short run first, verify it trains + saves + loads + the
   gates move, THEN launch long. *(Standing rule; the 500-step lichess_sharp smoke caught
   nothing this time — cheap insurance either way.)*

6. **Health gates on every run, from the standard battery.** Effective rank (collapse),
   d_step vs d_rand ratio (metric sharpness), asymmetry (irreversibility direction) — logged,
   not eyeballed. *(Scar: the field collapsed to ~1-6 dims of 512 TWICE before these became
   routine.)*

7. **Fixed eval sets, fixed seeds.** Evaluate on the frozen sets (krrkbp_test_n200,
   dtm_endgame holdout split by stored seed) — never re-sample eval data per run. Seeds live
   in the stored args.

8. **Watch long runs early and often.** First check at ~1 min, then ~5-min cadence; watchdog
   on log growth (CPU busy != progress). Runs > 1 h get a Monitor on failure signatures.

## DON'T

9. **Don't hand-code concepts into play.** Exact concept computations (escape volume, threat
   flags) are diagnostic instruments and data-generation labels ONLY (DECISIONS sec 4).

10. **Don't overload one representation with competing objectives.** The DTM-into-d lesson:
    cramming a scalar regression into the quasimetric crushed rank 15 -> ~1 (DECISIONS sec 7).
    Separate heads for separate quantities; ordering losses over regression where possible.

11. **Don't quote numbers that a script didn't print.** VERDICT lines only, into JOURNAL —
    and claims get the controls (the in-stratum lesson: raw cohesion said "patterns cluster",
    the controlled number said "phase shells"). Retract loudly when a control kills a claim.

12. **Don't run more than one disk-heavy job at a time** (shard streaming, tablebase probing,
    macrostate builds). *(Scar: the 2026-07-20 I/O starvation destabilized the laptop.)*

13. **Don't serialize well-justified changes into N runs — bundle and record** (the
    no-one-lever rule), but keep every bundle's contents listed in JOURNAL so rejections stay
    interpretable (conditional-rejections rule: re-test shelved mechanisms after field
    promotions).

## Amendments (2026-07-23, post the 25k veto-gate miss)

14. **A smoke test must smoke the CLAIM, not just the machinery.** The multichannel smoke
    verified train/save/load/collapse gates and passed — while the run's actual objective (a
    veto-reading channel gap) was structurally impossible from its data. Every run's smoke =
    mechanical checks + a MINIATURE of the acceptance instrument (even at silly-small scale,
    the SIGN of the effect shows) — which requires the acceptance instrument to be WRITTEN
    BEFORE LAUNCH (pre-registered), not after. *(Scar: measure_veto_channels.py was written
    5h into the run; AUC 0.37 anti-correlation was visible at any checkpoint.)*

15. **Audit data BALANCE before training on a contrast between datasets.** Any objective that
    compares/differences two data sources (channels, cohorts, contrast branches) gets a static
    distribution audit first — support overlap on the probe region is the precondition for the
    comparison to mean anything. *(Scar: regime-1 vs regime-2 phase overlap was 0.13; the
    8-second audit_channel_balance.py check would have vetoed the launch config. The balanced
    replacement overlaps 0.95.)*

16. **Materialize expensive derived labels; cache expensive probes.** Exact labels (forceability
    DFS, DTM rollouts, deniedness sets) are results — save them next to the VERDICT, don't
    discard after aggregating. Tablebase probes now go through the persistent sqlite cache in
    catspace/tb.py (the 2026-07-20 'cache tablebase probes' lesson, finally implemented).

## DATA & FRAMEWORKS (Kaveh 2026-07-26)

8. **DVC-track every dataset.** Every generated dataset (`data/records/`, `data/derived/`,
   `data/shards/`) is versioned with `dvc add` -> commit the `.dvc` pointer; never commit the raw
   bytes (gitignored `/data/**`). Reproducible + shareable. *(Reuse, don't re-ingest — see the
   lichess `.dvc` set.)*

9. **Scaffold training with frameworks — don't hand-roll infra.** Use `catspace/train/scaffold.py`:
   MLflow tracking + checkpoint ladders w/ provenance + health gates + **Ray Tune** for HP sweeps /
   parallel trials. Parallelize to the frameworks' ability (Tune parallel trials; DataLoader
   `num_workers`; process-pool data gen). *Verified on this py3.14+MPS box:* Ray **Tune** works;
   Ray **Train**'s distributed-worker abstraction does NOT (cloudpickle/controller-actor fails under
   py3.14) — scaffold with Tune, not Train (revisit when Ray supports 3.14). Tune trainables MUST be
   top-level functions that import inside their body (cloudpickle serializes referenced globals by
   value).
