# memory approaches

Registry for the **memory** component: `query(...)` / `store(...)`.

Every directory under `approaches/` must have an entry here, and every entry must have a
directory — `scripts/check_approaches.py` enforces both directions. Schema is defined in
`repo_structure.md` § "approaches.md schema".

`status` is one of `active`, `parked`, `superseded-by:<name>`.

---

## vector_store_retrieval

- **folder** — `approaches/vector_store_retrieval/`
- **status** — active
- **hypothesis** — Mate/win/draw are *surfaces*, not poles, so distance-to-goal is better
  estimated non-parametrically by retrieving the nearest seen positions and their outcomes
  than by a single learned scalar. Qdrant makes the banks persistent and shareable.
- **definition of done** — Composed retrieval distance beats the parametric value head on
  held-out distance-to-outcome, and the index answers within the per-move search budget.
- **results** — JOURNAL.md (position memory / non-parametric distance, 2026-07-19)
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## checkpoint_trap_bank

- **folder** — `approaches/checkpoint_trap_bank/`
- **status** — active
- **hypothesis** — An embedding bank of mined trap *contexts*, each linked to the checkpoint
  position its game actually sprang, lets the engine recognize a trap it has seen before and
  propose it as a plan.
- **definition of done** — Trap-move proposal rate and confirmed-trap rate improve over the
  no-memory baseline on the smoke harness.
- **results** — JOURNAL.md: switching CheckpointBank to the JEPA T1 encoder moved confirm rate
  34% -> 64% and trap-sourced 57% -> 74% on the v0 smoke harness.
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## experience_store

- **folder** — `approaches/experience_store/`
- **status** — active
- **hypothesis** — Games played, positions searched and when they were added belong in one
  persistence layer with real provenance, not in ad hoc per-script files.
- **definition of done** — Every generator and every consumer of engine experience reads and
  writes through this store; a run can be reconstructed from it.
- **results** — JOURNAL.md (experience store, 2026-07-25)
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## goal_region_bank

- **folder** — `approaches/goal_region_bank/`
- **status** — active
- **hypothesis** — A goal is a *region*: scoring against a bank of exemplar B-embeddings by
  best-over-bank beats collapsing the region to one centroid vector.
- **definition of done** — Best-over-bank scoring beats the centroid baseline on region-reach
  measurement.
- **results** — JOURNAL.md, 2026-07-13 measurement that motivated the change
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## fast_field_knn

- **folder** — `approaches/fast_field_knn/`
- **status** — active
- **hypothesis** — A fast, in-memory, per-move-updatable evidence store over embedding space
  (the FAST field) can re-price the SLOW trained embedding mid-game — the two-timescale design.
- **definition of done** — Re-priced reach beats raw slow-field reach mid-game at equal search
  budget, via the `search:puct_mcts` `repricing.py` hook.
- **notes** — Supplies the `MemoryField` that `search/puct_mcts/src/repricing.py` consumes; it
  is not yet selected by any end-to-end config in `catspace/approaches.md`.
- **results** — —
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## plan_ledger

- **folder** — `approaches/plan_ledger/`
- **status** — active
- **hypothesis** — Live engine state and intent (which plans are open, why one was blocked)
  belongs in a single in-process sqlite ledger, strictly separate from training data.
- **definition of done** — The planner's wake/block decisions are reconstructable from the
  ledger alone, and no dataset ever reads from it.
- **results** — M4 work item 1, 2026-07-29
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh

## competence_map

- **folder** — `approaches/competence_map/`
- **status** — active
- **hypothesis** — Kaveh's Method 2: measuring engine performance per region of embedding
  space tells us where the engine is weak, and that map is itself a memory the planner can
  consult.
- **definition of done** — The map separates regions by measured performance beyond noise, and
  a planner reading it avoids its own weak regions.
- **results** — —
- **added** — 2026-07-10 · **owner** — Kaveh Shoorideh
