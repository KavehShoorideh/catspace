# Implementation plan — stratified adversarial quasimetric

*Companion to ARCHITECTURE_STRATIFIED.md. Phased, with the standing constraints baked in:
checkpointing + parallelization on long runs, real tests, import-don't-hand-roll, viz that
BOTH of us can watch, and ask-don't-grind on genuine forks.*

## Cross-cutting principles (apply to every phase)

- **Checkpointing.** No long process is all-or-nothing. Data gen writes **per-material/per-chunk
  shards incrementally** (+ a `--merge-only` / skip-existing resume); training checkpoints every N
  steps AND at every stratum boundary (`_le3…_le7`); every artifact records provenance.
- **Parallelization.** Data gen is embarrassingly parallel (ProcessPoolExecutor over
  material×worker chunks — already in place). Training is GPU compute-bound on one MPS device (not
  multiprocess-parallelizable — established); we optimize FLOPs (batch size, fused passes) instead.
- **Tests.** `pytest` for every new module: label correctness vs `python-chess`, quasimetric
  axioms (identity, triangle inequality on the IQE head), strata invariants (captures reduce piece
  count; reverse is never an edge), reachability-mask logic, encode round-trips. CI-runnable, no
  GPU required for the unit layer.
- **Viz (measure-but-visible).** Every metric that gates a decision also renders: live training
  curves (loss terms, d_pos, cap-asym, per-material spearman) written to an auto-refreshing HTML;
  the UMAP atlas after each run; a small dashboard served by the existing `play_server`. If it's a
  VERDICT number, there's a picture of it.
- **Import, don't hand-roll.** See the table below; validate any swap for numerical equivalence
  before deleting our version.
- **Ask, don't grind.** On a genuine fork (>~15 min stuck, or a design choice that changes the
  plan), stop and use the phone-push question tool rather than guessing.

## Import decisions (don't hand-roll)

| Component | Decision | Notes |
|---|---|---|
| IQE / quasimetric head | **Adopt `torch-quasimetric` (torchqmet)** — Wang's official pkg | We hand-roll `catspace/nn/iqe.py`. Migrate the head to torchqmet, validate byte-equivalence on a batch, keep our F/B tower + training. Do it **after** M1 validates the geometry (don't refactor the foundation mid-proof). |
| QRL loss | **Reference `quasimetric-rl`**, keep our chess-integrated loop | Their repo assumes d4rl/GCRL envs; we borrow the loss form, not the harness. |
| Tablebase / move-gen / rules | **`python-chess`** (already used) | Syzygy WDL/DTZ probing, legal moves, `is_irreversible`. Never hand-roll chess rules. |
| Adversarial search (minimax/MCTS) | **Evaluate `OpenSpiel`** vs our `planner/search.py` | OpenSpiel has battle-tested alpha-beta + MCTS for zero-sum games. Our search integrates the field heuristic + tablebase leaves. Decide at M4 (see open decisions). |
| Landmark graph search | **`networkx`** (Dijkstra / Floyd–Warshall) | Don't hand-roll shortest paths for the keypoint graph. |
| Dim-reduction / clustering | **`umap-learn`, `scikit-learn`** (already used) | UMAP, kNN, silhouette. |
| Second-domain benchmark (M5) | **`OGBench`** (Park et al.) | Offline GCRL envs with a monotone/bottleneck structure to demo generality. |
| Plots / dashboard | **matplotlib + the existing viz/ + play_server** | Reuse `experiments/viz/*` and `catspace/viz/*`. |

---

## M1 — Lean stratified data + L1 geometry + UMAP (the foundation)
*Restart of the run I killed, lean and checkpointed. This validates the load-bearing claim: the
cooperative geometry has clean piece-count strata, one-way captures, and mate-pole order.*
- **Steps.** (1) Add incremental shard-checkpointing to `gen_stratified_perfect.py`; regenerate
  **≤6 only** (fast, exact). (2) Train L1 with the four policy-independent terms + the per-stratum
  curriculum checkpoints. (3) UMAP atlas + quantitative structure.
- **Deliverables.** `data/derived/stratified_perfect.npz` (≤6), `iqe_stratified.pt` (+`_le*`),
  `stratified_umap.png/html`.
- **Tests.** label correctness vs python-chess on a sample; triangle inequality on the IQE head;
  capture edges reduce piece count; reachability-mask unit tests.
- **Viz.** 4-panel UMAP (strata / material / mate-pole / WDL) with 7p→6p arrows; live training-curve HTML.
- **VERDICT gates.** per-material spearman(d,DTM) > 0; capture one-way ≥ (target) ×; piece-count
  silhouette > 0. If any fails → investigate, don't tune blindly.
- **Checkpoint/parallel.** gen: 8-worker + incremental shards. train: MPS, batch 128, ckpt every 400.

## M2 — L2 adversarial heads + adversarial-distance composition validation (the novelty)
- **Steps.** (1) Train the **remoteness/DTM-to-region** categorical head + the **committor** head on
  frozen L1, on exact tablebase labels. (2) **Validate the adversarial-quasimetric claims on
  ground truth:** does DTM-to-region satisfy the region-triangle inequality (measure violations)?
  Is `−ln P(win)` approximately quasimetric (measure triangle-inequality residuals)? When does
  `L1 ⊕ L2` recover the true minimax value (gap vs tablebase)?
- **Deliverables.** `l2_remoteness.pt`, `l2_committor.pt`, an **adversarial-distance report**
  (composition residual histograms, calibration).
- **Tests.** committor is a martingale on optimal transitions (Doob check); head accuracy on held-out.
- **Viz.** triangle-inequality-residual histograms; committor surface over the UMAP; near-mate
  guidance test (mate-within-5) as a picture.
- **This is the paper's core novelty experiment** — it's low-compute and uses ground truth.

## M3 — ε_n extrapolation study + past-frontier UMAP (the tractability verdict)
- **Steps.** Extend data to 7p (cheap config: capture-extension / retrograde-one-ply for exact
  capture-adjacent labels). Measure **ε_n vs stratum**: field-vs-tablebase where truth exists,
  spot-check search above. Does per-stratum training damp inherited error or does it compound?
- **Deliverables.** `epsilon_n_vs_stratum.png` (the tractability answer), extended UMAP with 7p.
- **Tests.** 7p capture edges land in solved 6p; retrograde labels match forward search.
- **Viz.** ε_n curve; the "does it climb" picture. **This decides how far the method reaches.**

## M4 — Inference recursion + planner
- **Steps.** Vector DB of labeled L1 embeddings (reuse `position_memory`); OOD gate =
  nearest-labeled-neighbor distance; recursive uncertainty-gated minimax descent (leaf = retrieval/L2,
  backup = minimax, deepen on retrieval-vs-shallow-search disagreement). Landmark graph via networkx.
- **Deliverables.** `analyze(position)` end-to-end on out-of-sample high-piece-count positions; a
  navigable atlas showing the descent path + subgoals.
- **Tests.** on known tablebase positions the recursion returns the exact value; wormhole/false-shortcut guards.
- **Viz.** the descent-through-strata animation over the UMAP (arrows = captures/subgoals) — the
  "watch it plan" view you asked for.
- **Open decision at this phase:** OpenSpiel vs our search (see below).

## M5 — Generalization (stretch): second non-chess domain + L3
- **Steps.** Instantiate the *same* pluggable interface (verifier + stratification) on an
  OGBench-style toy with a monotone/bottleneck coordinate — show the mechanism isn't chess-specific.
  Then L3 human-playability from lichess.
- **Deliverables.** a second-domain extrapolation result; the "it's a mechanism, not a chess trick"
  evidence that de-risks the contribution.

---

## Open decisions (I'll ask via phone-push rather than guess)

1. **Next-step priority after M1 data lands** — geometry-UMAP first, or jump to the adversarial-
   distance validation (the novelty you love)? *(asking now)*
2. **torchqmet migration timing** — after M1 (recommended) or defer entirely while our iqe.py works.
3. **M4 search backend** — adopt OpenSpiel (battle-tested, heavy dep) vs extend our field-integrated
   `planner/search.py`. Decide when we reach M4.
