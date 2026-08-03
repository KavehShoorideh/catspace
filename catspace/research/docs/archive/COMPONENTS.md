# Components — what each piece does

A map of the moving parts, since the system is growing. Grouped by role. See
GLOSSARY.md for term definitions, ARCHITECTURE.md for the layer diagram, and
JOURNAL.md for why each exists.

## Representation (the learned embedding)

- **`catspace/nn/fb.py` — `TorchFB`**: the Forward–Backward reachability embedding.
  `F(s)` (state, omega-conditioned), `B(g)` (goal), and `score(F,B)` = how
  reachable g is from s. Config flags stack:
  - `quasimetric=True`: `score = f@W@g − d(f,g)` with `d` a real triangle-
    inequality metric (planning distance); the residual `r=f@W@g` is unconstrained.
  - `two_horizon=True`: adds a NEAR head (short-gap, cosine) beside the FAR head
    (long-gap, quasimetric). Baseline for the depth-axis idea; superseded by the
    sharpness reframe but kept.
  - `distributional=True`: adds a CATEGORICAL head over ply-gap distance bins
    (`dist_logits`/`dist_entropy`); the quasimetric `d` stays the distance, the
    entropy is an auxiliary uncertainty readout. (Option B; its entropy did NOT
    detect sharpness — see JOURNAL.)
- **`catspace/nn/encoder.py` — `BoardEncoder`**: the shared conv trunk feeding the heads.

## Readout / search (turning the embedding into moves)

- **`FBBoardPolicy`** (policy_fb.py): greedy depth-1/2 readout (the original).
- **`FBSearchPolicy`**: beam-limited minimax over `F@z`, node-budget (`max_nodes`)
  based. The workhorse. Also exposes:
  - `.reliability(board)` — **Method 1 sharpness sensor**: shallow-vs-deep
    reachability-ranking disagreement ("does thinking harder change my mind here").
    Self-referential, reachability-native. 0 = quiet, 1 = sharp/unreliable.
- **`FBPlanPolicy`**: plan-persistence (commit to a subgoal, replan on drop/stall).
  Tested null vs plain search so far; kept.
- **`FBTwoHorizonPolicy`**: readout for a two-horizon checkpoint (far/near modes).
- **`FBAdaptiveSearchPolicy`** (NEW): reliability-gated search. Combines Method 1
  (`.reliability`, exact) and Method 2 (competence map, cheap). EITHER method sharp
  → extra search; BOTH sharp → deepen until the top move stabilizes ("certainty");
  quiet → base budget. The fix for "uniform more-search doesn't help" — search
  more only where deeper search changes the decision.
- **`soft_min_bank` / goal banks**: region-goal readout (nearest-exemplar). Rejected
  at play; kept as an instrument.

## Sharpness / competence (where to think harder)

- **`FBSearchPolicy.reliability`** — Method 1 (see above): exact per-position, needs
  the deep search.
- **`catspace/competence.py` — `CompetenceMap`** — Method 2: a kNN reliability field
  over embedding space. Predicts unreliability from `F(s)` alone (cheap, no deep
  search), generalizes ("this region has been sharp for me"). Built offline by
  `experiments/build_competence_map.py` (stamps a corpus with embedding +
  Method-1 reliability; reports held-out generalization).
- **`catspace/goal_bank.py`**: harvest/embed mate exemplars for region goals.

## Goals

- **`zgoals`** (in each checkpoint): `MATE_W`/`MATE_B` (far centroid),
  `MATE_*_NEAR` (near centroid), `MATE_DIFF` (outcome direction). Built in
  `train_lichess_fb.embed_zgoals`.

## Data & training

- **`catspace/data/lichess.py`, `data/shards.py`**: stream Lichess → packed-bitboard
  shards; `LichessPairSource` samples (anchor, goal) pairs; `MixedPairSource` blends
  human + self-play.
- **`experiments/train_lichess_fb.py`**: the training loop. Loss terms (all optional,
  composable): cosine-InfoNCE (base), ply-gap calibration, asymmetry margin,
  two-horizon stratified, categorical distributional. `--ckpt-every` saves a
  step-tagged LADDER for early stopping.
- **`experiments/selfplay_generate.py`**: self-play games → Lichess-schema shards
  (moves + result only, never engine evals). `--endgame-start-frac` curriculum.
  (Stage 2 of the loop — logging the search TREE, not just the line — is TODO.)

## Evaluation / instruments

- **`experiments/experiment_report.py`** (+ `experiment_leaderboard.py`): arena vs
  Stockfish with the hard leakage-audit gate; one JSON record per run.
- **`experiments/acpl_probe.py`**: average-centipawn-loss blunder metric vs Stockfish.
- **`experiments/krrkbp_arena.py`** + `catspace/diagnostic_krrkbp.py`: the
  tablebase-verified K+R+R vs K+B+P endgame conversion benchmark (paired stats).
- **`experiments/qm_fitness_probe.py`**: quasimetric health suite (Syzygy calibration,
  horizon retrieval, asymmetry, triangle violation, degeneracy).
- **`experiments/sharpness_bench.py`**: tablebase sharpness diagnostic. **RETIRED as
  arbiter** (sharpness is an invented concept — validate by PLAY, not by a label);
  kept for position sampling + structural signals + endgame sanity only.
- **`catspace/audit.py`**: the Stockfish-leakage gate (static + provenance).

## Design docs

- **`UNCERTAINTY_DESIGN.md`**: the sharpness reframe (depth→uncertainty→self-
  reliability) and the two-method system + the self-play distillation loop.
- **`TWO_HORIZON_DESIGN.md`**: the (superseded-axis) two-horizon spec.
- **`GLOSSARY.md`**, **`ARCHITECTURE.md`**, **`JOURNAL.md`**.
