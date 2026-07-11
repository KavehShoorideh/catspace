# Catspace Glossary

Reference for the embedding and evaluation concepts used throughout the codebase.

---

## Embedding concepts

### Forward–Backward (FB) embedding
The core representation learned via `TorchFB` (catspace/nn/fb.py). Two convolutional board encoders produce:
- **F(s)**: the forward embedding of a position s — "what's reachable from here?" under optimal play.
- **B(g)**: the backward embedding of a goal position g — "how did we get here?" from the goal's perspective.

The key invariant: **reach(s, g) = F(s) · B(g)**, a dot product in shared embedding space. High reach means g is plausibly reachable from s.

The forward encoder is **omega-conditioned** (depends on player strength and time); the backward encoder sees only the board.

### Omega (ω) conditioning
Player-strength and time conditioning applied to the forward encoder. Omega includes:
- **White Elo bin** (one of 10 buckets: 800–1000, 1000–1200, ..., 2000+)
- **Black Elo bin** (same)
- **Clock bucket** (time control: bullet, blitz, rapid, classical)

The reachable cone depends on who's playing — strong players explore different positions than weak players. So F(s|ω) is "positions reachable by players like these."

### Cosine InfoNCE loss
Contrastive self-supervised learning objective (catspace/nn/fb.py, experiments/train_lichess_fb.py). Given a game state s and a later position g from the same game:
- **Positive pair**: (F(s), B(g)) should have high dot product.
- **Negative pairs**: (F(s), B(g')) for g' from other games should have low dot product.
- **Cosine variant**: embeddings are L2-normalized before the dot product, so reach is a cosine similarity [−1, 1]. This is essential on real boards where activation norms shrink with material loss; unnormalized dot products inherit that decline.

Training on Lichess games (seeded sampling across 11M positions; see LichessPairSource).

### Reach (or reach value)
The cosine similarity `F(s) @ B(g)` after L2 normalization, ranging [−1, 1]. Higher reach means positions s and g are more "aligned" in the embedding space — g is more plausibly reachable from s under optimal play for the given ω.

Not to be confused with "reachability" in chess (whether a position is legally reachable); reach is a learned score from the embedding.

### Goal vectors (z-vectors, zgoals)
Aggregate backward embeddings: summary vectors for outcome positions, stored in the checkpoint.

- **z_MATE_W**: mean B over all checkmate positions where white won (result = +1).
- **z_MATE_B**: mean B over all checkmate positions where black won (result = −1).
- **z_MATE_DIFF**: z_MATE_W − z_MATE_B, the outcome *direction*. Used as the objective in planning and for the zero-label eval baseline.

Computed once per shard scan; re-embedded at every periodic checkpoint save (catspace/nn/fb.py).

---

## Eval concepts

### Descriptive head
A 2-layer MLP that reads F(s) and predicts the result of the game (white win / draw / loss) as a 3-way softmax. Trained on actual game outcomes from Lichess. Answers "what actually happens from here among players like these?"

`EvalHead(n_out=3)` in catspace/nn/eval_head.py. Output is converted to expected score via P(W) + 0.5·P(D).

### Normative head
A 2-layer MLP that reads F(s) and predicts the Stockfish eval as a sigmoid. Trained on [%eval] annotations in Lichess games. Answers "what should happen under best play?"

`EvalHead(n_out=1)` in catspace/nn/eval_head.py.

### Divergence (descriptive vs normative)
The difference between what humans do and what the engine says they should do. Large positive divergence = "humans overperform the eval" (trap positions). Large negative divergence = "humans underperform the eval" (missed wins).

Used to identify positions where the human-eval mismatch is largest (JOURNAL.md, top divergent positions list).

### Frozen probe (vs joint fine-tuning)
By default, F is **detached** during eval-head training — the gradient doesn't flow back to the FB model. This treats the pre-trained representation as fixed and only trains the 2-layer linear probe. `--joint` flag allows fine-tuning both (experimental; off by default until the frozen probe's signal is understood).

---

## Ablation & baseline concepts

### DESC_AUC (Descriptive AUC)
Area-under-curve of the descriptive head's expected-score prediction, separating positions from white-won games from positions from white-lost games (on holdout data). Ranges [0, 1]; 0.5 = random; 1.0 = perfect.

Measures: does the frozen F(s) embedding capture enough outcome signal to discriminate won from lost?

**Calibration at 30k steps:** 0.625 (F), 0.596 (B), 0.636 (F⊕B). The F > B flip shows outcome signal lives in forward structure, not static board features.

### NORM_SPEAR (Normative Spearman)
Spearman rank correlation between the normative head's prediction and Stockfish winprob (on annotated holdout positions). Ranges [−1, 1]; 0 = no correlation; 1 = perfect agreement.

Measures: does the frozen F(s) agree with the engine's eval?

**At 30k steps:** 0.482 (F), 0.376 (B), 0.516 (F⊕B).

### BASE_AUC, BASE_SPEAR (zero-label baseline)
The score of F(s) · z_MATE_DIFF *without any probe training* — pure geometry. Reported as:
- BASE_AUC: AUC of the zero-label readout on won/lost separation.
- BASE_SPEAR: Spearman correlation with Stockfish winprob.

**At 30k steps:** 0.598 (AUC), 0.369 (Spear). When BASE approaches the trained probe values, it means the embedding's geometric structure is doing most of the work; the probe adds only marginal refinement.

### Representation ablation (--repr F / B / FB)
Three versions of the eval-head experiment:
- **F**: probe reads F(s) only (the hypothesis under test).
- **B**: probe reads B(s) only (control: if B ≈ F, outcome info is static board features).
- **FB**: probe reads [F(s), B(s)] concatenated (control: if FB > F, F is losing value-relevant info that B keeps).

**Reading at 30k:** F > B emerged (DESC_AUC 0.625 vs 0.596), confirming outcome signal lives in forward structure, not board-only features.

---

## Decomposition concepts

### Reach slope (REACH_SLOPE, DIFF_SLOPE)
Per-game Spearman rank correlation between ply count and reach value, averaged over 200 holdout games. Measures whether reach monotonically increases / decreases through a game.

- **REACH_SLOPE_WON / REACH_SLOPE_LOST**: correlation with reach toward z_MATE_W. At 30k: +0.671 won / +0.587 lost — both rise through games.
- **DIFF_SLOPE_WON / DIFF_SLOPE_LOST**: correlation with reach toward z_MATE_DIFF (the outcome direction). At 30k: +0.174 won / −0.080 lost — correct separation, winners drift positive.

Used to validate that the embedding captures trajectory-level structure (not just static positions).

### Decomposer (meet-in-the-middle waypoint planning)
Recursive algorithm in catspace/planner/decompose.py. For a hop s → g:
1. Score each pool waypoint m by the bottleneck: min(F(s)·B(m), F(m)·z_g).
2. Split at the argmax waypoint if it improves the reach.
3. Recurse on both legs until every leaf is executable or a give-up rule fires.

**Pool** is a set of candidate waypoints (positions from Lichess holdout games, embedded under the planner's own omega).

### Give-up rules (decomposer termination)
Stop splitting when:
- **no_midpoint**: best waypoint's gain ≤ min_gain — the hop is hard, not long; splitting won't help.
- **unlikely_territory**: bottleneck reaches below tau_floor — we're in territory the field considers very unlikely.
- **dry_out**: two consecutive splits each improved by < dry_gain — diminishing returns; we're converging without arriving.
- **budget**: depth cap reached (anytime algorithm: return the tree so far).

### FRAC_IMPROVED (decomposer metric)
Fraction of start positions where the best waypoint beats the direct reach. At 30k: 0.825 — 82.5% of middlegames have a usable stepping-stone.

### MEAN_GAIN (decomposer metric)
Average improvement in bottleneck reach (Δ in cosine units, [0, 1]) when splitting vs. going direct. At 30k: 0.43 — substantial; median start improves by 43 percentage points.

### Arc property
On synthetic unit-circle geometry, the geodesic-midpoint waypoint is exactly the arc middle. On real boards, "arc property" = chosen waypoints sit plausibly between starts and goals in game progression. At 30k: waypoint ply mean 68.3 vs start ply mean 30.2 — genuine endgame stepping-stones.

### tau_exec, tau_floor (decomposer thresholds)
- **tau_exec**: min reach threshold for a hop to be considered "executable." Calibrated as the median reach of positions ≤10 plies before mate in won holdout games. At 30k: 0.2364.
- **tau_floor**: lower bound; positions with reach < floor are "unlikely territory." Calibrated as q10 of all start reaches. At 30k: −0.1762.

---

## Data & training concepts

### Holdout (held-out test set)
Positions from games where `game_id % 50 == 0`. Never seen during training. Used for validation and eval-head training. ~20% of the data; split deterministically by game ID.

### LichessPairSource (data loader)
Streams (s, g) pairs from Lichess shards: a board s and a later position g from the same game, with g sampled geometrically (gamma-distributed ply distance, default gamma=0.98). Covers 11M positions across 12 1GB shards of rated standard games from 2019-01.

### Shard (data storage)
One .npz file with arrays: packed (bitboard + metadata), meta (turn/castling/etc), ply, clock, eval_cp (Stockfish eval), result (±1 or 0), white_elo, black_elo, game_id. Built by catspace/data/shards.py; one shard ≈ 1M positions ≈ 119MB.

### Lichess [%eval] annotations
Optional Stockfish evaluation in the PGN comment, parsed as eval_cp (centipawns). Used to train the normative head. Not every position has an eval; finite annotation rate (~19k / 224k positions in the 1GB holdout).

---

## Experiment & logging

### JOURNAL.md
Running lab notes at the repo root. Every experiment entry: what was done, wall-clock timings, VERDICT lines verbatim from the output, and interpretation. Newest entry last. (See research_workflow_journal_timing memory.)

### Artifacts
Checkpoints, logs, and generated outputs under `artifacts/`:
- `data/derived/lichess_fb.pt`: main trained FB model.
- `data/derived/lichess_fb_step2000.pt`: step-2000 backup for before/after.
- `data/derived/eval_heads.pt`: trained descriptive + normative probes.
- `artifacts/generated/logs/`: training and eval logs with timings.

### VAL_TOP1, VAL_TOP8
Validation set metrics during training (catspace/nn/fb.py, val_metrics):
- **TOP1**: fraction of (s, g) pairs where B(g) is the highest-ranked goal among all holdout goals (top-1 recall).
- **TOP8**: fraction where B(g) is in the top 8 (top-8 recall).
- **Chance baseline** (for batch size B): 1/B and 8/B respectively.

At 30k: VAL_TOP1 = 0.033 (16.9× chance), VAL_TOP8 = 0.179 (11.4× chance). The model is significantly better than random but still learning.

---

## Naming & abbreviations

- **FB**: forward–backward (the embedding model).
- **MPS**: Metal Performance Shaders (Apple GPU acceleration).
- **Elo bin**: one of 10 strength buckets; used in omega.
- **DTM**: distance-to-mate (used in toy endgame experiments, not real boards).
- **Cosine reach**: L2-normalized dot product (the reach metric for real boards).
- **Descriptor vs normative**: game outcomes vs engine evals; two different eval heads.
- **Frozen probe**: F is fixed; only the linear head trains.
- **Waypoint**: an intermediate position in a decomposed plan.

---

## See also

- **ARCHITECTURE.md**: high-level system design and invariants.
- **README.md**: project overview and lessons learned.
- **JOURNAL.md**: timestamped results, timings, and interpretation.
