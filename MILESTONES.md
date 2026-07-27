# MILESTONES.md — the locked 30,000-foot plan (Kaveh, 2026-07-27)

**This file is the roadmap.** Details inside a milestone may change; the milestone structure and
the LOCKED DECISIONS must not. Re-litigating a locked decision requires Kaveh's explicit say-so,
recorded here with a date. (Origin: Kaveh's 2026-07-27 reset — "the details can change, the
overall 30k-foot plan shouldn't.")

## North star

An engine that beats fallible opponents by **planning through their errors**: use a
**reachability field** to map the outcome basins and their **transition points under different
player models**; train a **high-level planner** to navigate to those transition points via
**subgoals** (transition points are themselves subgoals on the way to mate); use **MCTS to probe**
when near a subgoal or uncertain, guided by the reachability field under the opponent model and
clock; hand off to **tablebases at ≤7 pieces**. Transitions are a function of the ENTIRE game
state: position+history, both players (Elo + style), clocks, possibly past game outcomes.
Chess is the laboratory; the machinery (fields, transition estimators, subgoal planners) is the
research product, headed for publication.

## Empirical foundation (M0 — DONE 2026-07-27)

- Outcome basins are REAL and sharp under perfect play: SF-vs-SF Win↔Loss barrier ≈ **0.00**
  (absorbing); perfect WDL bimodal at ALL material; 36% of outcome entropy explained by position.
- Human play (1400–1800) LEAKS: Win↔Loss ≈ **0.27/0.29** per ~6 plies; only 7% explained by
  position; the midgame is a jumble that **crystallizes at ~15–22 pieces** (bimodality 0.39→0.97).
- ⇒ **The exploitable edge IS the human leak**, concentrated in the crystallization zone.
- Stopgap transition estimator (Maia policy × SF committor-loss) **predicts real blunders**
  (ρ≈+0.54 smoke; top-B quartile 37.5% real-blunder rate vs 0% bottom).
- Scripts: `transition_map/bands/time`, `msm_basins`, `committor_by_material`,
  `sf_wdl_by_material`, `sf_vs_human_bands`, `engine_vs_human_basins`, `blunder_model`.

## Locked decisions

1. **Geometry-first navigation.** The engine navigates on reachability distances — d(s, subgoal
   region), terminating in d(s, TB-won→mate). WDL/committor outputs are permitted ONLY as (a)
   analysis instrumentation and (b) transition **labels** (a transition = a crossing between
   outcome basins). Never as the engine's primary navigation value.
2. **Pretrained trunk + IQE head.** The board encoder is a **frozen pretrained Leela-family
   trunk** (Maia/lc0, via `lc0 leela2onnx` → `lczerolens` torch module — VERIFIED 2026-07-27:
   batched forward 277 pos/s CPU, policy+wdl heads, trunk hookable). New head = IQE quasimetric.
   We do not hand-roll board encoders. Encoder input = lc0 112 planes (8-position history
   included; **no clock in the encoder**).
3. **Context enters the TRANSITION ESTIMATOR, not the encoder.**
   `T(φ(s), clock_both, Elo_both, z_both, ply, [past outcomes]) → per-side crossing risk`
   (t_win, t_loss → net flux Φ = t_win − t_loss, sharpness σ = t_win + t_loss).
4. **Player model = known Elo part + unknown residual z.** Population prior p(z | Elo);
   per-player offline embeddings where history exists (name-masked ids); **in-game posterior
   tightening** of z from observed moves (cold start = prior). Matilda-style residual design.
5. **Transition points ARE subgoals.** Plans are chains:
   … → transition region (THEIR error zone, not ours) → won region → TB-won (≤7p) → mate.
   The planner holds a PORTFOLIO of subgoals (optionality + denial + opportunism —
   `catspace/planner/optionality.py`, built & 16/16 tested). MCTS probes when near or uncertain,
   guided by the d-field under the opponent model + clock. Tablebase plays ≤7p directly.
6. **One best line.** Always work on the current best-guess architecture. A superseded approach
   is killed immediately — runs stopped, docs marked — no parallel maintenance of inferior lines.
7. **Ops bar.** Every dataset DVC-tracked; every training MLflow-tracked (`catspace/train/
   scaffold.py`); batched tensor ops (MPS/GPU) over per-position subprocesses wherever possible;
   every comparison statistically rigorous (anytime-valid e-values / SPRT / bootstrap CIs, n
   pre-registered). No journal numbers without a printed script VERDICT.

## Milestones

### M1 — Substrate: Leela-trunk IQE reachability field
Freeze a pretrained trunk (Maia-1500/1900 and/or a strong small lc0 net — pick by gate metrics);
attach an IQE head (+ thin adapter). Train the head on (i) same-game ply-gap pairs from mixed
human + engine corpora (full-phase, openings included), (ii) tablebase DTZ anchors (d_mate),
(iii) repulsion/unreachability.
- **Gates:** pair-order ≥ 0.94 (match ClockField v3); d_mate-vs-DTZ ≥ +0.81 in-distribution AND
  better than 0.505 off-distribution (trunk should generalize); startpos/opening values sane
  (fixes the opening-blindness class of bug structurally); eff-rank healthy. Same eval protocol
  as v3 for a fair kill decision.
- **Kills on green:** ClockField line (v2/v3/v4 plans), committor-greedy readouts.

### M2 — Transition estimator T(s, context) — the centerpiece
A context head over frozen φ(s): input [clock_mover, clock_opp, Elo_mover, Elo_opp, z, ply, …] →
per-side crossing risk (Φ, σ).
- **M2a** rating + clock conditioning. Requires per-move clocks → extend Stage-A game records
  with `[%clk]` arrays (raw PGNs have them; re-run records build). Train on real games:
  label = realized committor swings (SF-labeled subset) + game outcomes; distill the Maia×SF
  stopgap where labels are thin.
- **M2b** per-player z offline (identity records, players ≥20 games, rating-residual style).
- **M2c** in-game z tightening (posterior update from move surprisals; cold start = prior).
- **Infra:** batched ONNX Maia policies (tensor ops; 277+ pos/s vs ~1/s subprocess) for both
  training targets and MCTS opponent models.
- **Gates:** beats the stopgap B(s,r) on held-out real-blunder prediction (ρ, quartile lift,
  calibration); clock effect real on matched positions (risk ↑ as clock ↓); rating monotonic;
  z adds statistically significant lift over Elo-only.

### M3 — Transition atlas + subgoal generator (the map)
For an opponent context, map reachable high-flux transition regions:
score(region) = crossing flux (T) × reachability (d) [× exemplar density]. Deliverables:
per-context atlas visualizations + a queryable API `(s, context) → ranked subgoal regions`.
- **M3b — Concept mining (Kaveh 2026-07-27):** matched case-control on similar positions that
  did vs didn't transition → **attacking factors** (pins, hanging pieces, king exposure,
  tension, …) vs **protective factors**; hand-coded extractors first, SAE/CAV stack later.
- **Gates:** out-of-sample validation — games passing through predicted-high-flux regions show
  elevated actual crossing rates; concept effects significant under matching.

### M4 — Planner: subgoal-chain navigation (the strategist)
Wire the M3 generator into the built portfolio planner (optionality/denial/opportunism);
chain through TB-won regions to mate; re-plan opportunistically each ply.
- **Gates:** vs fixed Maia — planner-on steers play into predicted-high-flux regions (mean T of
  reached positions ↑ vs planner-off, e-value significant) AND lifts score.

### M5 — MCTS as the probe (the prober)
Reachability-guided search: node signal = progress on d-to-active-subgoal (+ flux shaping; NO
WDL leaf values); expansion weighted by the OPPONENT MODEL (expectimax over Maia/z policy —
already measured better than minimax: 0.125 vs 0.094); clock-aware via T; TB handoff ≤7p
(built). Planner triggers probes when near a subgoal or uncertain.
- **Gates:** strength-per-node curve vs the Maia ladder; ≥ 0.125 shallow baseline at equal
  budget, scaling with nodes; beats a WDL-guided ablation at equal nodes.

### M6 — Close the loop: the exploiter
Full-stack play vs the Maia ladder (then other bots): measure the **exploitation dividend** =
score(with opponent model) − score(opponent-agnostic) at equal node budget, SPRT/e-values,
in-game z tightening on. Publishable evaluation + digest write-up.

**Sequencing:** M1 → M2 → (M3 ∥ M5) → M4 → M6. M3b concept mining can run any time after M2.

## Deferred / out of scope (no work without a recorded plan change)
Dockerized service stack; RL-trained plan selector (revisit after M4); non-board endings
(time/resign as outcome classes); viz niceties beyond the atlas.

## Acronyms & symbols (canonical — GLOSSARY.md defers to this)

**Chess & engines**
| term | meaning |
|---|---|
| WDL | Win / Draw / Loss — the three outcomes; a "WDL head" outputs (p_win, p_draw, p_loss). |
| SF | Stockfish — strongest open engine; our near-perfect reference/oracle. |
| lc0 / Leela | Leela Chess Zero — open AlphaZero-style neural engine; "Leela-family trunk" = the body of an lc0-format net. |
| Maia | (name, not acronym) lc0-format nets trained to predict HUMAN moves at a rating band (maia-1100…1900); our rating-conditioned human policy. |
| AZ | AlphaZero — DeepMind's self-play engine; "AZ-style" = policy+value net + PUCT search. |
| TB | (Syzygy) tablebase — precomputed PERFECT play for all ≤7-piece positions. |
| DTZ | Distance To Zeroing — TB metric: plies to an irreversible move (capture/pawn push/mate) under perfect play. Our exact distance anchor. |
| DTM | Distance To Mate — plies to checkmate under perfect play (older experiments). |
| Elo | rating scale (named after Arpad Elo — NOT an acronym). |
| PGN / FEN | Portable Game Notation (game text) / Forsyth–Edwards Notation (single-position text). |
| UCI / SAN | Universal Chess Interface (engine protocol) / Standard Algebraic Notation (move text). |
| CCRL / TCEC | Computer Chess Rating Lists / Top Chess Engine Championship — engine-game archives. |
| KQvK, KRRvKBP… | material classes: White's pieces "v" Black's (K king, Q queen, R rook, B bishop, N knight, P pawn). |
| ply | one half-move (one player's move). |
| ACPL | Average CentiPawn Loss — mean eval loss per move, hundredths of a pawn (legacy metric here). |

**Our method (math / ML)**
| term | meaning |
|---|---|
| IQE | Interval Quasimetric Embedding — the head that turns two embeddings into an ASYMMETRIC distance with the triangle inequality guaranteed by construction (Wang & Isola). Our d(·,·). |
| quasimetric | a distance where d(a,b) ≠ d(b,a) is allowed (chess reachability is directional). |
| φ (small phi) | the board EMBEDDING vector from the trunk. |
| d(s,g), d_mate | learned reachability distances: position→region, position→mate boundary. |
| committor c(s) | from transition-path theory: P(hit the win boundary before the others) UNDER A PLAY MEASURE. Play-measure-dependent (human c ≠ perfect c). |
| T(s, context) | the TRANSITION ESTIMATOR (M2): per-side basin-crossing risk given clocks/Elos/z/ply. |
| Φ (capital Phi) | net favorable flux = t_win − t_loss. ⚠ not the same symbol as φ the embedding. |
| σ (sigma) | sharpness = t_win + t_loss (how swingy a position is). |
| z | the UNKNOWN part of the player model — a style-residual embedding on top of known Elo. |
| MCTS | Monte Carlo Tree Search — build a tree by repeated select/expand/evaluate/backup. |
| PUCT | Prior + Upper Confidence bound applied to Trees — the AZ selection rule MCTS uses. |
| expectimax / minimax | back up the EXPECTED value over opponent replies (fallible foe) vs the WORST-case (perfect foe). |
| MSM | Markov State Model — discretize states, count transitions, analyze the matrix. |
| PCCA(+) | Perron Cluster Cluster Analysis — spectral grouping of an MSM into metastable basins. |
| TPT | Transition Path Theory — committors, reactive flux, transition rates. |
| MFPT | Mean First Passage Time — expected steps to first reach a target set. |
| SAE | Sparse AutoEncoder — unsupervised concept dictionary (M3b later stage). |
| CAV / TCAV | (Testing with) Concept Activation Vectors — supervised concept directions (M3b). |
| NLL | Negative Log-Likelihood (training objective). |
| OOD | Out-Of-Distribution — inputs unlike the training data. |
| UMAP / t-SNE | nonlinear 2-D projections for the maps (Uniform Manifold Approx. & Projection / t-dist. Stochastic Neighbor Embedding). |
| KDE | Kernel Density Estimate — the smooth histograms in ridgeline plots. |

**Metrics & statistics**
| term | meaning |
|---|---|
| ECE | Expected Calibration Error — mean gap between predicted probability and observed frequency (0 = perfectly calibrated). |
| MAE | Mean Absolute Error. |
| ρ / Spearman | rank correlation (−1…+1); our ordering-quality metric. |
| eff_rank | effective rank — how many dimensions an embedding really uses; the collapse gate. |
| CI | Confidence Interval. |
| SPRT | Sequential Probability Ratio Test — stop-when-evidence-suffices A/B for engine matches (fishtest-style). |
| e-value / anytime-valid | evidence measure you may inspect at ANY time without p-hacking; our A/B harness. |
| A/B, h2h | two-variant controlled comparison; head-to-head. |

**Infrastructure**
| term | meaning |
|---|---|
| DVC | Data Version Control — git-style versioning of datasets/models (pointer files in git, bytes in cache). |
| MLflow | experiment tracker — logs params/metrics/artifacts per training run (`mlflow ui`). |
| ONNX | Open Neural Network Exchange — portable net format; `lc0 leela2onnx` → loadable in PyTorch via lczerolens. |
| MPS | Metal Performance Shaders — Apple-silicon GPU backend for PyTorch (our `device="mps"`). |
| HP | hyperparameter. |
| WAL | Write-Ahead Log — SQLite journal mode (banned for the probe cache; grew unbounded, filled the disk). |
| npz / parquet | NumPy zipped arrays / columnar table format (our tensors / our game records). |

**Legacy (appears in old files & journal history only)**
| term | meaning |
|---|---|
| FB, F(s)/B(g) | the OLD two-encoder Forward/Backward field (superseded by single-space φ; survives in filenames like `lichess_fb*.pt`). |
| ω (omega) | the old strength/time conditioning of F (superseded; context now enters T, not the encoder). |
| InfoNCE | the contrastive loss the legacy FB field used (Noise-Contrastive Estimation family). |
