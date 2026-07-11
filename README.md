# catspace — a latent chess planner

(package `catspace`; formerly "latent-chess-planner-toys" — the toy-domain
milestone grew into the real-data phase, so the name did too)

Planning as a trajectory of the **cone** (discounted successor measure) in a learned
embedding, with self-discovered concepts — validated on exact small domains before
scaling to real data. This package contains everything built and learned in the
toy-domain phase: code, experiments, interactive viewers, figures, and the full
findings/bug ledger.

**Status: toy phase complete. Recommended next step (agreed): train on REAL data
(Lichess) at full-board scale — see §6.**

---

## 1. The idea in one paragraph

Embed chess states so that a state's position is determined by its *future*: the
discounted successor measure ("the cone") factorized as ρ(g|s) ≈ F(s)·B(g)
(tabular analogue of Forward–Backward representations; Touati & Ollivier 2021).
A **plan** steers the cone toward a goal (the MATE absorbing state — the earlier
DTM≤3 "region" goal was ablated away as an unnecessary oracle leak). Concepts are
*discovered*, not coded: VQ tokens over cone shapes, spectral structure, and
regions in the embedding, audited afterward against exact ground truth that the
model never saw. Everything below runs on one CPU in minutes.

## 2. The rung ladder (domains, all 5×5, all exact)

| rung | domain | states | character | headline results |
|---|---|---|---|---|
| 1 | **KRk** | 7,040 | one-sided | 95.5% mate vs random black from 32k random games (26× baseline, near the 100% ceiling); VQ tokens self-organize by distance-to-mate; neural FB generalizes to held-out states with **zero gap** (ρ=0.46 holdout vs 0.41 train; 99.8% conversion from unseen starts) |
| 2 | **KRRk** | 50,980 (+KRk stratum) | irreversible strata (rook capture = chute) | curriculum policy iteration: **97.7% vs optimal defense** at 1.11× tempo; rook-loss rate 45%→2.3% across rounds — rook safety learned, never named |
| 3 | **KRkn** | 158,232 (+KRk stratum) | two-sided: forks, pins, checks both ways, **39.5% game-theoretic draws**, max DTM 43 | reverse-start curriculum: 17.7%→48.7% conversion vs optimal; minimax readout: →68.5%; +3-ply search: **70%**, rook-loss 0.3%, tempo 1.11, 38.5% of wins tempo-perfect; WIN/DRAW frontier discovered at AUC 0.70 with zero labels; 91% of wins trade the knight first ("simplify when winning", emergent) |

## 3. The five architecture lessons (each discovered empirically here)

1. **The cone is opponent-conditioned** (KRk): a cone estimated under random-black
   play collapses against optimal defense. Fix: re-estimate under the induced
   dynamics — policy iteration with an opponent curriculum (45%→86%). The ω
   (opponent model) is load-bearing, not an extension.
2. **Exploration needs a curriculum when the opponent punishes** (KRkn): the knight
   hunts the rook, so naive data never reaches mate. Reverse-start curriculum
   (begin near mate, anneal outward) restores the signal.
3. **The readout's opponent model matters as much as the field's** (KRkn): switching
   move scoring from MEAN over replies (training mix) to MIN (the actual minimax
   opponent) was worth +20 points of conversion by itself — the max/expectation
   asymmetry, cashed out.
4. **Three readout regimes, all measured** (KRkn): expectation-deep backups = safe
   but passive; minimax-deep = pessimistic collapse (noise minimized to the floor;
   conversion 3% beyond 5 plies); **shallow (1–3 ply) minimax on the learned field
   = best**. Empirical case for selective/receding-horizon depth, not full-width.
5. **Score conventions are load-bearing** (KRk): two readouts of the *same* field
   differed 24% vs 99.8% purely on how a rook-capture outcome was scored (neutral
   vs 0.1%-quantile). Terminal-outcome scoring must live in one tested place.

## 4. Complete bug ledger (found in review, all fixed)

| bug | symptom | root cause | fix |
|---|---|---|---|
| sign-flipped "optimal" black | mates FASTER than DTM (impossible) — caught by Kaveh | black chose replies *minimizing* white's DTM | maximize; capture=draw=black's best. Invalidated an early "6/6 vs optimal" claim (retracted in RESULTS) |
| broken DTM ceiling | "optimal" white converting only 42% | DRAW absorbing scored 0.0 (= good) | draw scored as bad → ceiling = 100% |
| readout ω-mismatch | 48.7% vs 68.5% same field | MEAN over replies vs minimax opponent | MIN over replies at readout |
| draw-penalty convention | 24% vs 99.8% same field | capture outcome scored 0.0 (neutral) in one player | score draws at the 0.1% reach quantile |
| goal-region oracle leak | task spec used ground-truth DTM≤3 | oracle in the goal vector | ablated: B[MATE] alone matches/beats it — adopted |
| viewer "chosen" highlight | random-baseline games showed top-scored move as chosen while playing another (caught by Kaveh) | UI marked rank-1 candidate, not the played move | played-move flag in data; engine labels on game buttons |
| DTM/plies units confusion | "DTM 17 but mate in 9?" | viewer counted white moves, DTM counts plies | dual units + per-move tempo verdict (plies-≥-DTM law now verified per game) |
| shared start looked different | two same-size rings stacked | rendering, not data (verified identical states) | single shared start marker |
| numpy JSON | serialization crash | np scalars in payload | default converter |

## 5. File map & how to run

As of the layered refactor, the research code lives in the installable
`catspace/` package (clean interfaces: chain/scoring/readout/opponents/
game/arena/cone/concepts/planner/data/viz), with runnable drivers in
`experiments/`. See `ARCHITECTURE.md` for the layer diagram and the
project's invariants; this section is just the how-to-run.

Package (`catspace/`):
- `board.py` — 5×5 geometry; `domains/{krk,krkn,krrk}.py` — movegen, terminals,
  retrograde DTM, `build_chain()`/`compute_dtm()`
- `chain.py` — the one `TransitionChain` (CSR) representation every domain
  builds; `exact_P`/`empirical_P`
- `scoring.py` — `TerminalScores`, the single tested source of terminal-outcome
  conventions (README lesson 5, now a regression test, not a recurring bug)
- `cone/{tabular,neural}.py` + `cone/embedding.py` — `QuasimetricEmbedding`
  protocol (`TabularFB` via randomized SVD, `NeuralFB` via InfoNCE), `GoalSpec`,
  pluggable by name through `EMBEDDING_METHODS`
- `planner/{readout,policy,plans,selector,move_identity}.py` — MEAN/MIN
  aggregation + k-ply backup (lesson 3/4), `Policy` implementations, plan
  memory (`PlanMemory`/`PlanStore`: block reasons + event/drift wake triggers),
  `PlanSelector`/`MoveIdentity` protocols
- `opponents.py`, `game.py`, `arena.py` — `Opponent` protocol (`optimal_reply_table`
  is THE vectorized B_opt), `play_game`/`rollout_transitions`, `evaluate()`
- `train/{curriculum,checkpoints}.py` — `CurriculumTrainer` (replaces the ~6
  near-identical PI-round copies), counts-based bounded checkpoints
- `concepts.py` — `ConceptQuantizer` protocol, `KMeansVQ` (K exposed as a
  hyperparameter, replaces 7 copies)
- `data/{sources,shards,lichess,encode}.py` — `PairSource` protocol
  (`ChainRolloutSource` for toy rollouts); a WORKING streaming Lichess pipeline
  (zstandard, header-level prefilter, packed-bitboard shards)
- `viz/{projection,payload,build_html,plots}.py` — `Projection2D` protocol
  (`PCAProjection`/`TSNEProjection`[/`UMAPProjection`]), `FittedMap` (replaces
  `tsne_cache.pkl`), the viewer JSON payload builders, and the HTML injection
  step that was previously a manual/uncommitted step
- `abtest.py` — paired matched-seed comparisons with anytime-valid e-value
  tests (`EValueTest`, `compare()`), for comparing methods with statistical rigor

Drivers (`experiments/`): `krk_rung1.py` (rank probe / learning curve / concept
audit / VQ tokens / engine eval), `diagnostics.py` (post-hoc D1–D3), `train_{krk_pi,
krkn,krrk}.py` (curriculum training), `krkn_search_sweep.py` (minimax-depth sweep
+ goal ablation), `generalization.py` (neural-FB holdout result), `plan_memory_demo.py`,
`compare_methods.py`, `build_lichess_shards.py`, `viz/build_krk_viewer.py`,
`viz/build_krkn_viewer.py`, `viz/static_maps.py`, `repro_check.py`.

Artifacts (in `artifacts/`): RESULTS-v3.md (the full findings document, v3.0–v3.3
addenda), roadmap-v2.md, and the historical PNGs/HTML viewers from the toy-phase
push (kept as the record; regenerated output goes to `artifacts/generated/`,
gitignored). `krkn-linked-viewer.html` is the main interactive viewer: linked
map+board, split white/black edges, opponent diamonds selectable, tap-an-edge,
alt fans, cones.

Reproduce from scratch (one CPU, ~30 min total):
```
pip install -e .                       # or: pip install -r requirements.txt
python -m catspace.domains.krk      # sanity: 7040 states, DTM max 19
python experiments/krk_rung1.py        # rung 1 end-to-end (~15s)
python experiments/train_krkn.py       # curriculum training (resumable; ~7 min)
python experiments/krkn_search_sweep.py  # search sweep + goal ablation
python experiments/viz/static_maps.py --which krkn --projection tsne
python experiments/viz/build_krkn_viewer.py
python experiments/repro_check.py      # diff against tests/baselines/expected.json
pytest                                 # fast suite; `pytest -m slow` for the rest
```

## 6. Next phase: REAL DATA (the recommendation, and why)

Agreed direction: **stop scaling exact toys; train on actual games.** Rationale:
- The complaint is correct: random/self-play data on toys can't produce the
  structural richness (openings, pawn chains, irreversibility DAG, human trap
  regions) that makes discovered concepts interesting. Real games *live* in the
  strategically meaningful part of state space.
- The toy oracles (tablebase DTM opponents, reverse-start curricula) don't exist
  at 8×8 — their real-world analogues are exactly what Lichess provides: opponents
  at every Elo (ω for free), per-move clocks, and billions of trajectories.
- Lesson 1 (opponent-conditioning) transfers directly: per-Elo-bin cones are the
  real ω-ensemble. Lesson 4 says the planner should be shallow-selective on the
  learned field. Lesson 5 says pin terminal scoring in one audited module.
- Honest caveat: data-trained cones are *descriptive* (human-mixture dynamics).
  For play strength beyond imitation, keep the toy-validated loop: bootstrap the
  field from data, then refine with targeted self-play (the PI curriculum,
  with Maia nets as the annealed opponent instead of the DTM oracle).

Concrete laptop plan (maps to roadmap-v2 Milestone 2):
1. Data: Lichess monthly dumps (database.lichess.org; per-move clocks since 2017).
   Start with one month, one Elo band.
2. Encoder: frozen lc0/Maia trunk (or small ResNet trained from scratch on
   positions); FB heads (F conditioned on clock/ω; B over board states only) —
   the neural.py InfoNCE recipe scales as-is: geometric-horizon future pairs.
3. Opponent ω: bin by Elo first (Maia-style); the KRkn B_opt machinery is replaced
   by "black's reply distribution = what humans at this Elo actually played".
4. Goal: B[MATE] (validated oracle-free); regions/tokens for interpretation only.
5. Evaluation: held-out games (reach calibration, win/draw AUC vs game results),
   then arena vs Maia at fixed Elo — the G-M3 gate from roadmap-v2.
6. Viewers transfer: swap the 5×5 renderer for python-chess SVG; the linked-map
   machinery (t-SNE transform, split edges, alt fans) is board-size agnostic.

## 7. Known limitations of this package
- Tabular F/B everywhere except `neural.py` — full-board requires the neural path.
- t-SNE maps: neighborhoods faithful, inter-cluster distances not metric; fit on
  stratified subsamples with out-of-sample transform for trajectories.
- The KRkn deep-conversion frontier (DTM>~20 vs optimal defense) remains open at
  toy scale — closing it is expected from better fields + selective search, both
  of which are the real-data phase's tools.
- Cone sprays visualize opponent uncertainty at ε=0.25 (a visualization choice —
  against the deterministic oracle the cone degenerates to a line).
