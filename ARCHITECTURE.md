# Architecture

Layered package (`latentchess/`) replacing the original pile of research
scripts (`code/`, deleted; git history preserves it). Layers, bottom to top:

```
board                  5x5 geometry (sq/rc/king/rook/knight moves)
  |
domains/{krk,krkn,krrk}  movegen, terminals, retrograde DTM
  |
chain                  TransitionChain: the ONE CSR representation every
  |                    domain builds (move_ptr/move_kind/out_ptr/out_flat)
  |
scoring                TerminalScores: the ONE tested terminal-outcome
  |                    convention (mate/draw/bwin), used by every readout
  |
cone/{tabular,neural}  QuasimetricEmbedding: F/B via SVD or InfoNCE,
  |  cone/embedding      reach(emb, goal, idx) -- the pluggable "how is the
  |                      quasimetric space built" seam (EMBEDDING_METHODS)
  |
concepts               ConceptQuantizer: VQ tokens (K exposed), pluggable
  |
planner/{readout,      MEAN|MIN aggregation + k-ply backup; Policy protocol;
  policy,plans,           PlanMemory (blocked-plan reasons + event/drift wake
  selector,                triggers); PlanSelector + MoveIdentity protocols
  move_identity}          (RL-based selection and decomposition generators
  |                       are the M1.5 research phase, not built here)
  |
opponents, game, arena  Opponent protocol; play_game/rollout_transitions;
  |                     evaluate() -> ArenaResult (conversion/tempo/rook-loss/...)
  |
train/{curriculum,      CurriculumTrainer (the ONE PI-round loop, replacing
  checkpoints}            ~6 near-identical copies); bounded npz checkpoints
  |
data/{sources,shards,   PairSource protocol; ChainRolloutSource (toy) and a
  lichess,encode}         WORKING streaming Lichess pipeline (never
  |                       materializes a decompressed .pgn; packed-bitboard
  |                       shards; bounded by --max-games/--max-gb)
  |
abtest                  Paired matched-seed comparisons + anytime-valid
  |                     e-value tests (EValueTest, compare()) -- how methods
  |                     get compared with statistical rigor
  |
viz/{projection,        Projection2D protocol (PCA/t-SNE/[UMAP]); FittedMap
  payload,build_html,     (replaces tsne_cache.pkl); viewer JSON payload
  plots}                  builders; the __DATA__ HTML injection step
  |
io/paths                repo-rooted data/derived/generated dirs (kills the
                         old hardcoded /home/claude/... paths)
```

`experiments/` holds thin, runnable drivers over this package (one CLI per
experiment/report, ~30-100 lines each) -- see README.md section 5.

## Invariants

1. **One chain representation.** Every domain (`krk`, `krkn`, `krrk`) builds
   a `TransitionChain` (CSR). No other chain type exists; readout/rollout/
   trainer code is written once against it.
2. **One terminal-scoring source.** `scoring.TerminalScores` is the only
   place mate/draw/white-terminal outcomes get a numeric value. This was a
   documented, repeated bug source (README lesson 5) before the refactor;
   `tests/test_readout.py`/`test_laws.py` now regression-test it.
3. **All randomness is seeded through `np.random.Generator` instances**,
   passed in or explicitly created -- never bare `np.random.*` module calls.
   Determinism per seed is a test requirement throughout.
4. **Artifacts only under `data/` (gitignored) and `artifacts/generated/`
   (gitignored).** Committed `artifacts/*.png`/`*.html`/`*.md` are the
   historical record, not regenerated in place. `latentchess/io/paths` is
   the only place that resolves these directories.

## What's deferred

- **M1.5 (hierarchical planning research, post-refactor):** subgoal
  decomposition (enabling sets from refutations, region-graph pathfinding,
  geodesic midpoints), give-up rules, `LearnedSelector`/`PlanMCTS`,
  displacement-based `MoveIdentity`, technique quantization. `PlanMemory`/
  `PlanSelector`/`MoveIdentity` protocols exist now with trivial baselines
  (`GreedyReach`, `SyntacticIdentity`/`RegionPairIdentity`) so this research
  has somewhere to plug in without another refactor.
- **Real-data milestone:** eval heads (normative + descriptive, frozen-probe
  then joint training), Stockfish/lc0-audited labels, torch + MPS training,
  a UCI Stockfish opponent against real `python-chess` boards, a full-board
  SVG viewer. The data layer (`PairSource`, packed-bitboard shards, the
  streaming Lichess pipeline) and the embedding/quantizer/projection
  registries are built so these are plug-ins, not rewrites.

Design discussion and the reasoning behind both deferred tracks lives in
project memory and `artifacts/RESULTS-v3.md`/`roadmap-v2.md`.
