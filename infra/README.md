# infra/ — everything around the engine, not in it

| piece | what it guarantees |
|---|---|
| `preempt.py` | SIGTERM/SIGINT ⇒ checkpoint + exit 0 (spot-instance contract) |
| `checkpoint.py` | FULL training state (model+opt+sched+step), atomic writes, `--resume auto` |
| `observability/` | one `*_metrics.jsonl` per run (+ MLflow mirror when `MLFLOW_TRACKING_URI` set); timers ride on rows |
| `cloud/` | storage & compute scaffolding (DVC remote, spot workflow) |

**The run contract** (every trainer follows it):
1. `PreemptGuard` installed before the loop; checked every step.
2. Checkpoint ladder (`_step{N}.pt`) + `_latest.pt`, all with optimizer state.
3. `--resume auto` continues from `_latest.pt` (step, opt, sched restored).
4. `RunLogger` rows at eval cadence: loss terms, eff_rank, throughput, timer
   splits — everything the figure tools (`tools/fig_train_curves.py`,
   `fig_probe_curve.py`) need, with no re-running.

Tested in `tests/test_infra.py`.

## DVC & MLflow (owned here, even where the tools pin their paths)

- **DVC**: `.dvc/` must live at repo root (tool requirement); policy and remote
  setup are infra's — see `cloud/README.md`. Rule: every dataset gets a `.dvc`
  pointer in git, bytes go to the remote.
- **MLflow**: `RunLogger` mirrors every metrics row to MLflow when
  `MLFLOW_TRACKING_URI` is set (e.g. `file:mlruns` locally, an HTTP server in
  the cloud). The JSONL stays the source of truth; MLflow is the dashboard.

## Parallelization & data movement (the decision record)

- Hot paths are batched tensor ops already: trainer batches, embed passes,
  MCTS child-batch evaluation, planner successor batches; corpus tokenization
  is multiprocessing-parallel (`build_jepa_corpus.py --workers`); Stockfish
  labeling runs one engine per core.
- **Memoization**: `catspace/search/memo.py` (`BoundedMemo`, LRU, hit-rate
  instrumented — rates surface in the engine's traces); the MCTS core keeps its
  own transposition/eval caches (historically 20–34% of a game's evals).
- **Ray: not yet.** On one Apple-Silicon box, unified memory means the
  data-movement problem Ray solves (cross-process/node object transfer) mostly
  does not exist, and its serialization would ADD movement. Ray Tune is already
  scaffolded for sweeps (`catspace/train/scaffold.py`); the genuine Ray fit is
  **parallel self-play generation for expert iteration** and multi-node
  training — adopt it when either lands off-laptop. (Ray Train blocked on
  py3.14 at last check; re-verify then.)

## Profiling (the zoom lens; RunLogger timers are the always-on layer)

- `infra/observability/profile.py::profile_block` — torch.profiler top-ops
  table + chrome trace for code you can edit.
- **py-spy** (installed) for live runs with zero code changes:
  `py-spy top --pid $(cat artifacts/experiments/<run>.pid)`.
