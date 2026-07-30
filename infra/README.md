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
