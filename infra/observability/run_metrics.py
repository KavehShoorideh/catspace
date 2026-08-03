"""infra/observability/run_metrics.py -- one metrics stream per training run,
so every figure tool can be fed after the fact.

RunLogger writes append-only JSONL rows ({"step":..., "wall_s":..., <metrics>})
next to the run's checkpoints, and mirrors to MLflow when MLFLOW_TRACKING_URI
is set (off-the-shelf observability; never hand-roll the dashboard). Timers are
context managers; their totals ride along on the next log() row as <name>_s.

    log = RunLogger("artifacts/experiments/jepa_pretrain")
    with log.timer("data"):   ...
    with log.timer("fwd"):    ...
    log.log(step=1000, l_dyn=0.12, eff_rank=41.2)

Consumers: tools/fig_train_curves.py (loss/rank/throughput panels),
tools/fig_probe_curve.py --loss (step:loss pairs).
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


class RunLogger:
    def __init__(self, out_prefix: str, mlflow_run: str = ""):
        self.path = Path(f"{out_prefix}_metrics.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self._last_log_t = self.t0
        self._last_step = 0
        self._timers: dict[str, float] = {}
        self._mlflow = None
        if os.environ.get("MLFLOW_TRACKING_URI"):
            try:
                import mlflow

                from catspace.io.paths import mlflow_uri
                mlflow.set_tracking_uri(mlflow_uri())
                mlflow.set_experiment(mlflow_run or Path(out_prefix).name)
                mlflow.start_run(run_name=Path(out_prefix).name)
                self._mlflow = mlflow
            except Exception:
                self._mlflow = None

    @contextmanager
    def timer(self, name: str):
        t = time.time()
        try:
            yield
        finally:
            self._timers[name] = self._timers.get(name, 0.0) + time.time() - t

    def log(self, step: int, **metrics):
        now = time.time()
        row = dict(step=int(step), wall_s=round(now - self.t0, 1),
                   steps_per_s=round((step - self._last_step)
                                     / max(now - self._last_log_t, 1e-9), 3),
                   **{k: round(float(v), 6) for k, v in metrics.items()},
                   **{f"{k}_s": round(v, 2) for k, v in self._timers.items()})
        self._last_log_t, self._last_step = now, step
        self._timers.clear()
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if self._mlflow is not None:
            try:
                self._mlflow.log_metrics(
                    {k: v for k, v in row.items()
                     if isinstance(v, (int, float)) and k != "step"}, step=int(step))
            except Exception:
                pass
        return row
