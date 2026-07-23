"""catspace/tracking.py -- MLflow experiment tracking, thin and failure-proof
(TRAINING_STANDARDS.md rule 4: use existing tracking, don't hand-roll).

Every trainer wraps its run in `track_run(name, args)`; params (full argparse
namespace), per-step metrics, and final verdict tags land in the local ./mlruns
store (inspect with `mlflow ui`). Tracking must NEVER kill a training run: every
call degrades to a no-op on any mlflow failure.
"""
from __future__ import annotations

import contextlib
from pathlib import Path


def _mlflow():
    try:
        import mlflow
        # sqlite backend (mlflow >= 3.14 deprecates the ./mlruns file store);
        # browse with: mlflow ui --backend-store-uri sqlite:///mlflow.db
        mlflow.set_tracking_uri(f"sqlite:///{Path(__file__).resolve().parents[1] / 'mlflow.db'}")
        return mlflow
    except Exception:
        return None


@contextlib.contextmanager
def track_run(experiment: str, args=None, run_name: str | None = None):
    """Context manager yielding a logger with .metrics(dict, step) / .tag(k, v).
    Logs all argparse params at entry. No-ops safely if mlflow is unavailable."""
    ml = _mlflow()

    class _Log:
        def metrics(self, d: dict, step: int | None = None):
            if ml is not None:
                with contextlib.suppress(Exception):
                    ml.log_metrics({k: float(v) for k, v in d.items()}, step=step)

        def tag(self, k: str, v):
            if ml is not None:
                with contextlib.suppress(Exception):
                    ml.set_tag(k, str(v))

    if ml is None:
        yield _Log()
        return
    try:
        ml.set_experiment(experiment)
        with ml.start_run(run_name=run_name):
            if args is not None:
                with contextlib.suppress(Exception):
                    ml.log_params({k: str(v)[:250] for k, v in vars(args).items()})
            with contextlib.suppress(Exception):     # per-record provenance: every
                import subprocess                    # training run pins its code commit
                ml.set_tag("git_commit", subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True).stdout.strip())
            yield _Log()
    except Exception:
        yield _Log()
