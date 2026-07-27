"""catspace/train/scaffold.py -- the STANDARD training scaffold (Kaveh 2026-07-26: "use ML
frameworks like Ray to scaffold training instead of hand-rolling"). Composes the frameworks so a
new trainer supplies ONLY a plain-PyTorch step_fn; all infra comes from established tools:

  * TRACKING  -> MLflow, via the existing failure-proof catspace.tracking.track_run (params = full
                 args, per-step metrics, verdict tags, git commit; ./mlflow.db, `mlflow ui`).
  * CHECKPOINTS -> ladders + embedded provenance (TRAINING_STANDARDS 1 & 3): <out>_step{N}.pt at an
                 interval PLUS a rolling <out>_latest.pt, each embedding the full args namespace +
                 git commit + step. Never only-final; distinct runs use distinct --out.
  * HEALTH GATES -> a gates_fn(model)->dict logged every eval (eff_rank etc., TRAINING_STANDARDS 6).
  * SWEEPS / ORCHESTRATION -> Ray TUNE (tune_sweep): parallel trials + grid/random search.

FRAMEWORK CHOICE (verified on THIS machine, py3.14 + Apple MPS, 2026-07-26): Ray TUNE works on MPS;
Ray TRAIN's distributed worker/controller-actor abstraction does NOT (cloudpickle + Controller-actor
state queries fail under py3.14) -- so we scaffold with Ray Tune (Ray's HP-search/orchestration
strength) + a plain in-process loop, NOT Ray Train. Documented deviation; revisit Ray Train when it
supports py3.14. Everything degrades gracefully: no ray -> single in-process run; no mlflow -> no-op.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from catspace.tracking import track_run


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def save_torch_ckpt(model, out_prefix, step: int, args=None, extra: dict | None = None,
                    latest: bool = True) -> Path:
    """Ladder checkpoint with embedded provenance (TRAINING_STANDARDS 1 & 3). Writes
    <out_prefix>_step{step}.pt (+ <out_prefix>_latest.pt) carrying state_dict, step, the full args
    namespace, git commit, and any `extra` (e.g. optimizer state, config). Returns the step path."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "step": step,
        "args": (vars(args) if args is not None and not isinstance(args, dict) else args) or {},
        "git_commit": _git_commit(),
        "arch": type(model).__name__,
    }
    if extra:
        payload.update(extra)
    step_path = out_prefix.with_name(f"{out_prefix.name}_step{step}.pt")
    torch.save(payload, step_path)
    if latest:
        torch.save(payload, out_prefix.with_name(f"{out_prefix.name}_latest.pt"))
    return step_path


@dataclass
class TrainConfig:
    out: str                                   # checkpoint prefix (distinct per run -- no overwrite)
    steps: int = 1000
    ckpt_every: int = 200                      # ladder interval
    eval_every: int = 100                      # gates + metric logging interval
    experiment: str = "catspace"               # MLflow experiment name
    run_name: str | None = None
    report_tune: bool = False                  # call ray.train/tune report (set True inside a sweep)
    device: str = "auto"
    extra: dict = field(default_factory=dict)


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def standard_train(step_fn: Callable, model, cfg: TrainConfig, args=None,
                   gates_fn: Callable | None = None) -> dict:
    """Run the STANDARD loop. step_fn(model, step) -> {metric: value} does ONE optimizer step
    (the only thing a trainer must supply). The scaffold handles MLflow logging, checkpoint ladders,
    health gates, and (inside a sweep) tune reporting. Returns the last metrics dict."""
    last: dict = {}
    with track_run(cfg.experiment, args=args, run_name=cfg.run_name) as log:
        for step in range(1, cfg.steps + 1):
            metrics = step_fn(model, step) or {}
            if step % cfg.eval_every == 0 or step == cfg.steps:
                if gates_fn is not None:
                    with torch.no_grad():
                        metrics = {**metrics, **(gates_fn(model) or {})}
                log.metrics(metrics, step=step)
                if cfg.report_tune:
                    try:
                        from ray import tune
                        tune.report(metrics)
                    except Exception:
                        pass
                last = metrics
            if step % cfg.ckpt_every == 0 or step == cfg.steps:
                save_torch_ckpt(model, cfg.out, step, args=args, extra=cfg.extra)
        for k, v in last.items():
            log.tag(f"final_{k}", v)
    return last


def tune_sweep(trainable: Callable, param_space: dict, num_samples: int = 1,
               metric: str = "loss", mode: str = "min", resources: dict | None = None):
    """Ray TUNE sweep (Ray's HP-search strength; verified on MPS). `trainable(config)` is a plain
    function that trains and calls ray.train.report/tune.report. Returns the best ray ResultGrid
    result. Falls back to a single in-process trainable(...) call if ray is unavailable."""
    try:
        from ray import tune
    except Exception:
        return trainable(param_space)                       # graceful: no ray -> single run
    trainable_r = tune.with_resources(trainable, resources) if resources else trainable
    tuner = tune.Tuner(trainable_r, param_space=param_space,
                       tune_config=tune.TuneConfig(num_samples=num_samples))
    results = tuner.fit()
    return results.get_best_result(metric=metric, mode=mode)


# --------------------------------------------------------------------------------------------------
def _smoke_trainable(config):
    """A Ray Tune trainable MUST be a top-level function that IMPORTS EVERYTHING INSIDE ITS BODY and
    references no outer module globals -- cloudpickle serializes referenced globals BY VALUE (which
    can drag in non-serializable module objects), but in-body imports resolve BY REFERENCE on the
    worker. This is THE pattern real trainers must follow to parallelize under Tune."""
    import torch as t, torch.nn as tnn
    from ray import tune
    d = t.device("mps" if t.backends.mps.is_available() else "cpu")
    m = tnn.Linear(8, 1).to(d); o = t.optim.Adam(m.parameters(), config["lr"]); loss = None
    for _ in range(25):
        x = t.randn(64, 8, device=d); loss = ((m(x) - x.sum(1, keepdim=True)) ** 2).mean()
        o.zero_grad(); loss.backward(); o.step()
    tune.report({"loss": float(loss)})


def _tests():
    """Smoke: the scaffold trains a tiny model end-to-end on this machine's device, writes a ckpt
    ladder with provenance, and runs a Ray Tune sweep -- the whole framework path, no hand-rolled
    infra."""
    import tempfile, torch.nn as nn
    ok = True

    def check(name, cond):
        nonlocal ok; ok &= bool(cond); print(f"  {'OK ' if cond else 'FAIL'} {name}")

    dev = resolve_device("auto")
    print(f"  device = {dev}")
    tmp = Path(tempfile.mkdtemp())

    # 1. standard_train: ladder + latest on disk, provenance embedded, gates logged.
    model = nn.Linear(8, 1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    def step_fn(m, step):
        x = torch.randn(64, 8, device=dev)
        loss = ((m(x) - x.sum(1, keepdim=True)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        return {"loss": float(loss)}

    def gates_fn(m):
        w = m.weight.detach().cpu().float()
        return {"eff_rank": float((torch.linalg.svdvals(w) > 1e-6).sum())}

    cfg = TrainConfig(out=str(tmp / "smoke"), steps=40, ckpt_every=20, eval_every=10,
                      experiment="catspace_smoke")
    last = standard_train(step_fn, model, cfg, args={"lr": 1e-2, "note": "smoke"}, gates_fn=gates_fn)
    check("standard_train returns metrics incl gate", "loss" in last and "eff_rank" in last)
    check("ladder step ckpts written", (tmp / "smoke_step20.pt").exists() and (tmp / "smoke_step40.pt").exists())
    check("rolling latest written", (tmp / "smoke_latest.pt").exists())
    payload = torch.load(tmp / "smoke_step40.pt", weights_only=False)
    check("provenance embedded (args + git + step + arch)",
          payload["args"].get("note") == "smoke" and payload["step"] == 40
          and "git_commit" in payload and payload["arch"] == "Linear")

    # 2. tune_sweep: Ray Tune grid over lr, on this device, returns a best result.
    from ray import tune
    best = tune_sweep(_smoke_trainable, {"lr": tune.grid_search([1e-2, 3e-3])}, metric="loss", mode="min")
    got_best = hasattr(best, "config") and "lr" in best.config
    check("tune_sweep returns a best trial (Ray Tune on this device)", got_best)

    print("ALL SCAFFOLD TESTS PASSED" if ok else "SCAFFOLD TESTS FAILED")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _tests() else 1)
