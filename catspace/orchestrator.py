"""catspace/engine/orchestrator.py -- PROBE ORCHESTRATION on RAY (Kaveh 2026-07-25:
'maybe ray is the right framework... don't hand roll').

Single-flight, position-indexed async probe execution with milestone streaming:
  - submit(kind, epd, fn): identical in-flight keys COALESCE onto one running task
    (never double-calculate); finished keys resolve from the memo instantly
  - fn(report) runs as a Ray task; report(stage, payload) streams milestones through the
    coordinator actor; subscribers poll events(key) or pass on_event to submit
  - wait / as_completed / invalidate(kind) as before

The coordinator (a Ray actor) owns {key -> state, stages, ObjectRef}; because it is an
actor, MULTIPLE PROCESSES (game workers) sharing one Ray instance also share the memo and
the in-flight table -- the cross-process layer the hand-rolled version could not offer.
ray.init is lazy (ignore_reinit_error) and local by default."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import ray


def _ensure_ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False,
                 logging_level="ERROR", num_cpus=4)


@ray.remote
class _Coordinator:
    def __init__(self):
        self.jobs: dict = {}       # key -> {"status", "stages": [..], "result", "error"}

    def try_claim(self, key) -> str:
        """returns 'claimed' (caller runs it) | 'inflight' | 'done'."""
        j = self.jobs.get(key)
        if j is None:
            self.jobs[key] = {"status": "running", "stages": [], "result": None, "error": None}
            return "claimed"
        return "done" if j["status"] in ("done", "error") else "inflight"

    def report(self, key, stage, payload):
        self.jobs[key]["stages"].append((stage, payload))

    def finish(self, key, result=None, error=None):
        j = self.jobs[key]
        j["result"] = result; j["error"] = error
        j["status"] = "error" if error else "done"

    def state(self, key):
        return self.jobs.get(key)

    def invalidate(self, kind=None):
        self.jobs = {k: j for k, j in self.jobs.items()
                     if j["status"] == "running" or (kind is not None and k[0] != kind)}

    def stats(self):
        from collections import Counter
        return dict(Counter(j["status"] for j in self.jobs.values()))


@ray.remote
def _run_job(coord, key, fn):
    def report(stage, payload):
        coord.report.remote(key, stage, payload)
    try:
        res = fn(report)
        ray.get(coord.finish.remote(key, result=res))
        return res
    except Exception as e:                                  # noqa: BLE001
        ray.get(coord.finish.remote(key, error=f"{type(e).__name__}: {e}"[:200]))
        raise


@dataclass
class Job:
    key: tuple
    orch: "ProbeOrchestrator"

    def done(self) -> bool:
        s = self.orch._state(self.key)
        return s is not None and s["status"] in ("done", "error")

    def stages(self) -> list:
        s = self.orch._state(self.key)
        return list(s["stages"]) if s else []

    def result(self, timeout: float | None = None) -> Any:
        deadline = None if timeout is None else time.time() + timeout
        while True:
            s = self.orch._state(self.key)
            if s and s["status"] == "done":
                return s["result"]
            if s and s["status"] == "error":
                raise RuntimeError(s["error"])
            if deadline and time.time() > deadline:
                raise TimeoutError(str(self.key))
            time.sleep(0.02)


class ProbeOrchestrator:
    def __init__(self, max_workers: int = 4):
        _ensure_ray()
        self.coord = _Coordinator.options(max_concurrency=8).remote()
        self.stats_local = {"submitted": 0, "coalesced": 0, "memo_hits": 0, "computed": 0}

    def _state(self, key):
        return ray.get(self.coord.state.remote(key))

    def submit(self, kind: str, epd: str, fn: Callable, on_event=None) -> Job:
        key = (kind, epd)
        self.stats_local["submitted"] += 1
        claim = ray.get(self.coord.try_claim.remote(key))
        if claim == "claimed":
            self.stats_local["computed"] += 1
            _run_job.remote(self.coord, key, fn)
        elif claim == "inflight":
            self.stats_local["coalesced"] += 1
        else:
            self.stats_local["memo_hits"] += 1
        job = Job(key, self)
        if on_event is not None:                            # poll-based event pump
            import threading

            def _pump():
                seen = 0
                while True:
                    s = self._state(key)
                    if s is None:
                        return
                    for stage, payload in s["stages"][seen:]:
                        on_event(key, stage, payload)
                    seen = len(s["stages"])
                    if s["status"] in ("done", "error"):
                        on_event(key, s["status"],
                                 s["result"] if s["status"] == "done" else s["error"])
                        return
                    time.sleep(0.03)
            threading.Thread(target=_pump, daemon=True).start()
        return job

    def wait(self, job: Job, timeout: float | None = None) -> Any:
        return job.result(timeout)

    def as_completed(self, jobs: list[Job], timeout: float | None = None):
        pending = list(jobs)
        while pending:
            for j in list(pending):
                if j.done():
                    pending.remove(j)
                    yield j
            time.sleep(0.02)

    def invalidate(self, kind: str | None = None):
        ray.get(self.coord.invalidate.remote(kind))

    def stats(self):
        d = dict(self.stats_local)
        d["coordinator"] = ray.get(self.coord.stats.remote())
        return d

    def shutdown(self):
        pass                                                # ray lifecycle owned by caller
