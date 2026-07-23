"""catspace/engine/orchestrator.py -- PROBE ORCHESTRATION (Kaveh 2026-07-25: 'shoot off
multiple probe requests and they stream in results in parallel as they pass certain
milestones... yet they don't double calculate anything: global memoization, waiting system
on pending results and notifications of result completions, indexed by position').

Single-flight async execution, keyed by (kind, position-epd):
  - submit() with a key already DONE      -> handle resolves instantly from the memo
  - submit() with a key already IN FLIGHT -> the request COALESCES onto the running job
    (never double-calculate); its callbacks attach to the same stream
  - long jobs call report(stage, payload) as they pass milestones -> every subscriber
    gets (key, stage, payload) events as they happen; 'done' is the final event
  - wait()/result() block on a condition; as_completed() streams finished handles

Threads (torch forwards release the GIL; probe workloads are net/geometry-bound). This is
the IN-PROCESS primitive; cross-process sharing stays on the file/sqlite layer (banks,
experience store) by design."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Job:
    key: tuple
    status: str = "pending"                    # pending -> running -> done | error
    result: Any = None
    error: str | None = None
    stages: list = field(default_factory=list)
    done_evt: threading.Event = field(default_factory=threading.Event)
    callbacks: list = field(default_factory=list)


class ProbeOrchestrator:
    def __init__(self, max_workers: int = 4):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._jobs: dict[tuple, Job] = {}      # pending + running + done (memo)
        self.stats = {"submitted": 0, "coalesced": 0, "memo_hits": 0, "computed": 0}

    def submit(self, kind: str, epd: str, fn: Callable[[Callable], Any],
               on_event: Callable | None = None) -> Job:
        """fn(report) -> result; call report(stage, payload) at milestones.
        Identical (kind, epd) submissions coalesce; finished ones hit the memo."""
        key = (kind, epd)
        with self._lock:
            self.stats["submitted"] += 1
            job = self._jobs.get(key)
            if job is not None:
                if job.status == "done":
                    self.stats["memo_hits"] += 1
                    if on_event:
                        on_event(key, "done", job.result)
                else:
                    self.stats["coalesced"] += 1
                    if on_event:
                        job.callbacks.append(on_event)
                        for stage, payload in job.stages:   # replay milestones already passed
                            on_event(key, stage, payload)
                return job
            job = Job(key)
            if on_event:
                job.callbacks.append(on_event)
            self._jobs[key] = job

        def _run():
            def report(stage, payload):
                with self._lock:
                    job.stages.append((stage, payload))
                    cbs = list(job.callbacks)
                for cb in cbs:
                    cb(key, stage, payload)
            job.status = "running"
            try:
                job.result = fn(report)
                job.status = "done"
                self.stats["computed"] += 1
            except Exception as e:                          # noqa: BLE001
                job.error = f"{type(e).__name__}: {e}"[:200]
                job.status = "error"
            with self._lock:
                cbs = list(job.callbacks)
            for cb in cbs:
                cb(key, "done" if job.status == "done" else "error",
                   job.result if job.status == "done" else job.error)
            job.done_evt.set()
        self._pool.submit(_run)
        return job

    def wait(self, job: Job, timeout: float | None = None) -> Any:
        job.done_evt.wait(timeout)
        if job.status == "error":
            raise RuntimeError(job.error)
        return job.result

    def as_completed(self, jobs: list[Job], timeout: float | None = None):
        pending = list(jobs)
        while pending:
            for j in list(pending):
                if j.done_evt.wait(0.01):
                    pending.remove(j)
                    yield j

    def invalidate(self, kind: str | None = None):
        """drop memoized results (e.g., after a bank/field change made them stale)."""
        with self._lock:
            self._jobs = {k: j for k, j in self._jobs.items()
                          if j.status != "done" or (kind is not None and k[0] != kind)}

    def shutdown(self):
        self._pool.shutdown(wait=False)
