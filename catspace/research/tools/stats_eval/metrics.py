"""catspace/metrics.py -- inference-pipeline observability (Kaveh 2026-07-25: 'a breakdown
of different delays caused by each step in the ML inference pipeline... use some open
source observability framework'). Prometheus histograms per pipeline stage; the engine's
existing times-dict writers observe here too; /metrics on the assistant server, scraped by
Prometheus, panelled in Grafana (both in the compose stack). Degrades to no-ops when
prometheus_client is absent."""
from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram, generate_latest

    STAGE = Histogram("catspace_stage_seconds", "per-stage inference latency",
                      ["stage"], buckets=(.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10))
    REQS = Counter("catspace_requests_total", "requests", ["path"])

    def observe(stage: str, seconds: float):
        STAGE.labels(stage=stage).observe(seconds)

    def count(path: str):
        REQS.labels(path=path).inc()

    def latest() -> bytes:
        return generate_latest()
except Exception:                                            # pragma: no cover
    def observe(stage, seconds): pass
    def count(path): pass
    def latest() -> bytes: return b""
