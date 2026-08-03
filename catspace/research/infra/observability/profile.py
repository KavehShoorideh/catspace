"""catspace.research.catspace/research/infra/observability/profile.py -- bottleneck measurement, off the shelf.

Two layers:
  profile_block  : torch.profiler wrapper (CPU+MPS ops) -> prints the top-K ops
                   by self time and writes a chrome trace (open in
                   chrome://tracing or Perfetto). For code you can edit.
  py-spy         : sampling profiler for RUNNING processes -- no code changes,
                   near-zero overhead. The tool for "what is this run doing":
                       py-spy top --pid <pid>          live top-functions view
                       py-spy dump --pid <pid>         stack snapshot (hangs!)
                       py-spy record -o prof.svg --pid <pid> --duration 30
                   Installed in the venv; works on the detached launch.sh runs.

RunLogger timers (run_metrics.py) remain the always-on coarse layer; this file
is the zoom lens.
"""
from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def profile_block(name="block", trace_path="", top=15, row_limit_chars=90):
    import torch
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU]
    with profile(activities=acts, record_shapes=False, with_stack=False) as prof:
        yield prof
    tbl = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=top)
    print(f"[profile:{name}]\n" + "\n".join(
        ln[:row_limit_chars] for ln in tbl.splitlines()))
    if trace_path:
        prof.export_chrome_trace(trace_path)
        print(f"[profile:{name}] chrome trace -> {trace_path}")
