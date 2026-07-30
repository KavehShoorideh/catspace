"""infra -- everything around the engine, not in it (Kaveh 2026-07-30):
preemption safety, full-state checkpoints, run observability, cloud scaffolding.
"""
from infra.checkpoint import (latest_resumable, load_training_state,   # noqa: F401
                              save_training_state)
from infra.observability.run_metrics import RunLogger                  # noqa: F401
from infra.preempt import PreemptGuard                                 # noqa: F401
