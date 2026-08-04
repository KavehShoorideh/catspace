"""catspace.research.infra -- everything around the engine, not in it (Kaveh 2026-07-30):
preemption safety, full-state checkpoints, run observability, cloud scaffolding.
"""
from catspace.research.infra.checkpoint import (latest_resumable, load_training_state,   # noqa: F401
                              save_training_state)
from catspace.research.infra.observability.run_metrics import RunLogger                  # noqa: F401
from catspace.research.infra.preempt import PreemptGuard                                 # noqa: F401
