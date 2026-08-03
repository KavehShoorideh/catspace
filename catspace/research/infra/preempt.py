"""catspace.research.catspace/research/infra/preempt.py -- spot-instance/preemption safety for training loops.

Contract (the standard one: Run:ai / SageMaker / cluster schedulers): on SIGTERM
the process checkpoints and exits 0. Training loops do:

    guard = PreemptGuard()
    for step in ...:
        ...train...
        if guard.should_stop or step % ckpt_every == 0:
            save_training_state(...)
            if guard.should_stop:
                print("PREEMPT: checkpointed at step", step); sys.exit(0)

SIGINT gets the same graceful treatment (first Ctrl-C = checkpoint+quit; second
= immediate KeyboardInterrupt).
"""
from __future__ import annotations

import signal


class PreemptGuard:
    def __init__(self, signals=(signal.SIGTERM, signal.SIGINT)):
        self.should_stop = False
        self._seen = 0
        for s in signals:
            try:
                signal.signal(s, self._handler)
            except (ValueError, OSError):
                pass                                  # non-main thread / platform

    def _handler(self, signum, frame):
        self._seen += 1
        if self._seen >= 2 and signum == signal.SIGINT:
            raise KeyboardInterrupt
        self.should_stop = True
