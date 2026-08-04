"""catspace.research.infra contract tests: preemption flag, full-state checkpoint round-trip,
metrics JSONL."""
import json
import os
import signal

import numpy as np
import torch
import torch.nn as nn

from catspace.research.infra import (PreemptGuard, RunLogger, latest_resumable,
                   load_training_state, save_training_state)


def test_preempt_guard_sets_flag():
    g = PreemptGuard(signals=(signal.SIGUSR1,))
    assert not g.should_stop
    os.kill(os.getpid(), signal.SIGUSR1)
    assert g.should_stop


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = nn.Linear(4, 3)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)
    x = torch.randn(8, 4)
    for _ in range(3):
        opt.zero_grad(); m(x).sum().backward(); opt.step(); sched.step()
    p = tmp_path / "run_latest.pt"
    save_training_state(p, m, opt, sched, step=3, cfg={"d": 4}, meta={"k": "v"})
    m2 = nn.Linear(4, 3)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, 10)
    step, ck = load_training_state(p, m2, opt2, sched2)
    assert step == 3 and ck["cfg"]["d"] == 4
    assert torch.allclose(m.weight, m2.weight)
    assert sched2.last_epoch == sched.last_epoch
    assert opt2.state_dict()["state"].keys() == opt.state_dict()["state"].keys()
    assert latest_resumable(str(tmp_path / "run")) == str(p)
    assert latest_resumable(str(tmp_path / "missing")) is None


def test_runlogger_jsonl_and_timers(tmp_path):
    log = RunLogger(str(tmp_path / "run"))
    with log.timer("work"):
        np.zeros(10)
    row = log.log(step=100, loss=1.5)
    assert row["loss"] == 1.5 and "work_s" in row
    rows = [json.loads(ln) for ln in open(tmp_path / "run_metrics.jsonl")]
    assert rows[0]["step"] == 100 and rows[0]["wall_s"] >= 0
