"""catspace.research.catspace/research/infra/checkpoint.py -- FULL training-state checkpoints (model + optimizer +
scheduler + step + metadata), atomic write, resumable. The training-standards
rule (ckpt ladders, no overwrites without metadata) plus the preemption
contract need optimizer state — a bare state_dict cannot resume correctly.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


def save_training_state(path, model, opt=None, sched=None, step=0, cfg=None,
                        meta=None):
    tmp = str(path) + ".tmp"
    torch.save({"state_dict": model.state_dict(),
                "opt": opt.state_dict() if opt else None,
                "sched": sched.state_dict() if sched else None,
                "step": int(step), "cfg": cfg or {}, "meta": meta or {}}, tmp)
    os.replace(tmp, path)                             # atomic (same filesystem)
    return path


def load_training_state(path, model, opt=None, sched=None, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    if opt is not None and ck.get("opt"):
        opt.load_state_dict(ck["opt"])
    if sched is not None and ck.get("sched"):
        sched.load_state_dict(ck["sched"])
    return int(ck.get("step", 0)), ck


def latest_resumable(out_prefix: str):
    """<out>_latest.pt if it exists and carries optimizer state, else None."""
    p = Path(f"{out_prefix}_latest.pt")
    if not p.exists():
        return None
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        return str(p) if ck.get("opt") else None
    except Exception:
        return None
