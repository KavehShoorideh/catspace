"""End-to-end approaches: named wirings of the four components.

An end-to-end approach picks one planner, one-or-more searches, one-or-more
encoders and one-or-more memories, and says how they connect. This is the layer
`catspace.deployment` serves -- deployment never reaches into a component itself.

    from catspace.approaches import load
    engine = load("mate_finisher").build()
"""
from __future__ import annotations

import importlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def list_configs() -> tuple[str, ...]:
    return tuple(sorted(
        p.name for p in _HERE.iterdir()
        if p.is_dir() and (p / "config.py").exists()))


def load(name: str):
    """Import an end-to-end config module (must expose CONFIG and build())."""
    if not (_HERE / name / "config.py").exists():
        raise LookupError(f"no end-to-end approach {name!r}; available: {list_configs()}")
    return importlib.import_module(f"catspace.approaches.{name}.config")
