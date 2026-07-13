"""io/paths.py — repo-rooted, env-overridable data/artifact directories.

Kills the hardcoded /home/claude/toykrk/ and /mnt/user-data/outputs/ paths
(from an old sandbox) and the CWD-dependent bare filenames that broke
cross-module artifact handoff on other machines.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    p = Path(os.environ.get("LCP_DATA", REPO_ROOT / "data"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def derived_dir() -> Path:
    p = data_dir() / "derived"
    p.mkdir(parents=True, exist_ok=True)
    return p


def generated_dir() -> Path:
    p = REPO_ROOT / "artifacts" / "generated"
    p.mkdir(parents=True, exist_ok=True)
    return p


def experiments_dir() -> Path:
    """Structured experiment_report.py JSON records -- small and valuable
    (a research history, like JOURNAL.md), NOT in .gitignore unlike
    artifacts/generated/'s regenerable viz output."""
    p = REPO_ROOT / "artifacts" / "experiments"
    p.mkdir(parents=True, exist_ok=True)
    return p


def lichess_dir() -> Path:
    p = data_dir() / "lichess"
    p.mkdir(parents=True, exist_ok=True)
    return p


def shards_dir() -> Path:
    p = data_dir() / "shards"
    p.mkdir(parents=True, exist_ok=True)
    return p


def newest_shard_dir() -> Path:
    """Most recently modified shard directory under data/shards -- the default
    dataset for training/eval drivers."""
    dirs = [p for p in shards_dir().iterdir() if p.is_dir() and list(p.glob("shard_*.npz"))]
    if not dirs:
        raise SystemExit("no shard dirs under data/shards -- run experiments/build_lichess_shards.py first")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def save_array(name: str, arr, sub: str = "derived") -> Path:
    import numpy as np
    base = derived_dir() if sub == "derived" else data_dir() / sub
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{name}.npy"
    np.save(path, arr)
    return path


def load_array(name: str, sub: str = "derived"):
    import numpy as np
    base = derived_dir() if sub == "derived" else data_dir() / sub
    return np.load(base / f"{name}.npy")
