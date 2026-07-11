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


def lichess_dir() -> Path:
    p = data_dir() / "lichess"
    p.mkdir(parents=True, exist_ok=True)
    return p


def shards_dir() -> Path:
    p = data_dir() / "shards"
    p.mkdir(parents=True, exist_ok=True)
    return p


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
