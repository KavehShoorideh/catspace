"""WHERE A ROUND LIVES. Everything a round produces goes to data/derived/loop/round_NNN (git-ignored, like all of
data/derived); the reviewable record -- the ledger, the frozen inputs, the promoted ideas, the measurements, the gate
-- goes to artifacts/loop/round_NNN, and the basis to artifacts/loop/basis.jsonl, both committed: a round is one
commit and a rollback is a revert (the plan put the basis under data/derived; that directory is ignored by git, so the
basis moved to where a commit can carry it). Tests point both roots elsewhere with CATSPACE_LOOP_DATA /
CATSPACE_LOOP_RECORD."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from catspace.io import paths


def data_root() -> Path:
    p = Path(os.environ.get("CATSPACE_LOOP_DATA") or (paths.derived_dir() / "loop"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def record_root() -> Path:
    p = Path(os.environ.get("CATSPACE_LOOP_RECORD") or (paths.REPO_ROOT / "artifacts" / "loop"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def round_dir(n: int) -> Path:
    d = data_root() / f"round_{n:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def record_dir(n: int) -> Path:
    d = record_root() / f"round_{n:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())


def code_version() -> str:
    """The git commit of the code, with +dirty when the loop's code or the Lean package has uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "catspace", "lean"], cwd=paths.REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"
