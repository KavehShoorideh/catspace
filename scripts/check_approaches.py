#!/usr/bin/env python
"""check_approaches.py -- every approach on disk is registered, and every registered
approach exists on disk.

An `approaches.md` that drifts from the tree is worse than no registry at all: it is the
thing a newcomer (or an agent) reads to decide what exists. This runs in CI and in the
P13 verification checklist.

    .venv/bin/python scripts/check_approaches.py

Exit 0 = in sync. Exit 1 = drift, printed per registry.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from catspace import registry
from catspace.io.paths import REPO_ROOT

WRAPPER = REPO_ROOT / "catspace"
COMPONENTS_DIR = WRAPPER / "research" / "components"

# Files/dirs under catspace/approaches/ that are shared machinery, not an approach.
NOT_AN_APPROACH = {"__pycache__"}


def registered(md: Path) -> set[str]:
    """Approach names are the level-2 headings of an approaches.md."""
    if not md.exists():
        raise SystemExit(f"missing registry: {md.relative_to(REPO_ROOT)}")
    return {m.group(1).strip()
            for m in re.finditer(r"^## +(\S+)\s*$", md.read_text(), re.M)}


def on_disk(base: Path, marker: str | None) -> set[str]:
    """Directories under `base` that are approaches. `marker` (if given) is a path that
    must exist inside the directory for it to count."""
    if not base.is_dir():
        return set()
    return {p.name for p in base.iterdir()
            if p.is_dir() and p.name not in NOT_AN_APPROACH
            and not p.name.startswith(".")
            and (marker is None or (p / marker).exists())}


def compare(label: str, md: Path, disk: set[str]) -> list[str]:
    reg = registered(md)
    problems = []
    for name in sorted(disk - reg):
        problems.append(f"{label}: {name!r} exists on disk but is not in "
                        f"{md.relative_to(REPO_ROOT)}")
    for name in sorted(reg - disk):
        problems.append(f"{label}: {md.relative_to(REPO_ROOT)} lists {name!r} but there is "
                        f"no such directory")
    if not problems:
        print(f"ok  {label:22s} {len(reg)} approaches in sync")
    return problems


def main() -> int:
    problems: list[str] = []

    # Wrapper-level end-to-end approaches: a config has config.py, a harness does not, so
    # any directory with an __init__.py or a config.py counts.
    disk = {p.name for p in (WRAPPER / "approaches").iterdir()
            if p.is_dir() and p.name not in NOT_AN_APPROACH
            and ((p / "config.py").exists() or (p / "__init__.py").exists())}
    problems += compare("approaches", WRAPPER / "approaches.md", disk)

    for component in registry.COMPONENTS:
        base = COMPONENTS_DIR / component / "approaches"
        problems += compare(component, COMPONENTS_DIR / component / "approaches.md",
                            on_disk(base, "src/__init__.py"))

    # Every config named in the wrapper registry must actually resolve, and every component
    # slot it names must exist -- the registry is only useful if it is executable.
    from catspace import approaches as e2e
    for name in sorted(e2e.list_configs()):
        mod = e2e.load(name)
        cfg = getattr(mod, "CONFIG", None)
        if cfg is None:
            problems.append(f"approaches: {name!r} has config.py but no CONFIG")
            continue
        try:
            cfg.validate()
        except (LookupError, ValueError) as exc:
            problems.append(f"approaches: {name!r} CONFIG.validate() failed: {exc}")
        else:
            print(f"ok  config {name:16s} slots resolve")

    for p in problems:
        print(f"FAIL {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
