"""Repoint `from experiments.X import ...` at wherever X actually landed, and delete the
sys.path.insert prelude that made those imports work in the first place.

The map is derived from git's own rename detection, not from a hand-written table, so it
cannot disagree with what was actually moved.

Note on importability: an approach's experiments/ directory deliberately has no
__init__.py -- that is what keeps it out of setuptools' find() and out of the wheel. It is
still importable, as a PEP 420 namespace portion inside the regular package above it, so
research scripts can go on importing each other the way a lab notebook does.

Throwaway tooling for the 2026-08-03 restructure; delete once the migration lands.

    python tools/_migration/rewrite_experiment_imports.py            # report
    python tools/_migration/rewrite_experiment_imports.py --apply
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".venv", ".git", ".claude", "node_modules", "__pycache__", ".dvc", "mlruns", "_migration"}
# Historical docs keep their old paths on purpose; they get a banner instead (P12).
SKIP_SUFFIXES = {".md"}

SYSPATH = re.compile(
    r"^sys\.path\.insert\(0,\s*str\(Path\(__file__\)\.resolve\(\)\.parents\[\d+\]\)\)\n",
    re.M)


def moved_modules() -> dict[str, str]:
    """old dotted module -> new dotted module, from `git status` rename records."""
    out = subprocess.run(["git", "status", "--porcelain=1", "-z", "--find-renames"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    fields = out.split("\0")
    mapping: dict[str, str] = {}
    i = 0
    while i < len(fields):
        rec = fields[i]
        if not rec:
            i += 1
            continue
        status, _, path = rec[:2], rec[2], rec[3:]
        if status[0] == "R":
            new, old = path, fields[i + 1]        # porcelain -z: R gives NEW then OLD
            i += 2
            if old.startswith("experiments/") and old.endswith(".py"):
                mapping[dotted(old)] = dotted(new)
        else:
            i += 1
    mapping.update(SUPPLEMENT)
    return mapping


# Moved in the P8 commit, so they are already in history rather than in `git status`.
SUPPLEMENT = {
    "experiments.arena_real": "catspace.approaches.gauntlet_harness.experiments.arena_real",
    "experiments.play_vs_maia": "catspace.approaches.gauntlet_harness.experiments.play_vs_maia",
    "experiments.play_traced": "catspace.approaches.gauntlet_harness.experiments.play_traced",
    "experiments.bootstrap_mate_engine": "catspace.approaches.bootstrap_mate.src.engine",
}


def dotted(path: str) -> str:
    return path[:-3].replace("/", ".")


def files() -> list[Path]:
    out = []
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        # scripts/ holds symlinks whose targets just moved; P11 recreates them. Rewriting
        # through a symlink would edit the target twice anyway.
        if p.is_symlink() or not p.is_file():
            continue
        out.append(p)
    return sorted(out)


def main() -> int:
    apply = "--apply" in sys.argv
    mapping = moved_modules()
    print(f"{len(mapping)} moved modules")

    # Longest first so experiments.viz.foo is not clobbered by experiments.viz.
    keys = sorted(mapping, key=len, reverse=True)
    patterns = [(re.compile(rf"(?<![\w.]){re.escape(k)}(?![\w])"), mapping[k]) for k in keys]

    unresolved: Counter[str] = Counter()
    changed = 0
    for f in files():
        txt = orig = f.read_text()
        txt = SYSPATH.sub("", txt)
        for rx, new in patterns:
            txt = rx.sub(new, txt)
        for m in re.finditer(r"(?<![\w.])experiments\.[a-z0-9_.]+", txt):
            unresolved[m.group(0)] += 1
        if txt != orig:
            changed += 1
            if apply:
                f.write_text(txt)

    print(f"{'rewrote' if apply else 'would rewrite'} {changed} files")
    if unresolved:
        print("\nSTILL REFERENCING experiments.* (need manual attention):")
        for k, n in unresolved.most_common():
            print(f"  {n:4d}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
