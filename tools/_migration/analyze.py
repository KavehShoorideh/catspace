"""Pre-migration inventory: import hubs, path literals, and sys.path hacks.

Throwaway tooling for the 2026-08-03 restructure; delete once the migration lands.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP = {".venv", ".git", "node_modules", "__pycache__", ".dvc", "mlruns", "data", "artifacts"}

PATH_LITERAL = re.compile(r'["\'](?:\./)?(data/|artifacts/|docs/|maia2_models/|mlflow\.db|mlruns/)[^"\']*["\']')
SYSPATH = re.compile(r"sys\.path\.insert")


def py_files() -> list[Path]:
    out = []
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP for part in p.relative_to(ROOT).parts):
            continue
        out.append(p)
    return sorted(out)


def main() -> None:
    files = py_files()
    # who imports whom, restricted to first-party top-level packages
    first_party = {"catspace", "experiments", "tools", "infra", "contrib", "tests"}
    importers: dict[str, set[str]] = defaultdict(set)
    modcount: Counter[str] = Counter()
    syspath_files: list[str] = []
    path_lit: dict[str, list[str]] = defaultdict(list)

    for f in files:
        rel = str(f.relative_to(ROOT))
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if SYSPATH.search(text):
            syspath_files.append(rel)
        for m in PATH_LITERAL.finditer(text):
            path_lit[rel].append(m.group(0).strip("\"'"))
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            for mod in mods:
                top = mod.split(".")[0]
                if top not in first_party:
                    continue
                modcount[mod] += 1
                importers[mod].add(rel)

    exp_hubs = {m: len(v) for m, v in importers.items()
                if m.startswith("experiments") and len(v) >= 2}
    report = {
        "n_py_files": len(files),
        "n_syspath_files": len(syspath_files),
        "syspath_files": syspath_files,
        "n_files_with_path_literals": len(path_lit),
        "path_literals": path_lit,
        "experiments_import_hubs": dict(sorted(exp_hubs.items(), key=lambda kv: -kv[1])),
        "catspace_module_usage": dict(sorted(
            ((m, len(v)) for m, v in importers.items() if m.startswith("catspace")),
            key=lambda kv: -kv[1])),
    }
    out = ROOT / "tools" / "_migration" / "inventory.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=False))
    print(f"py files: {len(files)}")
    print(f"sys.path.insert files: {len(syspath_files)}")
    print(f"files with hardcoded data/artifact path literals: {len(path_lit)}")
    print(f"\nexperiments/ import hubs (imported by >=2 files):")
    for m, n in list(report["experiments_import_hubs"].items())[:40]:
        print(f"  {n:3d}  {m}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
