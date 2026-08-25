"""Static checks for SILENT failure patterns -- bugs that produce correct-looking output.

Kaveh 2026-08-25: "make sure fixes are placed at the right layer of abstraction for all
similar users." A bug fixed only where it was found is not fixed; the knowledge lives in
one file's diff and the next instance is written next week. This module is the right
layer for the class of defects that a reviewer cannot see and a test does not catch,
because the code is CORRECT -- just quietly wrong about cost or scope.

Each check below exists because the pattern actually bit this repo, and each carries the
measurement that justifies caring.

Run it:
    .venv/bin/python -m catspace.research.tools.lint_patterns            # ratchet
    .venv/bin/python -m catspace.research.tools.lint_patterns --all      # full list
    .venv/bin/python -m catspace.research.tools.lint_patterns --baseline # re-baseline

The ratchet: known instances are recorded in lint_baseline.json and tolerated, so the
check can be adopted without a 92-site refactor, but any NEW instance fails. Fix a site
and re-baseline; the count only goes down.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path

BASELINE = Path(__file__).with_name("lint_baseline.json")


def _py_files():
    out = subprocess.run(["git", "ls-files", "*.py"], capture_output=True, text=True)
    return [f for f in out.stdout.split() if f.endswith(".py")]


def check_lazy_npz(path: str, tree: ast.AST):
    """np.load('.npz') returns a LAZY NpzFile: every ["key"] re-reads and re-decompresses
    the whole array. Subscripting one inside a loop, when the handle was opened outside
    it, is silently 10-40x slow.

    Measured 2026-08-25 in future_kind_labels: a 737 KB array decompressed 118,000 times
    per shard -- 115s for a phase that takes 3s once the read is hoisted.

    Fix: hoist `arr = z["key"]` above the loop (or use load_npz_eager).
    """
    # Scope PER FUNCTION. Collecting np.load assignments file-wide conflates two
    # different `data` variables in two different functions and flags code that is
    # already correct -- shards.py binds every array once, on purpose, and was
    # reported anyway. A lint people stop trusting is worse than no lint.
    scopes = [n for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module))]
    hits = []
    for scope in scopes:
        inner = {id(x) for n in ast.iter_child_nodes(scope)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 for x in ast.walk(n)}
        def in_scope(node):
            return id(node) not in inner
        loads: dict[str, list[int]] = {}
        for n in ast.walk(scope):
            if not in_scope(n):
                continue
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                fn_ = n.value.func
                nm = getattr(fn_, "attr", None) or getattr(fn_, "id", None)
                if nm == "load" and getattr(getattr(fn_, "value", None), "id", "") in ("np", "numpy"):
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            loads.setdefault(t.id, []).append(n.lineno)
            # a var REBOUND from anything else (e.g. materialised into a dict) is no
            # longer a lazy handle -- shards.py's `data = {k: npz[k] ...}` pattern
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id in loads:
                        loads.pop(t.id, None)
        if not loads:
            continue
        for loop in [n for n in ast.walk(scope)
                     if isinstance(n, (ast.For, ast.While)) and in_scope(n)]:
            lo = loop.lineno
            for sub in ast.walk(loop):
                if not isinstance(sub, ast.Subscript):
                    continue
                base = sub.value
                while isinstance(base, ast.Subscript):
                    base = base.value
                if not (isinstance(base, ast.Name) and base.id in loads):
                    continue
                key = sub.slice
                if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                    continue
                if all(a < lo for a in loads[base.id]):
                    hits.append((sub.lineno,
                                 f'{base.id}["{key.value}"] indexed in a loop'))
    # a subscript nested in two loops is visited once per enclosing loop; report the
    # SITE, not the number of loops around it
    return sorted(set(hits))


def check_sorted_prefix(path: str, tree: ast.AST):
    """Slicing a PREFIX off an index array that is sorted, to make a subsample.

    A prefix of a sorted index is the oldest shards only. When the data window is
    compositionally uneven this silently excludes an entire row-kind and its gauges read
    0.000 -- which is indistinguishable from converged. It cost a run's sf-spearman
    reading 0.000 while the sf training loss was healthy at 3.361.

    Fix: np.random.default_rng(seed).choice(idx, n, replace=False), fixed seed so the
    subsample stays comparable step to step.
    """
    hits = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Subscript) or not isinstance(n.slice, ast.Slice):
            continue
        name = getattr(n.value, "id", "")
        # index arrays in this repo are named te_i / tr_i / idx / rows / ii
        if name not in ("te_i", "tr_i", "hold_i", "val_i"):
            continue
        sl = n.slice
        if sl.lower is None and sl.upper is not None:      # x[:n] -- a prefix
            hits.append((n.lineno, f"{name}[:n] takes a PREFIX of a sorted index"))
    return hits


def check_main_guard_forward_ref(path: str, tree: ast.AST):
    """A function CALLED in `if __name__ == "__main__":` but DEFINED below it.

    Python runs the guard where it sits, so anything defined after it does not exist
    yet: NameError, at the end of a run rather than the start. Caught in losses.py on
    2026-08-24 (the whole loss suite died) and written again in test_resume.py on
    2026-08-25 -- twice in two days, by the same author, which is why it is a tool now
    and not a note.

    Fix: put the guard last, after every def.
    """
    hits = []
    defs = {n.name: n.lineno for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for n in ast.walk(tree):
        if not isinstance(n, ast.If):
            continue
        t = n.test
        is_guard = (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                    and t.left.id == "__name__"
                    and any(isinstance(c, ast.Constant) and c.value == "__main__"
                            for c in t.comparators))
        if not is_guard:
            continue
        for call in ast.walk(n):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                d = defs.get(call.func.id)
                if d is not None and d > n.lineno:
                    hits.append((call.lineno,
                                 f"{call.func.id}() is defined at line {d}, BELOW the "
                                 f"__main__ guard that calls it -> NameError"))
    return sorted(set(hits))


CHECKS = {
    "lazy-npz": check_lazy_npz,
    "sorted-prefix": check_sorted_prefix,
    "main-guard-forward-ref": check_main_guard_forward_ref,
}


def run() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {k: [] for k in CHECKS}
    for f in _py_files():
        if f.endswith("lint_patterns.py"):
            continue
        try:
            tree = ast.parse(Path(f).read_text())
        except Exception:
            continue
        for name, fn in CHECKS.items():
            for lineno, msg in fn(f, tree):
                found[name].append(f"{f}:{lineno}: {msg}")
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="list every instance")
    ap.add_argument("--baseline", action="store_true", help="record current as tolerated")
    a = ap.parse_args()

    found = run()
    if a.baseline:
        BASELINE.write_text(json.dumps({k: sorted(v) for k, v in found.items()}, indent=1))
        print(f"  baselined {sum(len(v) for v in found.values())} instances -> "
              f"{BASELINE.name}; only NEW ones fail from now on")
        return 0

    old = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    rc = 0
    for name, hits in found.items():
        known = set(old.get(name, []))
        new = [h for h in hits if h not in known]
        fixed = len(known) - len([h for h in hits if h in known])
        print(f"  {name:<14} {len(hits):>4} instances "
              f"({len(new)} NEW, {fixed} fixed since baseline)")
        for h in (hits if a.all else new):
            print(f"      {h}")
        if new:
            rc = 1
    print("  " + ("NEW instances found -- see above" if rc else
                  "no new instances; the ratchet holds"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
