"""PLAN 79.37 PART A: THE RULES' SHARE OF A DECISION (SYSTEM_SPEC 11, "The crux of layer 2 ... (ii) The target should be set by a measurement of what share of a decision's time the rules take"; ledger rows 45, 44; piece: rules).
A decision is one call of the wake (`wake_decide`, as the matches call it) at 0.2 s, with the config the matches played (abcd), over a FIXED set of positions: decisions of the wake logged in the match of plan 79.36 (set A), chosen by a seeded
shuffle, one position once. The rules are what layer 2 gives the layers above: the kernel's answers (`Game for KernelGame`: moves, apply, outcome, mover, key, move_name, captured, taken_at, changed, pass, base_facts, units) and the outside
library under them (shakmaty). Nothing on the engine's path is changed:
  (1) THE SHARE comes from a sampling profiler already on the machine (`sample`, 1 ms): the call tree of the process that makes the decisions, one thread; the share of a call is the samples under its frames (its outermost frame on each stack) over the
      samples under `wake_decide`. The shared module of the machine is read, never rebuilt.
  (2) THE CALLS A DECISION come from a PRIVATE build of the binding (in the scratchpad, never in the repository: the same sources with one counter added in each question of `Game for KernelGame`, and one function that reads them), put on PYTHONPATH
      before the shared module; the program prints the path of the module it imported. The same sampling is run on it too, so the two shares can be read against each other.
INTERFACE: `positions(logs, count, seed) -> [fen]`; `parse_sample(path) -> Tree`; `shares(tree) -> dict`; the commands `positions`, `run` (the process that makes the decisions; prints `pid` and `ready`), `profile` (starts `run`, samples it, prints
the OBS lines). Called by the command line only.
  PYTHONPATH=<worktree>[:<private module folder>] LCP_DATA=<main>/data .venv/bin/python scripts/rules_share_print.py profile --logs <the match's folder of logs> [--seconds 40] [--tag name]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from catspace.io import paths

OUT = ("rules_share_79_37",)
SEC = 0.2
CONFIG = "abcd"
RULES_CRATES = ("cs_game", "shakmaty")                      # a frame is a rules frame when the crate that DEFINES the function is layer 2's or the outside library's (a type argument naming them does not count)


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


def positions(logs: Path, count: int, seed: int) -> list:
    """`count` distinct positions of the wake's logged decisions in `logs` (the match's game logs), by a seeded shuffle of the sorted list."""
    seen = []
    for f in sorted(Path(logs).glob("*.jsonl.gz")):
        with gzip.open(f, "rt") as h:
            for line in h:
                r = json.loads(line)
                if r.get("kind") == "decision" and isinstance(r.get("wake"), dict) and r.get("position"):
                    seen.append(r["position"])
    seen = sorted(set(seen))
    random.Random(seed).shuffle(seen)
    return seen[:count]


# ------------------------------------------------------------------------------------------------------------------------------------------ the call tree of `sample`

class Node:
    __slots__ = ("sym", "n", "kids", "col")

    def __init__(self, sym: str, n: int, col: int):
        self.sym, self.n, self.col, self.kids = sym, n, col, []


NODE = re.compile(r"^([\s+!:|]*?)(\d+) (\S+)\s+\(in ([^)]*)\)")


def parse_sample(path: Path) -> list:
    """The roots of the call tree (one per thread) of a `sample` report."""
    lines = Path(path).read_text().splitlines()
    start = lines.index("Call graph:")
    end = next(i for i, ln in enumerate(lines) if ln.startswith("Total number in stack"))
    roots: list = []
    stack: list = []
    for ln in lines[start + 1:end]:
        m = NODE.match(ln)
        if not m:
            continue
        col, n, sym = len(m.group(1)), int(m.group(2)), m.group(3)
        node = Node(re.sub(r"\.llvm\.\d+", "", sym), n, col)
        while stack and stack[-1].col >= col:
            stack.pop()
        (stack[-1].kids if stack else roots).append(node)
        stack.append(node)
    return roots


def segments(sym: str) -> list:
    """The length-prefixed names of a Rust v0 symbol, in order (a name is a number n and then n characters that begin with a letter)."""
    sym = re.sub(r"Cs[0-9A-Za-z]+_", "C", sym)               # the hash that names a crate
    sym = re.sub(r"B[0-9a-zA-Z]*_", "", sym)                  # a back reference
    sym = re.sub(r"(?<=[A-Z])s[0-9a-zA-Z]*_", "", sym)        # an impl's disambiguator
    out, i = [], 0
    while i < len(sym):
        m = re.compile(r"(\d+)([A-Za-z])").match(sym, i)
        if m:
            n = int(m.group(1))
            out.append(sym[m.end(1):m.end(1) + n])
            i = m.end(1) + n
        else:
            i += 1
    return out


def readable(sym: str) -> str:
    """A short name of a Rust v0 symbol: the crate and its last two names (a method of the kernel by `KernelGame::<name>`)."""
    m = re.search(r"4Game(\d+)(\w+)$", sym)
    if m and "KernelGame" in sym:
        return "KernelGame::" + m.group(2)[:int(m.group(1))]
    seg = segments(sym)
    return "::".join(seg[:6]) if seg else sym[:60]


DEFINED_IN = re.compile(r"C[s]?[0-9A-Za-z]*_(\d+)([A-Za-z_][A-Za-z0-9_]*)")


def crate_of(sym: str) -> str:
    """The crate that defines the function a Rust v0 symbol names: the first crate in its path (the crates of the type arguments come after)."""
    m = DEFINED_IN.search(sym)
    return m.group(2)[:int(m.group(1))] if m else ""


def is_rules(sym: str) -> bool:
    return sym.startswith("_R") and crate_of(sym) in RULES_CRATES


def shares(roots: list) -> dict:
    """From the tree: the samples under `wake_decide`, the samples under a rules frame (any), the inclusive samples of each rules call (its outermost frames only), and the self samples of the rules' leaves."""
    out = {"decide": 0, "rules": 0, "calls": Counter(), "leaf": Counter(), "parent": Counter()}

    def walk(node: Node, under_decide: bool, rules_above: bool, seen: set, parent: str = ""):
        if "wake_decide" in node.sym and "pymethod" not in node.sym and not under_decide:
            under_decide = True
            out["decide"] += node.n
        mine = is_rules(node.sym)
        kid_n = sum(k.n for k in node.kids)
        self_n = node.n - kid_n
        if under_decide:
            if mine and not rules_above:
                out["rules"] += node.n
            if mine and node.sym not in seen:
                out["calls"][node.sym] += node.n
            if (mine or rules_above) and self_n > 0:
                out["leaf"][node.sym] += self_n
                out["parent"][(node.sym, parent)] += self_n
        for k in node.kids:
            walk(k, under_decide, rules_above or mine, seen | ({node.sym} if mine else set()), node.sym)

    for r in roots:
        walk(r, False, False, set())
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ the process that makes the decisions

def cmd_run(a) -> int:
    import cs_engine
    from catspace.incubator.analyst.discovery import whole_79_16 as W16
    print(f"module {cs_engine.__file__}", flush=True)
    W16.init_a()
    fens = json.loads((out_root(a.tag) / "positions.json").read_text())
    counters = getattr(cs_engine, "rules_counters", None)
    print(f"pid {os.getpid()}", flush=True)
    print("ready", flush=True)
    t_end = time.perf_counter() + a.seconds
    walls, per, n = [], [], 0
    last = dict(counters()) if counters else None
    while time.perf_counter() < t_end:
        for fen in fens:
            if time.perf_counter() >= t_end:
                break
            out = W16.ask(fen, SEC, W16.CONFIGS[CONFIG])
            walls.append(out["wall"])
            n += 1
            if counters:
                now = dict(counters())
                per.append({k: now[k] - last[k] for k in now})
                last = now
    res = {"decisions": n, "wall_mean": sum(walls) / max(1, n), "wall_sum": sum(walls), "positions": len(fens), "module": cs_engine.__file__}
    if per:
        res["calls_mean"] = {k: sum(p[k] for p in per) / n for k in per[0]}
        res["calls_min"] = {k: min(p[k] for p in per) for k in per[0]}
        res["calls_max"] = {k: max(p[k] for p in per) for k in per[0]}
    (out_root(a.tag) / f"run_{a.name}.json").write_text(json.dumps(res))
    print("done " + json.dumps(res), flush=True)
    return 0


def cmd_positions(a, say) -> list:
    fens = positions(Path(a.logs), a.count, a.seed)
    (out_root(a.tag) / "positions.json").write_text(json.dumps(fens))
    say(f"OBS 79.37 A positions: {len(fens)} positions of the wake's logged decisions in {Path(a.logs).name} (the match of plan 79.36, set A), a seeded shuffle (seed {a.seed}) of the sorted distinct list; each asked afresh at {SEC} s with the config {CONFIG}")
    return fens


def cmd_profile(a, say) -> int:
    root = out_root(a.tag)
    if a.logs and (not (root / "positions.json").exists() or a.fresh):
        cmd_positions(a, say)
    say(f"OBS 79.37 A load before: {os.getloadavg()}")
    env = dict(os.environ, OMP_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    proc = subprocess.Popen([sys.executable, str(Path(__file__)), "run", "--tag", a.tag, "--name", a.name, "--seconds", str(a.seconds)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    pid, module = None, None
    for ln in proc.stdout:
        if ln.startswith("module "):
            module = ln.split(" ", 1)[1].strip()
        if ln.startswith("pid "):
            pid = int(ln.split()[1])
        if ln.startswith("ready"):
            break
    say(f"OBS 79.37 A module: the process imported cs_engine from {module}")
    time.sleep(2.0)
    sample_path = root / f"sample_{a.name}.txt"
    subprocess.run(["sample", str(pid), str(a.sample), "-file", str(sample_path)], capture_output=True)
    final, tail = "", []
    for ln in proc.stdout:
        tail.append(ln)
        if ln.startswith("done "):
            final = ln[5:]
    proc.wait()
    if not final:
        raise SystemExit("the process that makes the decisions did not finish: " + "".join(tail[-12:]))
    run = json.loads(final)
    return report(a, run, sample_path, say)


def report(a, run: dict, sample_path: Path, say) -> int:
    say(f"OBS 79.37 A {a.name}: {run['decisions']} decisions in {run['wall_sum']:.1f} s of the wake's own wall, {run['wall_mean']:.4f} s a decision on average, over {run['positions']} positions (one thread; load after {os.getloadavg()})")
    roots = parse_sample(sample_path)
    s = shares(roots)
    d = max(1, s["decide"])
    say(f"OBS 79.37 A {a.name} sample: {a.sample} s at 1 ms; {s['decide']} samples under wake_decide (the decisions of the sampled seconds); the rules (any frame of the kernel's own code or of the library under it) {s['rules']} = {100 * s['rules'] / d:.1f} per cent of them")
    kernel = [(sym, n) for sym, n in s["calls"].items() if crate_of(sym) == "cs_game"]
    for sym, n in sorted(kernel, key=lambda x: -x[1]):
        say(f"OBS 79.37 A {a.name} call {readable(sym)}: {n} samples = {100 * n / d:.1f} per cent of the decision (inclusive of what it calls)")
    lib = [(readable(sym), n) for sym, n in s["calls"].items() if crate_of(sym) == "shakmaty"]
    merged: Counter = Counter()
    for nm, n in lib:
        merged[nm] = max(merged[nm], n)
    for nm, n in merged.most_common(8):
        say(f"OBS 79.37 A {a.name} library {nm}: {n} samples = {100 * n / d:.1f} per cent of the decision (inclusive)")
    leaf = Counter()
    for sym, n in s["leaf"].items():
        leaf[readable(sym)] += n
    for nm, n in leaf.most_common(8):
        say(f"OBS 79.37 A {a.name} leaf {nm}: {n} samples = {100 * n / d:.1f} per cent of the decision (the function's own time)")
    top = leaf.most_common(1)
    if top:
        for (sym, par), n in sorted(s["parent"].items(), key=lambda x: -x[1]):
            if readable(sym) == top[0][0]:
                say(f"OBS 79.37 A {a.name} leaf {top[0][0]} called by {readable(par)}: {n} samples = {100 * n / d:.1f} per cent of the decision")
                break
    if "calls_mean" in run:
        wall = run["wall_mean"]
        for k, c in run["calls_mean"].items():
            say(f"OBS 79.37 A {a.name} calls {k}: {c:.1f} a decision (least {run['calls_min'][k]}, most {run['calls_max'][k]})")
        sym_share = {readable(sym): n for sym, n in kernel}
        for k, c in run["calls_mean"].items():
            n = sym_share.get("KernelGame::" + k)
            if n and c:
                say(f"OBS 79.37 A {a.name} per call {k}: {1e6 * (n / d) * wall / c:.2f} microseconds on average (the share of the decision times its wall over its calls; it holds the calls it makes)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("positions", "run", "profile", "report"))
    ap.add_argument("--logs", default="")
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3737)
    ap.add_argument("--seconds", type=float, default=40.0, help="how long the process makes decisions")
    ap.add_argument("--sample", type=int, default=30, help="how long `sample` watches it (seconds)")
    ap.add_argument("--name", default="shared")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args(argv)
    root = out_root(a.tag)

    def say(line: str) -> None:
        print(line, flush=True)
        with open(root / "obs.txt", "a") as h:
            h.write(line + "\n")
    if a.stage == "positions":
        cmd_positions(a, say)
        return 0
    if a.stage == "run":
        return cmd_run(a)
    if a.stage == "report":
        return report(a, json.loads((root / f"run_{a.name}.json").read_text()), root / f"sample_{a.name}.txt", say)
    return cmd_profile(a, say)


if __name__ == "__main__":
    raise SystemExit(main())
