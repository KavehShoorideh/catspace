"""PLAN 78.5 THE NEW ENGINE ON PUZZLES: cs_engine.best_move (tactics `forcing`, then the LEDGER SEARCH) on lichess puzzles -- solved when our move is the
puzzle's answer (its first move). The judge is the variable: `count` (no values: a unit lost is a unit lost, ours first -- the default) and, as a COMPARISON only,
`worth` (the store's chess valuation plugged in). Per theme group (mates; material: advantage / crushing; other) and per judge: solved / total, the proved share,
the depth reached, the time. One puzzle per process (all cores).

  .venv/bin/python scripts/plan78_puzzles.py [--sets rand1_10,...] [--seconds 3] [--jobs 11] [--judges count,worth] [--tag x]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cs_engine as CE
from catspace.io import paths


def worth() -> dict:
    from catspace.incubator.analyst.discovery import lemma_store as LS
    return {k: int(v) for k, v in LS.valuation_of("chess").worth.items()}


def group(themes) -> str:
    t = set(ast.literal_eval(themes) if isinstance(themes, str) else themes)
    if any(x.startswith("mate") for x in t):
        return "mate"
    if t & {"advantage", "crushing"}:
        return "material"
    return "other"


def solve(job: tuple) -> dict:
    p, judge, seconds, w = job
    t = time.time()
    mv, proved, trace, nodes, _f = CE.best_move("chess", p["fen"], [], seconds, w if judge == "worth" else {})
    depth = next((int(m.group(1)) for x in trace for m in [re.search(r"at depth (\d+)", x)] if m), 0)
    return {"id": p["id"], "set": p["set"], "judge": judge, "move": mv, "answer": p["answer"], "solved": mv == p["answer"], "proved": proved,
            "group": group(p.get("themes", [])), "depth": depth, "nodes": nodes, "sec": time.time() - t, "trace": trace[-1][:400]}


def main(argv: list) -> int:
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(f"rand{i}_10" for i in range(1, 22)))
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--jobs", type=int, default=11)
    ap.add_argument("--judges", default="count,worth")
    ap.add_argument("--tag", default="v0")
    a = ap.parse_args(argv)
    puzzles = [dict(p, set=s) for s in a.sets.split(",") for p in PA.load_set(s)]
    w = worth()
    jobs = [(p, j, a.seconds, w) for j in a.judges.split(",") for p in puzzles]
    t0 = time.time()
    with Pool(a.jobs) as pool:
        res = pool.map(solve, jobs, chunksize=1)
    out = Path(paths.experiment(f"plan78_puzzles_{a.tag}.json"))
    out.write_text(json.dumps(res, indent=1))
    print(f"OBS plan 78.5 puzzles ({a.tag}): {len(puzzles)} puzzles of {len(a.sets.split(','))} sets at {a.seconds} s each, judges {a.judges}; wall {time.time() - t0:.0f} s -> {out}")
    for j in a.judges.split(","):
        rs = [r for r in res if r["judge"] == j]
        by = defaultdict(Counter)
        for r in rs:
            for g in (r["group"], "all"):
                by[g]["n"] += 1
                by[g]["solved"] += r["solved"]
                by[g]["proved"] += r["proved"]
                by[g]["depth"] += r["depth"]
        print(f"   judge {j}" + (" (COMPARISON: the store's chess valuation plugged in)" if j == "worth" else " (no values: the default)") + ": " +
              "; ".join(f"{g} {c['solved']}/{c['n']} solved ({100 * c['solved'] / c['n']:.0f} %), proved {c['proved']}, mean depth {c['depth'] / c['n']:.1f}"
                        for g, c in sorted(by.items(), key=lambda kv: (kv[0] != "all", kv[0]))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
