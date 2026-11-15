"""PLAN 78.47 A WHERE THE FORCING READ SPENDS ITS POSITIONS: a measure over a puzzle set (positions only: no answer is read, nothing of a puzzle reaches the engine).

For every position the Rust forcing read is made at the given fuse and its own account of the work is summed: per KIND OF WORK (why a position was asked for: the root, the child of one of our
moves to see what the move claims, the position after a reply to see what stands, the other side's gain, the other side's sign moves, the test whether a threat move of ours is left; each again
when it is read inside a deeper threat move of ours) the positions read (what the fuse counts), the exchange-frame nodes, the nodes of the goal read, the calls answered from the read's memory, the
seconds; and the replies by kind (took something | quiet) and how many of each were answered from the stored result without a new read (plan 78.47 B).

  .venv/bin/python -m catspace.incubator.analyst.discovery.forcing_cost_measure --variant chess --sets dev_300 [--fuse N] [--limit N] [--jobs N] [--tag NAME]
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import time
from collections import Counter

_ST: dict = {}


def _init(variant: str) -> None:
    from catspace.incubator.analyst.discovery import question_sheet as QS
    from catspace.incubator.analyst.discovery import sheet_rust as SR
    env = QS.Env(variant)
    _ST.update(model=SR.model(env, QS.store_worth(variant)))


def run_one(arg: tuple) -> dict:
    pos, fuse = arg
    out = {"read": 0, "cut": 0, "nodes": 0, "seconds": 0.0, "work": {}, "replies_took": 0, "replies_quiet": 0, "unchanged_took": 0, "unchanged_quiet": 0}
    t0 = time.perf_counter()
    try:
        res = _ST["model"].forcing(pos, fuse)
    except ValueError:
        return out
    out["seconds"] = time.perf_counter() - t0
    out["read"] = 1
    out["cut"] = int(bool(res["cut"]))
    out["nodes"] = res["nodes"]
    for k, v in res["cost"].items():
        if isinstance(v, dict):
            out["work"][k] = v
        else:
            out[k] = v
    return out


def merge(parts: list) -> dict:
    tot = {"read": 0, "cut": 0, "nodes": 0, "seconds": 0.0, "replies_took": 0, "replies_quiet": 0, "unchanged_took": 0, "unchanged_quiet": 0}
    work: dict = {}
    for p in parts:
        for k in tot:
            tot[k] += p[k]
        for k, v in p["work"].items():
            w = work.setdefault(k, Counter())
            for f, x in v.items():
                w[f] += x
    tot["work"] = work
    return tot


def obs_lines(tag: str, tot: dict, fuse: "int | None") -> list:
    n = max(1, tot["read"])
    L = [f"OBS forcing cost {tag}: positions read {tot['read']}, fuse {fuse if fuse is not None else 'default'}, the fuse stopped the read in {tot['cut']}, positions asked per position {tot['nodes'] / n:.0f}, "
         f"seconds per position {tot['seconds'] / n:.4f}"]
    all_pos = max(1, sum(w["positions"] for w in tot["work"].values()))
    all_sec = max(1e-12, sum(w["seconds"] for w in tot["work"].values()))
    for k, w in tot["work"].items():
        if not (w["positions"] or w["memory"]):
            continue
        L.append(f"OBS forcing cost {tag}: work '{k}': positions {int(w['positions'])} ({100 * w['positions'] / all_pos:.1f} pct), exchange-frame nodes {int(w['frame_nodes'])}, goal-read nodes {int(w['goal_nodes'])}, "
                 f"answered from memory {int(w['memory'])}, seconds {w['seconds']:.2f} ({100 * w['seconds'] / all_sec:.1f} pct of the time in positions)")
    rep = max(1, tot["replies_took"] + tot["replies_quiet"])
    L.append(f"OBS forcing cost {tag}: replies read {tot['replies_took'] + tot['replies_quiet']}: took something {tot['replies_took']} ({100 * tot['replies_took'] / rep:.1f} pct), quiet {tot['replies_quiet']}; "
             f"answered from the stored result: took {tot['unchanged_took']}, quiet {tot['unchanged_quiet']}")
    return L


def main(argv: "list | None" = None) -> int:
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", required=True)
    ap.add_argument("--sets", default="dev_300")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--fuse", type=int, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    for name in a.sets.split(","):
        recs = PA.load_set(name)
        if a.limit:
            recs = recs[:a.limit]
        t0 = time.perf_counter()
        with multiprocessing.get_context("spawn").Pool(a.jobs, initializer=_init, initargs=(a.variant,)) as pool:
            parts = pool.map(run_one, [(r["fen"], a.fuse) for r in recs], chunksize=1)
        tag = a.tag or name
        for line in obs_lines(tag, merge(parts), a.fuse):
            print(line, flush=True)
        print(f"OBS forcing cost {tag}: wall {time.perf_counter() - t0:.1f} s, jobs {a.jobs}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
