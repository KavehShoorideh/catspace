"""THE TACTICAL LABELS -- the short window (Kaveh 2026-10-13: "a long window that give us some probabilistic kind of Monte
Carlo tree search like search situation and the short window which gives us exact tactical stuff ... the density of
tactics is basically what we're predicting ... high value tactics for us ... no tactics for them ... we need to be able to
label the tactical situations ... and the density of them").

Every position of ours in the given games, labelled exactly (goals.rs tactic_profile): for us, and for them as if we
passed, at each value of the ladder (1 a pawn, 3 a piece, 5 a rook, 9 a queen, mate) the fewest moves forcing a gain of
that much within the window, how many first moves force it, how many the budget left unknown. The labels a long-window
model learns to predict; nothing here is learned.

  python -m catspace.incubator.analyst.discovery.tactics --games 'artifacts/experiments/castled_play_r1.json' --tag castled
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import Counter, defaultdict

from catspace.io import paths

DEPTH = 3                     # moves of the attacker: five plies and the reply that holds the gain
BUDGET = 200_000              # nodes per first move and value
LEVELS = (1, 3, 5, 9, "mate")


def _one(args):
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    fen, material, seed, out, depth, budget = args
    r = A.tactic_profile(fen, depth, budget)
    if r is None:
        return None
    side = lambda xs: None if xs is None else {("mate" if v > 100 else v): dict(fewest=n, moves=c, unknown=u) for v, n, c, u in xs}
    return dict(fen=fen, material=material, seed=seed, out=out, ours=side(r[0]), theirs=side(r[1]))


def density(x: dict | None) -> dict:
    """One side's tactics in a position: the best value forced (and how soon), how many (value, first move) pairs force
    something, unknowns."""
    if x is None:
        return dict(best=None, soon=None, count=0, unknown=0)
    forced = [(lv, x[lv]) for lv in LEVELS if x[lv]["moves"]]
    best = forced[-1] if forced else None
    return dict(best=best[0] if best else None, soon=best[1]["fewest"] if best else None,
                count=sum(v["moves"] for _, v in forced), unknown=sum(x[lv]["unknown"] for lv in LEVELS))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True, help="a glob of results JSONs (surprise_loop._game rows)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    a = ap.parse_args(argv)
    jobs, seen = [], set()
    for f in sorted(glob.glob(a.games)):
        for g in json.load(open(f)):
            for r in g["rows"]:
                if r["fen"] in seen:
                    continue
                seen.add(r["fen"])
                jobs.append((r["fen"], r["material"], g.get("seed"), g["out"], a.depth, a.budget))
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(a.workers) as p:
        rows = [x for x in p.imap(_one, jobs, chunksize=8) if x]
    out = paths.derived(f"tactics/{a.tag}_d{a.depth}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        for x in rows:
            fh.write(json.dumps(x) + "\n")
    dt = time.time() - t0
    print(f"OBS {len(rows)} positions labelled (window {a.depth} moves, budget {a.budget}) in {dt:.0f} s "
          f"({1000 * dt / max(len(rows), 1) * a.workers:.0f} ms a position on one core) -> {out}")
    by = defaultdict(list)
    for x in rows:
        by[x["material"]].append(x)
    for m, xs in sorted(by.items(), key=lambda kv: -len(kv[1]))[:12]:
        d_us = [density(x["ours"]) for x in xs]; d_th = [density(x["theirs"]) for x in xs]
        best = lambda ds: dict(Counter(d["best"] for d in ds if d["best"] is not None))
        unk = sum(d["unknown"] > 0 for d in d_us + d_th)
        print(f"OBS   {m:16s} {len(xs):5d} positions: ours forced {sum(d['best'] is not None for d in d_us):4d} (best {best(d_us)}); "
              f"theirs as if we passed {sum(d['best'] is not None for d in d_th):4d} (best {best(d_th)}); with an unknown {unk}")


if __name__ == "__main__":
    main()
