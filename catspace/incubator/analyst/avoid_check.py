"""GRADED BLUNDER WARNINGS, measured (Kaveh 2026-10-10: "probable blunder -> ascertain certainty; certain blunder -> do not
do it"; plan.rs grade_and_avoid). Census positions planned with AVOID off and on: the grades (certain / unavoidable),
and where the decision changed, Stockfish's (depth 12) loss of the old move and of the new one against its best; the
kept "unavoidable" decisions' loss beside them; time."""
from __future__ import annotations

import argparse
import os
import time

from .pool_check import _sf


def _one(args):
    fen, on = args
    os.environ["RAYON_NUM_THREADS"] = "1"
    import analyst_core
    analyst_core.set_avoid(on)
    tr, dec, proofs, nodes, ms, imm = analyst_core.plan_trace(fen)
    mv = (dec[1] if dec[0] in ("", "defend", "cheapest") else dec[0]) if dec else None
    grade = next((g for g in ("certain blunder", "unavoidable", "probable") for l in tr if l.strip().startswith(g)), None)
    return fen, mv, grade, ms, proofs


def main(argv=None):
    from multiprocessing import Pool
    from .waiting_frames import _games
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args(argv)
    fens = [f for _, fs in _games("c300", a.games) for f in fs]
    res = {}
    for on in (False, True):
        t0 = time.time()
        with Pool(a.workers) as p:
            res[on] = p.map(_one, [(f, on) for f in fens], chunksize=8)
        print(f"TIME avoid {'on ' if on else 'off'}: {time.time() - t0:.0f} s; ms median {sorted(r[3] for r in res[on])[len(fens) // 2]:.1f}, proofs mean {sum(r[4] for r in res[on]) / len(fens):.1f}", flush=True)
    grades = {}
    for r in res[True]:
        grades[r[2]] = grades.get(r[2], 0) + 1
    print(f"OBS {len(fens)} positions; grades of the warnings: {grades}")
    changed = [(x[0], x[1], y[1]) for x, y in zip(res[False], res[True]) if x[1] and y[1] and x[1] != y[1]]
    kept = [(y[0], y[1]) for y in res[True] if y[2] == "unavoidable" and y[1]]
    with Pool(a.workers) as p:
        lo = p.map(_sf, [(f, m) for f, m, _ in changed], chunksize=2)
        ln = p.map(_sf, [(f, m) for f, _, m in changed], chunksize=2)
        lk = p.map(_sf, kept, chunksize=2)
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    print(f"OBS decisions changed (certain blunders avoided): {len(changed)}")
    for name, v in (("the old move (the blunder)", lo), ("the new move", ln), ("kept as unavoidable", lk)):
        print(f"OBS   {name}: n {len(v)}, loss vs Stockfish's best median {med(v)} cp, within 50 cp {100 * sum(x <= 50 for x in v) / max(len(v), 1):.0f}%, losing 150+ {100 * sum(x >= 150 for x in v) / max(len(v), 1):.0f}%")
    better = sum(1 for o, n in zip(lo, ln) if n < o - 50); worse = sum(1 for o, n in zip(lo, ln) if n > o + 50)
    print(f"OBS   per position: the new move better by 50+ cp {better}, worse by 50+ cp {worse}, within 50 cp {len(lo) - better - worse}")


if __name__ == "__main__":
    main()
