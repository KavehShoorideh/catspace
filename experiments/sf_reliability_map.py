#!/usr/bin/env python
"""experiments/sf_reliability_map.py -- S1 of METASTABILITY_PLAN (Kaveh 2026-07-26):
"the reliability map of SF is the thing that solves it." SF is a strong REFERENCE, not an
oracle. In the endgame we HAVE truth (tablebase), so we can measure exactly WHERE SF's
verdict matches truth -- as a function of outcome class, material, and eval margin. That map
is what lets us weight SF honestly everywhere downstream.

Method: sample tablebase positions across many classes, bucket by TRUE WDL (white_pov_value),
get SF's eval at fixed depth, classify SF's WDL by a centipawn band, and compare. Parallel SF
workers (one engine + one cache-free TB handle each; endgame eval is embarrassingly parallel).
Reports SF WDL-accuracy overall and broken down by true outcome / class / margin / depth --
i.e. the reliability map (e.g. "SF calls X% of true DRAWS as wins" = fortress/insufficient
blindness; "misses Y% of deep WINS as draws").
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB, DEFAULT_SYZYGY
from experiments.gen_dtm_data import random_class_start
from experiments.value_fixed_point import white_pov_value

# hard-leaning: boundary classes where search (esp. shallow) diverges from tablebase truth --
# deep wins past the horizon, fortress/insufficient draws, DTZ/50-move cases.
CLASSES = ["KNNvKP",                                             # deep win (KNNvK is a draw!)
           "KQvKR", "KRvKB", "KRvKN", "KRvKP", "KQvKP",          # long wins / drawish boundaries
           "KBBvKN", "KQvKBB", "KRRvKQ", "KQvKRR",               # material-imbalance judgment
           "KPvKP", "KRvKR", "KQvKQ", "KBvKN",                   # mostly draws (fortress/insuff.)
           "KQvK", "KRvK", "KRRvK", "KBNvK", "KNNvK", "KBvK"]     # easy anchors (should be ~100%)


def _wdl_from_cp(cp, band):
    if cp is None:
        return None
    return "W" if cp >= band else ("L" if cp <= -band else "D")


def _true_wdl(v):
    return "W" if v == 1.0 else ("L" if v == 0.0 else "D")


def worker(task):
    classes, n, seed, depths, band, engine_path, syzygy = task
    rng = np.random.default_rng(seed)
    tb = TB(str(syzygy), cache_db=None)
    eng = chess.engine.SimpleEngine.popen_uci(engine_path)
    eng.configure({"Threads": 1, "Hash": 64})
    rows = []
    got = tries = 0
    while got < n and tries < n * 200:
        tries += 1
        cls = classes[rng.integers(0, len(classes))]
        b = random_class_start(rng, cls)
        if b is None or b.is_game_over():
            continue
        try:
            v = white_pov_value(b, tb)
        except Exception:
            continue
        true = _true_wdl(v)
        npieces = chess.popcount(b.occupied)
        for d in depths:
            try:
                info = eng.analyse(b, chess.engine.Limit(depth=d))
                sc = info["score"].white()
                cp = sc.score(mate_score=100000)
            except Exception:
                cp = None
            rows.append((cls, npieces, true, d, cp, _wdl_from_cp(cp, band)))
        got += 1
    eng.quit(); tb.close()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6000, help="total positions")
    ap.add_argument("--depths", type=int, nargs="*", default=[8, 14])
    ap.add_argument("--band", type=int, default=150, help="cp draw-band (|cp|<band => SF says draw)")
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    W = args.workers or max(1, (os.cpu_count() or 4) - 1)
    per = max(1, args.n // W)
    syz = str(DEFAULT_SYZYGY)
    tasks = [(CLASSES, per, args.seed + i, args.depths, args.band, args.engine, syz)
             for i in range(W)]
    print(f"[sf-reliability] {W} workers x {per} pos, depths {args.depths}, band {args.band}cp",
          flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for i, r in enumerate(ex.map(worker, tasks)):
            rows.extend(r)
            print(f"  worker {i+1}/{W}: {len(r)} evals [{time.time()-t0:.0f}s]", flush=True)

    # numpy arrays for fast aggregation
    cls = np.array([r[0] for r in rows]); npc = np.array([r[1] for r in rows])
    true = np.array([r[2] for r in rows]); dep = np.array([r[3] for r in rows])
    cp = np.array([np.nan if r[4] is None else r[4] for r in rows], float)
    pred = np.array([r[5] if r[5] is not None else "?" for r in rows])
    ok = pred == true

    def acc(mask):
        m = mask & (pred != "?")
        return (100 * ok[m].mean() if m.sum() else float("nan")), int(m.sum())

    print(f"\n=== SF RELIABILITY MAP ({len(rows)} evals, {time.time()-t0:.0f}s) ===")
    for d in args.depths:
        a, k = acc(dep == d)
        print(f"depth {d}: overall WDL-acc {a:.1f}%  (n={k})")
        for t in ("W", "D", "L"):
            a2, k2 = acc((dep == d) & (true == t))
            # where does SF send true-t positions? (confusion)
            conf = defaultdict(int)
            for p in pred[(dep == d) & (true == t)]:
                conf[p] += 1
            tot = sum(conf.values()) or 1
            mix = " ".join(f"{p}:{100*conf[p]/tot:.0f}%" for p in ("W", "D", "L", "?") if conf[p])
            print(f"    true {t}: acc {a2:.1f}% (n={k2})  SF-says[{mix}]")
    # worst classes at the deepest depth
    dd = max(args.depths)
    print(f"\nleast-reliable classes @ depth {dd}:")
    percls = []
    for c in CLASSES:
        a, k = acc((dep == dd) & (cls == c))
        if k >= 20:
            percls.append((a, k, c))
    for a, k, c in sorted(percls)[:8]:
        print(f"    {c:7} acc {a:.1f}% (n={k})")
    print("DONE sf_reliability_map", flush=True)


if __name__ == "__main__":
    main()
