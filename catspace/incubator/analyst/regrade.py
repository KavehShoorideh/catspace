"""A DETERMINISTIC GRADER (2026-10-12): the decision review's Stockfish carries its hash from one analysis to the next, so
the same move in the same position was graded 15 and 294 cp in two runs -- configurations a few decisions apart could
not be told apart. Here every (position, move) is graded once, each analysis a new game (fresh hash), depth 12, one
thread, and cached (data/derived/sf_grade_cache.json); every store given is then scored on the same grades.

  python -m catspace.incubator.analyst.regrade decision_review_seed8_base.json decision_review_seed8_new.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

from catspace.io import paths

CACHE = "sf_grade_cache.json"
DEPTH = 12
_ENG = None


def _score(key: str):
    global _ENG
    import chess
    import chess.engine
    if _ENG is None:
        _ENG = chess.engine.SimpleEngine.popen_uci("stockfish")
        _ENG.configure({"Threads": 1, "Hash": 16})
    fen, _, mv = key.partition("|")
    b = chess.Board(fen)
    if mv:
        b.push_uci(mv)
    info = _ENG.analyse(b, chess.engine.Limit(depth=DEPTH), game=object())
    # from the point of view of the side to move in `fen`
    pov = not b.turn if mv else b.turn
    return key, info["score"].pov(pov).score(mate_score=3000)


def wc(cp: int) -> float:
    return 2 / (1 + math.exp(-0.00368208 * max(-3000, min(3000, cp)))) - 1


def grade(stores: list[str], workers: int = 6) -> dict:
    from multiprocessing import Pool
    cpath = paths.derived(CACHE)
    cache = json.load(open(cpath)) if os.path.exists(cpath) else {}
    rows = {s: json.load(open(paths.experiment(s))) for s in stores}
    keys = {r["fen"] + "|" for rs in rows.values() for r in rs} | {r["fen"] + "|" + r["move"] for rs in rows.values() for r in rs}
    todo = sorted(k for k in keys if k not in cache)
    t0 = time.time()
    if todo:
        with Pool(workers) as p:
            for k, v in p.imap_unordered(_score, todo, chunksize=16):
                cache[k] = v
        json.dump(cache, open(cpath, "w"))
    print(f"OBS graded {len(todo)} new (position, move) pairs in {time.time() - t0:.0f} s; cache {len(cache)}")
    out = {}
    for s, rs in rows.items():
        g = [dict(r, best_d=cache[r["fen"] + "|"], got_d=cache[r["fen"] + "|" + r["move"]]) for r in rs]
        for r in g:
            r["loss_d"] = r["best_d"] - r["got_d"]
            r["blunder_d"] = wc(r["best_d"]) - wc(r["got_d"]) >= 0.3
        n = len(g); bad = sum(r["loss_d"] >= 150 for r in g); bl = sum(r["blunder_d"] for r in g)
        print(f"OBS {s}: {n} decisions; 150+ cp {bad} ({100 * bad / n:.1f}%); blunders by winning chances {bl} ({100 * bl / n:.1f}%)")
        out[s] = {r["fen"]: r for r in g}
    return out


def compare(a: dict, b: dict, show: int = 0) -> None:
    common = set(a) & set(b)
    ch = [f for f in common if a[f]["move"] != b[f]["move"]]
    fixed = [f for f in ch if b[f]["loss_d"] < 150 <= a[f]["loss_d"]]
    broken = [f for f in ch if a[f]["loss_d"] < 150 <= b[f]["loss_d"]]
    bf = [f for f in ch if a[f]["blunder_d"] and not b[f]["blunder_d"]]
    bb = [f for f in ch if b[f]["blunder_d"] and not a[f]["blunder_d"]]
    print(f"OBS   {len(common)} in common, {len(ch)} moves changed: 150+ cp fixed {len(fixed)}, broken {len(broken)}; blunders fixed {len(bf)}, broken {len(bb)}")
    for f in broken[:show]:
        print(f"OBS   broken: {a[f]['move']} ({a[f]['loss_d']}) -> {b[f]['move']} ({b[f]['loss_d']}) | {b[f]['summary']} | {f}")
    for f in fixed[:show]:
        print(f"OBS   fixed:  {a[f]['move']} ({a[f]['loss_d']}) -> {b[f]['move']} ({b[f]['loss_d']}) | {b[f]['summary']} | {f}")


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    show = int(next((x.split("=")[1] for x in sys.argv if x.startswith("--show=")), "0"))
    g = grade(args)
    for s in args[1:]:
        print(f"OBS {args[0]} -> {s}:")
        compare(g[args[0]], g[s], show)
