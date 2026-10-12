"""REPLAY A DIG (2026-10-12): the cases of a surprise-loop store (surprise_unfixed_*.json) decided again by the planner as
built now, each new move graded by Stockfish (depth 12, the loop's bar) -- per class of the dig, how many now fall under
the bar, and how many moved away from a move that was already fine (none here: every case lost 150+ before).

  python -m catspace.incubator.analyst.dig_replay --store surprise_unfixed_seed7.json --classes seed7
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

from catspace.io import paths

BAR = 150
# the seed-7 dig's classes (artifacts/experiments/missing_lemma_dig_seed7.md)
SEED7 = dict(
    A=[9, 18, 28, 37, 44, 58, 68], B=[15, 42, 43, 47, 61], C=[20, 22, 29, 51, 55, 60], D=[39, 64], E=[14, 21, 27, 31, 34, 38],
    F=[48], switches=[2, 16, 24, 62, 73],
    noise=[0, 1, 4, 6, 8, 10, 12, 19, 23, 32, 35, 41, 49, 53, 56, 57, 63, 69, 70, 72, 78])

_ENG = None


def _sf(b, depth=12):
    global _ENG
    import chess.engine
    if _ENG is None:
        _ENG = chess.engine.SimpleEngine.popen_uci("stockfish")
    return _ENG.analyse(b, chess.engine.Limit(depth=depth))


def _loss(fen: str, mv: str) -> int:
    import chess
    b = chess.Board(fen)
    i0 = _sf(b)
    a = b.copy(); a.push_uci(mv)
    i1 = _sf(a)
    return i0["score"].pov(b.turn).score(mate_score=3000) - i1["score"].pov(b.turn).score(mate_score=3000)


def _case(args):
    i, row = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core
    for kv in filter(None, os.environ.get("CATSPACE_FLAGS", "").split(",")):
        k, v = kv.split("=")
        getattr(analyst_core, f"set_{k}")(v == "1")
    r = analyst_core.plan_trace(row["fen"])
    dec = r[1]
    mv = None
    if dec:
        mv = dec[1] if dec[0] in ("", "defend", "cheapest") else dec[0]
    loss = _loss(row["fen"], mv) if mv else row["loss"]
    return dict(i=i, fen=row["fen"], old=row["move"], new=mv, sf=row["best_move"], old_loss=row["loss"], loss=loss, how=dec[3] if dec else None, ms=r[4])


def run(store: str, classes: str, workers: int, out: str | None) -> None:
    from multiprocessing import Pool
    rows = json.load(open(paths.experiment(store)))
    t0 = time.time()
    with Pool(workers) as p:
        res = sorted(p.imap_unordered(_case, list(enumerate(rows)), chunksize=1), key=lambda r: r["i"])
    cls = SEED7 if classes == "seed7" else {}
    of = {i: k for k, v in cls.items() for i in v}
    by = Counter(); fixed = Counter(); changed = Counter()
    for r in res:
        k = of.get(r["i"], "positional")
        by[k] += 1
        changed[k] += r["new"] != r["old"]
        fixed[k] += r["loss"] < BAR
    print(f"OBS dig replay: {len(res)} cases of {store} ({time.time() - t0:.0f} s on {workers} processes); decision changed {sum(changed.values())}, "
          f"now under {BAR} cp {sum(fixed.values())}; same as Stockfish's move {sum(r['new'] == r['sf'] for r in res)}; ms a position {sum(r['ms'] for r in res) / len(res):.0f}")
    for k in by:
        print(f"OBS   {k:10s}: {by[k]:2d} cases, changed {changed[k]:2d}, now under the bar {fixed[k]:2d}")
    for r in res:
        if r["new"] != r["old"]:
            print(f"OBS   [{r['i']}] {of.get(r['i'], 'positional')}: {r['old']} ({r['old_loss']}) -> {r['new']} ({r['loss']}); Stockfish {r['sf']}; {r['how']}")
    if out:
        with open(paths.experiment(out), "w") as fh:
            json.dump(res, fh, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="surprise_unfixed_seed7.json")
    ap.add_argument("--classes", default="seed7")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run(a.store, a.classes, a.workers, a.out)


if __name__ == "__main__":
    main()
