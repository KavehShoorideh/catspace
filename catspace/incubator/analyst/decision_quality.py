"""EVERY DECISION AGAINST STOCKFISH (2026-10-10): the planner's decision move on a fixed sample of census positions,
Stockfish's (depth 12) loss of it against its best, for planner settings compared on the same positions (here the
change flag gating the net's certification, on and off) -- the guard cannot judge a setting it shares."""
import os
import random
import sys
import time
from multiprocessing import Pool

from .pool_check import _sf


def _one(args):
    fen, gate = args
    os.environ["RAYON_NUM_THREADS"] = "1"
    import analyst_core
    analyst_core.set_change_gate(gate)
    t = time.time()
    tr, dec, proofs, *_ = analyst_core.plan_trace(fen)
    mv = (dec[1] if dec[0] in ("", "defend", "cheapest") else dec[0]) if dec else None
    return fen, mv, (dec[0] if dec else None), proofs, time.time() - t


if __name__ == "__main__":
    from .waiting_frames import _games
    fens = random.Random(5).sample([f for _, fs in _games("c300", 300) for f in fs], int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
    out = {}
    for gate in (False, True):
        with Pool(os.cpu_count()) as p:
            out[gate] = p.map(_one, [(f, gate) for f in fens], chunksize=8)
    both = [(a, b) for a, b in zip(out[False], out[True]) if a[1] and b[1]]
    with Pool(os.cpu_count()) as p:
        l0 = p.map(_sf, [(a[0], a[1]) for a, _ in both], chunksize=4)
        l1 = [x if a[1] != b[1] else None for (a, b), x in zip(both, l0)]
        diff = [(b[0], b[1]) for (a, b) in both if a[1] != b[1]]
        ld = p.map(_sf, diff, chunksize=4)
    it = iter(ld)
    l1 = [x if y is None else next(it) for x, y in zip(l0, l1)]
    for name, v, res in (("gate off", l0, out[False]), ("gate on ", l1, out[True])):
        n = len(v)
        print(f"OBS {name}: {n} decisions, loss vs Stockfish's best median {sorted(v)[n // 2]} cp, mean {sum(min(x, 1000) for x in v) / n:.0f} (capped 1000), "
              f"losing 150+ {100 * sum(x >= 150 for x in v) / n:.1f}%, 300+ {100 * sum(x >= 300 for x in v) / n:.1f}%; proofs mean {sum(r[3] for r in res) / len(res):.1f}, "
              f"time {sum(r[4] for r in res):.0f} s cpu")
    print(f"OBS decisions that differ: {len(diff)}")
