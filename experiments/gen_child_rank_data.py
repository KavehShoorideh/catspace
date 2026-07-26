#!/usr/bin/env python
"""experiments/gen_child_rank_data.py -- S2d data: the LOCAL move-selection signal the field
lacks. For won parents (White to move, pieces-only so DTZ~DTM and no 50-move noise), emit every
CHILD with its tablebase |DTZ| (one fast probe/child, ~2000/s) as a rank key, grouped by parent.
Won children carry |DTZ| (lower = better White move); children that throw the win (not won for
White) carry -1 = INF. A within-group rank loss + regression on this fixes both the coin-flip
move-selection AND the rank collapse (distinguishing many siblings spreads the representation).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import encode_meta, encode_packed
from catspace.tb import TB, DEFAULT_SYZYGY
from experiments.gen_dtm_data import random_class_start
from experiments.value_fixed_point import white_pov_value

CLASSES = ["KQvK", "KRvK", "KRRvK", "KBNvK", "KBBvK", "KQvKR", "KRvKN"]   # pieces-only


def worker(task):
    classes, n, seed, syzygy = task
    rng = np.random.default_rng(seed)
    tb = TB(str(syzygy), cache_db=None); syz = tb.tb
    out = []; got = tries = 0; gid = seed * 1_000_000
    while got < n and tries < n * 200:
        tries += 1
        cls = classes[rng.integers(0, len(classes))]
        b = random_class_start(rng, cls)
        if b is None or b.turn != chess.WHITE or b.is_game_over():
            continue
        if white_pov_value(b, tb) != 1.0:
            continue
        gid += 1
        for m in b.legal_moves:
            b.push(m)
            if b.is_checkmate():
                key = 0                                       # mate = distance 0 (attractor)
            elif b.is_game_over(claim_draw=True) or white_pov_value(b, tb) != 1.0:
                key = -1                                      # threw the win -> INF
            else:
                try:
                    key = abs(syz.probe_dtz(b))               # |DTZ|: lower = better White move
                except Exception:
                    key = -1
            out.append((encode_packed(b), encode_meta(b), int(key), gid))
            b.pop()
        got += 1
    tb.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parents", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="data/derived/child_rank_v1.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    W = args.workers or max(1, (os.cpu_count() or 4) - 1)
    syz = str(DEFAULT_SYZYGY)
    tasks = [(CLASSES, max(1, args.parents // W), args.seed + i, syz) for i in range(W)]
    print(f"[gen-child-rank] {W} workers x {args.parents // W} parents", flush=True)
    parts = []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for i, r in enumerate(ex.map(worker, tasks)):
            parts.append(r); print(f"  w{i+1}/{W}: {len(r)} rows [{time.time()-t0:.0f}s]", flush=True)
    rows = [x for p in parts for x in p]
    packed = np.stack([r[0] for r in rows]); meta = np.stack([r[1] for r in rows])
    key = np.array([r[2] for r in rows], np.int32); grp = np.array([r[3] for r in rows], np.int64)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, packed=packed, meta=meta, dtz=key, group=grp)
    won = key >= 1; mate = key == 0; inf = key == -1
    print(f"\n=== {args.out}: {len(rows)} child rows, {len(np.unique(grp))} groups [{time.time()-t0:.0f}s] ===")
    print(f"  won-children {won.sum()} (|dtz| med {int(np.median(key[won])) if won.any() else 0}) "
          f"| mate {mate.sum()} | INF(threw-win) {inf.sum()}")
    print("DONE gen_child_rank_data", flush=True)


if __name__ == "__main__":
    main()
