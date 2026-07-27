#!/usr/bin/env python
"""experiments/gen_clock_child_data.py -- CLOCK-AWARE child-rank data (Kaveh: represent the
approaching 50-move draw surface). Like gen_child_rank_data but every won parent gets a RANDOM
halfmove clock h, and labels are CLOCK-AWARE:
  * a won child is only really won if it can ZERO (pawn move/capture) or MATE before the 50-move
    draw: won iff white_pov_value==WIN AND |DTZ| <= 100 - child_halfmove; else it is drawn-by-
    clock -> INF (dtz=-1), even with winning material.
  * zeroing moves reset child_halfmove to 0 (python-chess auto) -> they ESCAPE the draw surface;
    that IS the progress signal the field must learn (near the surface, the winning moves are the
    zeroing ones).
The halfmove clock rides in the 20-plane feature stack (plane 18), so a field trained on
feature_planes CAN see and represent the draw surface. dtz sentinel: >=1 won, 0 mate, -1 INF.
"""
from __future__ import annotations

import argparse, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.data.encode import encode_meta, encode_packed
from catspace.tb import TB, DEFAULT_SYZYGY
from experiments.gen_dtm_data import random_class_start
from experiments.value_fixed_point import white_pov_value

CLASSES = ["KQvK", "KRvK", "KRRvK", "KBNvK", "KBBvK", "KQvKR", "KRvKN"]


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
        b.halfmove_clock = int(rng.integers(0, 99))          # CLOCK VARIATION
        gid += 1
        for m in b.legal_moves:
            b.push(m)
            ch = b.halfmove_clock                            # 0 if zeroing (pawn/capture), else h+1
            # ending type (categorical head): WIN_MATE0 DRAW_FIFTY1 STALE2 INSUF3 REP4 LOSS_MATE5
            if b.is_checkmate():
                key, end = 0, (0 if b.turn == chess.BLACK else 5)
            elif b.is_stalemate():
                key, end = -1, 2
            elif b.is_insufficient_material():
                key, end = -1, 3
            elif b.is_game_over(claim_draw=True) or white_pov_value(b, tb) != 1.0:
                key, end = -1, 4                             # repetition/other -> approx REP
            else:
                try:
                    d = abs(syz.probe_dtz(b))
                    if d <= (100 - ch):
                        key, end = d, 0                      # still winnable -> WIN_MATE
                    else:
                        key, end = -1, 1                     # can't zero in time -> DRAW_FIFTY
                except Exception:
                    key, end = -1, 1
            out.append((encode_packed(b), encode_meta(b), int(key), gid, int(end)))
            b.pop()
        got += 1
    tb.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parents", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="data/derived/clock_child_v1.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); W = args.workers or max(1, (os.cpu_count() or 4) - 1)
    tasks = [(CLASSES, max(1, args.parents // W), args.seed + i, str(DEFAULT_SYZYGY)) for i in range(W)]
    print(f"[gen-clock] {W} workers x {args.parents // W} parents", flush=True)
    parts = []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for i, r in enumerate(ex.map(worker, tasks)):
            parts.append(r); print(f"  w{i+1}/{W}: {len(r)} [{time.time()-t0:.0f}s]", flush=True)
    rows = [x for p in parts for x in p]
    packed = np.stack([r[0] for r in rows]); meta = np.stack([r[1] for r in rows])
    key = np.array([r[2] for r in rows], np.int32); grp = np.array([r[3] for r in rows], np.int64)
    end = np.array([r[4] for r in rows], np.int8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, packed=packed, meta=meta, dtz=key, group=grp, ending=end)
    won = key >= 1; inf = key == -1
    hm = np.minimum(meta[:, 6], 100)
    print(f"\n=== {args.out}: {len(rows)} rows [{time.time()-t0:.0f}s] ===")
    print(f"  won {won.sum()} | mate {(key==0).sum()} | INF {inf.sum()} | halfmove: "
          f"min {hm.min()} med {int(np.median(hm))} max {hm.max()}")
    # sanity: fraction INF should RISE with halfmove (draw surface)
    for lo, hi in [(0, 20), (40, 60), (80, 100)]:
        m = (hm >= lo) & (hm < hi)
        if m.sum(): print(f"  halfmove [{lo},{hi}): INF-fraction {100*inf[m].mean():.0f}% (n={int(m.sum())})")
    print("DONE gen_clock_child_data", flush=True)


if __name__ == "__main__":
    main()
