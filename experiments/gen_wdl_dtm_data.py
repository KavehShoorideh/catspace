#!/usr/bin/env python
"""experiments/gen_wdl_dtm_data.py -- S2 data (METASTABILITY_PLAN): fix the field that can't
mate (5% mate-rate; picks a DTM-reducing move only 52.7% -- coin flip -- because it was trained
only on optimal-LINE positions and never saw the OFF-optimal positions a bad move leads to).

Cure: for each winning parent (White to move, tablebase-won), emit the parent AND ALL its
children, each labeled with tablebase truth -- WDL (white_pov_value) + DTM (rollout_dtm if still
won, else INFINITE = threw the win). This gives:
  * off-optimal NEGATIVES: children that blunder the win -> dtm=INF (the stalemate/draw/loss
    repellers -- Defect 2 fix);
  * LOCAL resolution: siblings share a group_id so the trainer can rank children by true DTM
    (the move-selection signal the field lacked);
  * mate terminals dtm=0 (the collapsed attractor -- Defect 1 fix).
Plus a slice of pure DRAWN/LOST parents (dtm=INF) so the field sees basin interiors too.
Parallel workers, cache-free tablebase. dtm sentinel: >=1 won, 0 mate, -1 = INF (draw/loss).
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
from catspace.tb import TB, DEFAULT_SYZYGY, rollout_dtm
from experiments.gen_dtm_data import random_class_start
from experiments.value_fixed_point import white_pov_value

WIN_CLASSES = ["KQvK", "KRvK", "KRRvK", "KBBvK", "KBNvK", "KQvKP", "KRvKP", "KQvKR", "KRvKN"]
MIX_CLASSES = ["KRvKR", "KQvKQ", "KBvKN", "KNvK", "KBvK", "KPvKP", "KRvKB"]  # for draw/loss slice


def _emit(board, tb, out, gid):
    """append (packed, meta, dtm_sentinel, wdl, group) for a board (White-to-move winner root
    context). dtm: >=1 won plies, 0 mate, -1 = INF (not won for White)."""
    if board.is_checkmate():
        # side to move is mated. white delivered mate iff black to move.
        dtm = 0 if board.turn == chess.BLACK else -1
        out.append((encode_packed(board), encode_meta(board), dtm, 0 if dtm == 0 else -1, gid))
        return
    if board.is_game_over(claim_draw=True):
        out.append((encode_packed(board), encode_meta(board), -1, 1, gid))  # draw -> INF, wdl D
        return
    v = white_pov_value(board, tb)
    if v == 1.0:
        d = rollout_dtm(board, tb)
        d = d if (d is not None and d >= 1) else -1
        out.append((encode_packed(board), encode_meta(board), d, 2, gid))
    else:
        out.append((encode_packed(board), encode_meta(board), -1, 1 if v == 0.5 else 0, gid))


def worker(task):
    kind, classes, n, seed, syzygy, expand = task
    rng = np.random.default_rng(seed)
    tb = TB(str(syzygy), cache_db=None)
    out = []
    got = tries = 0
    gid = seed * 1_000_000
    while got < n and tries < n * 200:
        tries += 1
        cls = classes[rng.integers(0, len(classes))]
        b = random_class_start(rng, cls)
        if b is None or b.is_game_over():                # BOTH colors to move (fixes stm mismatch)
            continue
        v = white_pov_value(b, tb)
        if kind == "win":
            if v != 1.0:
                continue
            gid += 1
            _emit(b, tb, out, gid)                       # won position (either color to move) + DTM
            if expand and b.turn == chess.WHITE:         # cheap local signal: WDL-only children
                for m in b.legal_moves:                  # (child WDL identifies blunder-negatives)
                    b.push(m)
                    cv = white_pov_value(b, tb) if not b.is_game_over() else (
                        1.0 if b.is_checkmate() else 0.5)
                    d = -1 if cv != 1.0 else -2          # -2 = won child (rank via parent's group)
                    if b.is_checkmate() and b.turn == chess.BLACK:
                        d = 0
                    out.append((encode_packed(b), encode_meta(b), d, 2 if cv == 1.0 else 1, gid))
                    b.pop()
            got += 1
        else:
            if v == 1.0:
                continue
            gid += 1
            _emit(b, tb, out, gid)                        # draw/loss basin interior -> INF
            got += 1
    tb.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--win-parents", type=int, default=12000)
    ap.add_argument("--draw-loss", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="data/derived/wdl_dtm_v1.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    W = args.workers or max(1, (os.cpu_count() or 4) - 1)
    syz = str(DEFAULT_SYZYGY)
    tasks = ([("win", WIN_CLASSES, max(1, args.win_parents // W), args.seed + i, syz, False) for i in range(W)]
             + [("dl", MIX_CLASSES, max(1, args.draw_loss // W), args.seed + 100 + i, syz, False) for i in range(W)])
    print(f"[gen-wdl-dtm] {len(tasks)} chunks / {W} workers", flush=True)
    parts = []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for i, r in enumerate(ex.map(worker, tasks)):
            parts.append(r)
            print(f"  chunk {i+1}/{len(tasks)}: {len(r)} rows [{time.time()-t0:.0f}s]", flush=True)
    rows = [x for p in parts for x in p]
    packed = np.stack([r[0] for r in rows]); meta = np.stack([r[1] for r in rows])
    dtm = np.array([r[2] for r in rows], np.int32); wdl = np.array([r[3] for r in rows], np.int8)
    grp = np.array([r[4] for r in rows], np.int64)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, packed=packed, meta=meta, dtm=dtm, wdl=wdl, group=grp)
    won = dtm >= 1; mate = dtm == 0; inf = dtm == -1
    print(f"\n=== {args.out}: {len(rows)} rows [{time.time()-t0:.0f}s] ===")
    print(f"  won {won.sum()} (dtm med {int(np.median(dtm[won])) if won.any() else 0}, "
          f"max {int(dtm[won].max()) if won.any() else 0}) | mate {mate.sum()} | INF(draw/loss) {inf.sum()}")
    print(f"  wdl: W {int((wdl==2).sum())} D {int((wdl==1).sum())} L {int((wdl==0).sum())} "
          f"| groups {len(np.unique(grp))}")
    print("DONE gen_wdl_dtm_data", flush=True)


if __name__ == "__main__":
    main()
