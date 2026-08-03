#!/usr/bin/env python
"""catspace/research/components/planner/approaches/endgame_groundtruth/experiments/gen_tb_policy_data.py — label each DTM position with its
TABLEBASE-OPTIMAL move (Kaveh 2026-07-19 autonomous). Value-navigation (committor
0.567, board-DTM 0.533) is capped by KRRvKBP DTM being hard to regress (0.29).
Predicting the best MOVE is a ranking problem -- often easier than the exact DTM
value -- and it teaches conversion TECHNIQUE (the AlphaZero recipe). Adds a
`move_idx` (0..4095 from-to) column to the DTM dataset, in parallel.

Usage:
  .venv/bin/python catspace/research/components/planner/approaches/endgame_groundtruth/experiments/gen_tb_policy_data.py --workers 9 \
    --in data/derived/dtm_endgame.npz --out data/derived/dtm_policy.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.policy_head import move_index
from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import TB, tb_best_move
from catspace.io import paths


def _chunk(task):
    packed, meta, syzygy_dir = task
    tb = TB(syzygy_dir)
    idxs = np.full(len(packed), -1, dtype=np.int32)
    for i in range(len(packed)):
        b = board_from_packed(packed[i], meta[i])
        m = tb_best_move(b, tb)
        if m is not None:
            idxs[i] = move_index(m)
    tb.close()
    return idxs


def main():
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default=paths.derived("dtm_endgame.npz"))
    ap.add_argument("--out", default=paths.derived("dtm_policy.npz"))
    ap.add_argument("--syzygy-dir", default=str(paths.syzygy_dir()))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    dz = np.load(args.inp)
    packed, meta = dz["packed"], dz["meta"]
    n, W = len(packed), max(1, args.workers)
    bounds = np.linspace(0, n, W + 1, dtype=int)
    tasks = [(packed[bounds[i]:bounds[i + 1]], meta[bounds[i]:bounds[i + 1]], args.syzygy_dir)
             for i in range(W) if bounds[i + 1] > bounds[i]]
    with ProcessPoolExecutor(max_workers=W) as ex:
        parts = list(ex.map(_chunk, tasks))
    move_idx = np.concatenate(parts)
    ok = move_idx >= 0
    np.savez(args.out, packed=packed, meta=meta, dtm=dz["dtm"], material=dz["material"],
             move_idx=move_idx)
    print(f"VERDICT TB_POLICY_DATA n={n} labeled={int(ok.sum())} ({100*ok.mean():.1f}%) -> {args.out}")


if __name__ == "__main__":
    main()
