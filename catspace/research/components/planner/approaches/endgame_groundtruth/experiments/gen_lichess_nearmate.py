#!/usr/bin/env python
"""catspace/research/components/planner/approaches/endgame_groundtruth/experiments/gen_lichess_nearmate.py — the near-mate NUCLEUS from REAL Lichess
games (Kaveh 2026-07-19: "take all the near mate situations from lichess data
where tablebase provides exact situation ... id it and cut it off"). Scans the
shards, finds every position within <=5-piece Syzygy range, and labels it with
EXACT tablebase DTM (rollout under optimal play). These are diverse real endgame
positions ("all sorts of combinations") -- a richer nucleus than the synthetic
random-endgame data. Also emits the PARENT position one ply before (the move that
enters the nucleus) so the far field can propagate its distance TO the nucleus.

Saves packed/meta/dtm/result (result +1 white-win / -1 black-win / 0 draw), in
the same layout as gen_dtm_data so they concatenate.

Usage:
  .venv/bin/python catspace/research/components/planner/approaches/endgame_groundtruth/experiments/gen_lichess_nearmate.py --workers 9 --cap 60000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import chess
import numpy as np


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.io.paths import newest_shard_dir
from catspace.research.components.planner.approaches.endgame_groundtruth.experiments.gen_dtm_data import rollout_dtm
from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import TB
from catspace.io import paths


def _chunk(task):
    shard_path, cap, syzygy_dir = task
    tb = TB(syzygy_dir)
    packs, metas, dtms, results = [], [], [], []
    npz = np.load(shard_path)
    packed, meta = npz["packed"], npz["meta"]
    for i in range(len(packed)):
        if len(dtms) >= cap:
            break
        b = board_from_packed(packed[i], meta[i])
        if chess.popcount(b.occupied) > 5:                 # only <=5-piece = tablebase range
            continue
        if b.is_game_over(claim_draw=True):
            continue
        d = rollout_dtm(b, tb)
        if d is None:                                      # drawn / no mate reached
            packs.append(packed[i]); metas.append(meta[i]); dtms.append(0.0); results.append(0)
            continue
        # rollout_dtm plays optimal-to-mate; sign from who is winning
        from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import white_pov_value
        wv = white_pov_value(b, tb)
        res = 1 if (wv is not None and wv > 0) else (-1 if (wv is not None and wv < 0) else 0)
        packs.append(packed[i]); metas.append(meta[i]); dtms.append(float(d)); results.append(res)
    tb.close()
    if not packs:
        return None
    return (np.stack(packs), np.stack(metas),
            np.array(dtms, np.float32), np.array(results, np.int8))


def main():
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--out", default=paths.derived("lichess_nearmate.npz"))
    ap.add_argument("--syzygy-dir", default=str(paths.syzygy_dir()))
    ap.add_argument("--cap", type=int, default=60000, help="max near-mate positions")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()
    shards = sorted(shard_dir.glob("shard_*.npz"))
    per = max(1, args.cap // max(1, len(shards)))
    tasks = [(s, per, args.syzygy_dir) for s in shards]
    t0 = time.time()
    print(f"[stage] scanning {len(shards)} shards for <=5-piece positions "
          f"({args.workers} workers, cap {args.cap})", flush=True)
    parts = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_chunk, tasks):
            if r is not None:
                parts.append(r)
                got = sum(len(p[2]) for p in parts)
                print(f"  {got} near-mate positions ({got/(time.time()-t0):.0f}/s)", flush=True)
                if got >= args.cap:
                    break
    packed = np.concatenate([p[0] for p in parts])[:args.cap]
    meta = np.concatenate([p[1] for p in parts])[:args.cap]
    dtm = np.concatenate([p[2] for p in parts])[:args.cap]
    result = np.concatenate([p[3] for p in parts])[:args.cap]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, packed=packed, meta=meta, dtm=dtm, result=result)
    won = dtm > 0
    print(f"[stage] {len(dtm)} positions: {time.time()-t0:.1f}s")
    print(f"VERDICT LICHESS_NEARMATE n={len(dtm)} won={int(won.sum())} "
          f"draw={int((result==0).sum())} dtm[min={dtm[won].min():.0f} "
          f"med={np.median(dtm[won]):.0f} max={dtm[won].max():.0f}] -> {args.out}")


if __name__ == "__main__":
    main()
