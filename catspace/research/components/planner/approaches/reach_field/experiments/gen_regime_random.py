#!/usr/bin/env python
"""catspace/research/components/planner/approaches/reach_field/experiments/gen_regime_random.py -- regime-1 (RANDOM WALK) branches for the multichannel
field (INQUIRY_MULTICHANNEL_FIELD.md): sample anchor positions from the human stream, roll j
plies of uniform-random legal play (both sides), and write the walks as lichess-format shards
(each walk = one game) tagged regime=1. The cheapest channel: no engine, no tablebase --
pure could-happen dynamics, the contrast background against which purposeful channels stand out.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default=paths.shards("lichess_db_standard_rated_2019-01.prefix1gb"))
    ap.add_argument("--n-walks", type=int, default=8000)
    ap.add_argument("--j", type=int, default=12)
    ap.add_argument("--out-dir", default=paths.shards("regime_random_v1"))
    ap.add_argument("--regime", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(Path(args.shards) / "shard_*.npz")))
    z = np.load(files[0])
    N = len(z["game_id"])
    rows = rng.choice(N, size=args.n_walks * 2, replace=False)
    P, M = z["packed"], z["meta"]
    WE, BE, CK = z["white_elo"], z["black_elo"], z["clock"]

    pk, mt, ply, clk, res, we, be, gid, reg = [], [], [], [], [], [], [], [], []
    made = 0
    for r in rows:
        if made >= args.n_walks:
            break
        b = board_from_packed(P[r], M[r])
        if b.is_game_over(claim_draw=True):
            continue
        walk = [b.copy(stack=False)]
        dead = False
        for _ in range(args.j):
            moves = list(walk[-1].legal_moves)
            if not moves:
                dead = True; break
            c = walk[-1].copy(stack=False); c.push(moves[int(rng.integers(len(moves)))])
            walk.append(c)
            if c.is_game_over(claim_draw=True):
                break
        if dead or len(walk) < 4:
            continue
        for t, w in enumerate(walk):
            pk.append(encode_packed(w)); mt.append(encode_meta(w))
            ply.append(t); clk.append(float(CK[r])); res.append(0)
            we.append(int(WE[r])); be.append(int(BE[r])); gid.append(made); reg.append(args.regime)
        made += 1
        if made % 2000 == 0:
            print(f"  {made}/{args.n_walks} walks  [{time.time()-t0:.0f}s]", flush=True)

    np.savez_compressed(out / "shard_000.npz",
                        packed=np.stack(pk), meta=np.stack(mt),
                        ply=np.array(ply, np.int32), clock=np.array(clk, np.float32),
                        result=np.array(res, np.int8),
                        white_elo=np.array(we, np.uint16), black_elo=np.array(be, np.uint16),
                        game_id=np.array(gid, np.uint32), regime=np.array(reg, np.int8))
    print(f"VERDICT REGIME_RANDOM walks={made} rows={len(pk)} j={args.j} regime={args.regime} "
          f"-> {out}/shard_000.npz  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
