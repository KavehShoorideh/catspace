#!/usr/bin/env python
"""experiments/gen_escape_data.py -- training data for the LEARNED constraint value
(the mate mission, Kaveh 2026-07-23: tablebase-free conversion). Labels = the black king's
escape volume (catspace.diagnostics.escape_volume -- rules-only; allowed as a DATA LABEL,
forbidden at play; the NET plays, not the flood-fill).

Positions: the toy pool (dtm_endgame) + random-walk children (off-optimal states search
actually visits) + random KRRvK/KRvK (the ladder scenarios). No tablebase anywhere.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed, encode_meta, encode_packed
from catspace.diagnostics import escape_volume
from experiments.ladder_mate import random_krrvk


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--out", default="data/derived/escape_data_v1.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    dz = np.load(args.dtm_npz)
    P, M = np.asarray(dz["packed"]), np.asarray(dz["meta"])

    PK, MT, Y = [], [], []

    def add(b):
        PK.append(encode_packed(b)); MT.append(encode_meta(b)); Y.append(escape_volume(b))

    # 1/3 pool positions, 1/3 their random-walk children, 1/3 random ladder boards
    n_pool = args.n // 3
    idx = rng.choice(len(P), min(n_pool, len(P)), replace=False)
    for i in idx:
        add(board_from_packed(P[i], M[i]))
    for i in rng.choice(len(P), args.n // 3, replace=True):
        b = board_from_packed(P[i], M[i])
        for _t in range(int(rng.integers(1, 7))):
            mv = list(b.legal_moves)
            if not mv:
                break
            b.push(mv[int(rng.integers(len(mv)))])
        if not b.is_game_over():
            add(b)
    while len(PK) < args.n:
        b = random_krrvk(rng, central=bool(rng.integers(2)))
        if b is not None:
            add(b)
    Y = np.array(Y, np.float32)
    np.savez_compressed(args.out, packed=np.stack(PK), meta=np.stack(MT), escape=Y)
    print(f"VERDICT ESCAPE_DATA n={len(PK)} escape: min {Y.min():.0f} med {np.median(Y):.0f} "
          f"max {Y.max():.0f} -> {args.out}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
