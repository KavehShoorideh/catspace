#!/usr/bin/env python
"""catspace/research/tools/embeddings/embed_checkpoints.py -- Stage 1b: phi-embed the mined checkpoint corpus
and assign each checkpoint its structure id (v0 "atom" = nearest region centroid of the
v4 table bank; the sparse atom layer is deferred per the paper's own representation
ladder). Adds: cx_phi, ck_phi, ck_region to the miner npz -> new file."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np

from catspace.research.components.encoder.approaches.reachability_field.src import ReachabilityField                          # noqa: E402
from catspace.io import paths


def embed(rf, fens, bs=1024):
    from lczerolens import LczeroBoard
    out = np.zeros((len(fens), 64), np.float32)
    for i in range(0, len(fens), bs):
        boards = [LczeroBoard(f) for f in fens[i:i + bs]]
        out[i:i + bs] = rf.phi(boards).cpu().numpy()
        if i % (bs * 40) == 0:
            print(f"  {i:,}/{len(fens):,}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=paths.derived("checkpoints/checkpoints_v1_full.npz"))
    ap.add_argument("--table", default=paths.reach("region_table_v4.npz"))
    ap.add_argument("--out", default=paths.derived("checkpoints/checkpoints_v1_emb.npz"))
    args = ap.parse_args()
    t0 = time.time()
    d = dict(np.load(args.data, allow_pickle=True))
    bank = np.load(args.table, allow_pickle=True)["regions"].astype(np.float32)
    rf = ReachabilityField()
    print(f"[embed] {len(d['ck_fen']):,} checkpoints + {len(d['cx_fen']):,} contexts")
    ck_phi = embed(rf, d["ck_fen"])
    cx_phi = embed(rf, d["cx_fen"])
    d2 = (ck_phi * ck_phi).sum(1)[:, None] + (bank * bank).sum(1)[None, :] \
        - 2.0 * ck_phi @ bank.T
    ck_region = d2.argmin(1).astype(np.int32)
    occ = np.bincount(ck_region, minlength=len(bank))
    print(f"AUDIT checkpoint regions: {int((occ > 0).sum())}/{len(bank)} occupied | "
          f"top-region share {occ.max()/max(occ.sum(),1):.2%} | "
          f"median occupancy (occupied) {np.median(occ[occ > 0]):.0f}")
    np.savez_compressed(args.out, **d, ck_phi=ck_phi, cx_phi=cx_phi, ck_region=ck_region,
                        bank=bank)
    print(f"wrote {args.out} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
