#!/usr/bin/env python
"""experiments/gen_successor_edges.py -- LEGAL-SUCCESSOR edges for the geometry
objective (Kaveh 2026-07-20: the field captures what COULD be played -- legal
reachability -- not what is USUALLY played, which is the separate policy model).

For each near-mate state s we sample up to K legal successors s' (single plies) and
store (s, s') edges. Training pins d(s->s') ~ 1 (a legal move = one step); the ONE-WAY
structure then EMERGES from the directed graph -- an irreversible move's reverse is
not a legal edge, so it is never pinned and the negatives push it large. No capture /
pawn-death detection, no infinite-distance loss term.

Board state only: the halfmove clock + repetition are NOT part of this reachability
(they are separate monotone potentials for the planner), so shuffle-equivalent
positions stay identifiable. We store packed/meta; the trainer zeroes the clock+rep
planes into the distance tower.

Saves p_packed/p_meta (parent s), c_packed/c_meta (successor s').

Usage:
  .venv/bin/python experiments/gen_successor_edges.py --workers 9 --k 8 \
    --in data/derived/lichess_nearmate.npz --out data/derived/successor_edges.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed


def _chunk(task):
    packed, meta, k, seed = task
    rng = np.random.default_rng(seed)
    pp, pm, cp, cm, ir = [], [], [], [], []
    for i in range(len(packed)):
        b = board_from_packed(packed[i], meta[i])
        if b.is_game_over():
            continue
        mv = list(b.legal_moves)
        if not mv:
            continue
        idx = rng.permutation(len(mv))[:k]                 # up to k random legal successors
        for j in idx:
            m = mv[int(j)]
            # irreversible = the rule-defined set: pawn moves, captures, castling, AND
            # any move that reduces castling rights / gives up en passant. Its reverse
            # (child->parent) is a HARD negative -- pushed large (Kaveh 2026-07-20).
            irrev = b.is_irreversible(m)
            c = b.copy(stack=False); c.push(m)
            pp.append(packed[i]); pm.append(meta[i])
            cp.append(encode_packed(c)); cm.append(encode_meta(c)); ir.append(irrev)
    if not pp:
        return None
    return (np.stack(pp), np.stack(pm), np.stack(cp), np.stack(cm), np.array(ir, dtype=bool))


def main():
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/derived/lichess_nearmate.npz")
    ap.add_argument("--out", default="data/derived/successor_edges.npz")
    ap.add_argument("--k", type=int, default=8, help="successors sampled per state")
    ap.add_argument("--won-only", action="store_true", help="won near-mate states only")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    W = max(1, args.workers)
    dz = np.load(args.inp)
    packed, meta = dz["packed"], dz["meta"]
    if args.won_only and "dtm" in dz:
        sel = dz["dtm"] > 0
        packed, meta = packed[sel], meta[sel]
    n = len(packed)
    bnd = np.linspace(0, n, W + 1, dtype=int)
    tasks = [(packed[bnd[i]:bnd[i + 1]], meta[bnd[i]:bnd[i + 1]], args.k, i)
             for i in range(W) if bnd[i + 1] > bnd[i]]
    t0 = time.time()
    parts = []
    with ProcessPoolExecutor(max_workers=W) as ex:
        for r in ex.map(_chunk, tasks):
            if r is not None:
                parts.append(r)
    keys = ["p_packed", "p_meta", "c_packed", "c_meta", "irrev"]
    out = {kk: np.concatenate([p[j] for p in parts]) for j, kk in enumerate(keys)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"[stage] {len(out['p_packed'])} successor edges from {n} states "
          f"(k={args.k}, {100*out['irrev'].mean():.1f}% irreversible): {time.time()-t0:.1f}s")
    print(f"VERDICT SUCCESSOR_EDGES n={len(out['p_packed'])} -> {args.out}")


if __name__ == "__main__":
    main()
