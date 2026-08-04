#!/usr/bin/env python
"""experiments/build_field_subset.py -- Kaveh 2026-08-03: materialize a CONTIGUOUS, RAM-resident
subset of the combined field data, because training on the full set is 98% disk-bound.

The measurement that motivates this (printed by train_iqe_head.py --timing-every):

    [t] step 25/8000 | 4.67s/step (gather 4.57s = 98%, compute 0.10s)
                     | 17,654 rows = 552MB/step -> 121 MB/s | ETA 10.4h

Each step randomly gathers ~17.6k rows x 32KB = 552MB scattered across 71GB of trunk features on
a 36GB machine, so essentially every read is a page fault to disk at ~121 MB/s while the GPU sits
idle. Actual compute is 0.10 s/step: fit the working set in RAM and the same 8000 steps take
~30 minutes instead of 10.4 hours.

What this does:
  * UNIFORM random sample of rows -- not stratified. Uniform preserves the joint distribution of
    (source, basin, ply, terminal-ness) EXACTLY in expectation, so nothing about the committor
    being estimated is skewed. Stratifying to over-sample terminals would bias the basin
    probabilities toward endgames, which is precisely the quantity of interest.
  * writes ONE contiguous fp16 .npy in SORTED source/row order, so building it streams the two
    source memmaps sequentially instead of seeking (the same random-read cost this exists to
    avoid).
  * emits a matching metadata npz whose `source` is all-zero and `local_row` is 0..N-1, so
    train_iqe_head.py reads it through the unchanged DualFeats path.

Sizing: rows x 32KB. 600k -> 18.3GB, which leaves headroom on a 36GB box for the OS, the process
and the npz. Going much above that risks spilling back to disk and losing the entire point.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.losses import WIN, DRAW, LOSS

ROW_BYTES = 256 * 8 * 8 * 2                                  # fp16 trunk features


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combined", default="data/derived/field_combined_v1.npz")
    ap.add_argument("--n", type=int, default=600_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--out-feats", default="data/derived/trunk_feats/t1-256x10__combined_sub600k.npy")
    ap.add_argument("--out", default="data/derived/field_combined_sub600k.npz")
    ap.add_argument("--meta-only", action="store_true",
                    help="rewrite only the metadata npz (features already materialized)")
    args = ap.parse_args()
    t0 = time.time()

    z = np.load(args.combined, allow_pickle=True)
    meta = eval(str(z["_meta"][0]))
    N = len(z["y"])
    n = min(args.n, N)
    rng = np.random.default_rng(args.seed)
    take = np.sort(rng.choice(N, n, replace=False))           # sorted -> sequential source reads

    src = z["source"][take]
    loc = z["local_row"][take]
    gb = n * ROW_BYTES / 1024 ** 3
    print(f"[subset] {n:,} of {N:,} rows ({100*n/N:.1f}%) -> {gb:.1f}GB contiguous fp16")

    if args.meta_only:
        print("  --meta-only: reusing the existing feature file")
    mm = [np.load(p, mmap_mode="r") for p in meta["feats"]]
    shape = mm[0].shape[1:]
    out = (np.load(args.out_feats, mmap_mode="r") if args.meta_only else
           np.lib.format.open_memmap(args.out_feats, mode="w+", dtype=np.float16,
                                     shape=(n, *shape)))
    done = 0
    for s in ([] if args.meta_only else (0, 1)):
        m = np.flatnonzero(src == s)
        if not len(m):
            continue
        rows = loc[m]                                        # already ascending within a source
        for i in range(0, len(m), args.chunk):
            sl = m[i:i + args.chunk]
            out[sl] = mm[s][rows[i:i + args.chunk]]
            done += len(sl)
            if (i // args.chunk) % 10 == 0:
                el = time.time() - t0
                print(f"  {done:,}/{n:,} [{el:.0f}s, {done/max(el,1e-9):.0f} rows/s]", flush=True)
    if not args.meta_only:
        out.flush()
    del out

    keep = {k: z[k][take] for k in
            ("y", "n_to_end", "is_terminal", "is_tail", "game", "ply", "dtz", "result", "ending")}
    keep["source"] = np.zeros(n, np.int8)                    # single materialized file
    # `source` must be zeroed so DualFeats reads the ONE materialized file -- but the human/SF
    # split is the entire point of the downstream basin comparison, so preserve it separately.
    # (Caught after the first build: charting had no way to tell the two populations apart.)
    keep["orig_source"] = src.astype(np.int8)
    keep["local_row"] = np.arange(n, dtype=np.int64)
    sub_meta = dict(meta)
    sub_meta["feats"] = [args.out_feats, args.out_feats]
    sub_meta["subset_of"] = args.combined
    sub_meta["subset_n"] = n
    np.savez(args.out, **keep, _meta=np.array([repr(sub_meta)], dtype=object))

    # Report the distribution against the parent so "uniform preserved it" is checked, not assumed.
    print(f"\n  {'field':14s} {'parent':>10s} {'subset':>10s}")
    for name, pm, sm in [("human rows", z["source"] == 0, src == 0),
                         ("SF rows", z["source"] == 1, src == 1),
                         ("terminals", z["is_terminal"], keep["is_terminal"]),
                         ("tail rows", z["is_tail"], keep["is_tail"]),
                         ("basin=win", z["y"] == WIN, keep["y"] == WIN),
                         ("basin=draw", z["y"] == DRAW, keep["y"] == DRAW),
                         ("basin=loss", z["y"] == LOSS, keep["y"] == LOSS)]:
        print(f"  {name:14s} {100*pm.mean():>9.2f}% {100*sm.mean():>9.2f}%")
    print(f"  terminals in subset: {int(keep['is_terminal'].sum()):,} | "
          f"tail rows: {int(keep['is_tail'].sum()):,}")
    print(f"wrote {args.out_feats} ({gb:.1f}GB) + {args.out}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
