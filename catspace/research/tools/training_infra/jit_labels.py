"""Just-in-time Stockfish labelling with BOUNDED storage.

Kaveh 2026-08-24: "stockfish is pretty fast. I wonder if we can just generate the
labels right when we need them, cache them, and then delete them -- we get to use them
once and then throw them away because we don't have enough storage."

The arithmetic behind the design:

  * one SF label at depth 10 costs ~400 ms and yields ~33 scored moves AND ~33
    principal variations (~6 plies each) -- the trajectory data the operator wants.
  * keeping PVs costs +1.8 KB/position: 2.41x the payload, +7.9 GB over the 4.2M sf
    labels we hold. That does not fit.
  * but the trainer only consumes ~17 sf rows per step at ~2.5 s/step, so a
    2048-row shard lasts ~120 steps (~5 min). A handful of parallel SF workers
    outproduces that comfortably.

So: never persist PVs in the oracle DB. Stream shards into a RING, let the trainer
read them, and delete the oldest once the ring is full. Storage stays flat at
(ring_size x shard_bytes) regardless of how much we generate.

This module owns only the RING. Generation reuses the existing oracle path so there
is one definition of what a label is.
"""
import argparse
import glob
import os
import time


def ring_evict(shard_dir, keep, min_age_s=90.0, dry_run=False):
    """Delete the oldest shards beyond `keep`, never touching anything younger than
    `min_age_s`.

    The age floor matters: the trainer rescans on its own schedule and memory-maps
    what it finds, so deleting a shard it is about to read (or is holding) trades a
    storage problem for a correctness problem. Age is the cheap proxy for 'the
    trainer has had its chance'.
    """
    sh = sorted(glob.glob(os.path.join(shard_dir, "*.npz")), key=os.path.getmtime)
    if len(sh) <= keep:
        return 0, 0
    now = time.time()
    doomed = [f for f in sh[:-keep] if now - os.path.getmtime(f) > min_age_s]
    freed = 0
    for f in doomed:
        try:
            freed += os.path.getsize(f)
            if not dry_run:
                os.remove(f)
        except OSError:
            pass
    return len(doomed), freed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default="data/derived/gko1_shards")
    ap.add_argument("--keep", type=int, default=200,
                    help="ring size in shards; storage is flat at keep x ~1.6 MB")
    ap.add_argument("--min-age", type=float, default=90.0,
                    help="never delete a shard younger than this many seconds")
    ap.add_argument("--every", type=float, default=60.0,
                    help="seconds between sweeps (0 = one sweep and exit)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    total_n = total_b = 0
    while True:
        n, b = ring_evict(a.shards, a.keep, a.min_age, a.dry_run)
        total_n += n
        total_b += b
        held = len(glob.glob(os.path.join(a.shards, "*.npz")))
        if n:
            print(f"[jit] evicted {n} shard(s), {b/1e6:.1f} MB | ring holds {held}"
                  f"/{a.keep} | lifetime {total_n} shards, {total_b/1e9:.2f} GB reclaimed",
                  flush=True)
        if a.every <= 0:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
