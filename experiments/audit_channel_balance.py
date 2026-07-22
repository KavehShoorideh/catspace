#!/usr/bin/env python
"""experiments/audit_channel_balance.py -- the pre-launch data audit that would have caught
the 25k veto-gate confound BEFORE training (JOURNAL 2026-07-23): are the multichannel
regime sources PHASE-BALANCED (piece-count distributions overlapping), and how skewed are
the opponent-model cohorts? Pure static data check -- no model, no training.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def piece_counts(packed: np.ndarray) -> np.ndarray:
    b = packed.astype(np.uint64).view(np.uint8)
    return np.unpackbits(b.reshape(len(packed), -1), axis=1).sum(1).astype(np.int16)


def hist_line(pc: np.ndarray) -> str:
    bins = [(2, 6), (7, 10), (11, 16), (17, 24), (25, 32)]
    tot = len(pc)
    parts = [f"{lo}-{hi}p {100*((pc >= lo) & (pc <= hi)).mean():4.1f}%" for lo, hi in bins]
    return f"n={tot:>9,}  " + "  ".join(parts) + f"  median {int(np.median(pc))}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=200_000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    sources = [
        ("regime0 human (4gb)", "data/shards/lichess_db_standard_rated_2019-01.prefix4gb"),
        ("regime1 random-mid", "data/shards/regime_random_v1"),
        ("regime1b random-END", "data/shards/regime_random_endgame_v1"),
        ("regime2 sf_cont", "data/shards/sf_cont_endgame_v1"),
    ]
    print("VERDICT CHANNEL_BALANCE  piece-count distribution per regime source:")
    dists = {}
    for name, d in sources:
        files = sorted(glob.glob(str(Path(d) / "*.npz")))
        if not files:
            print(f"  {name:22s} MISSING ({d})")
            continue
        pcs = []
        for f in files[:3]:
            z = np.load(f)
            pk = z["packed"]
            idx = rng.choice(len(pk), min(args.sample // min(len(files), 3), len(pk)), replace=False)
            pcs.append(piece_counts(pk[idx]))
        pc = np.concatenate(pcs)
        dists[name] = pc
        print(f"  {name:22s} {hist_line(pc)}")
    # overlap coefficient between the channels that feed the veto gap
    def overlap(a, b):
        ha = np.histogram(a, bins=np.arange(2, 34))[0] / len(a)
        hb = np.histogram(b, bins=np.arange(2, 34))[0] / len(b)
        return float(np.minimum(ha, hb).sum())
    for k1, k2 in [("regime1 random-mid", "regime2 sf_cont"),
                   ("regime1b random-END", "regime2 sf_cont")]:
        if k1 in dists and k2 in dists:
            print(f"  OVERLAP({k1} vs {k2}) = {overlap(dists[k1], dists[k2]):.2f}  "
                  f"(1.0 = identical phase support; the gap signal needs high overlap)")

    ms = Path("data/derived/move_selection_v1.npz")
    if ms.exists():
        z = np.load(ms)
        co = z["cohort"]
        u, c = np.unique(co, return_counts=True)
        print("VERDICT COHORT_BALANCE  move-selection rows per Elo bin: "
              + "  ".join(f"bin{int(b)} {n:,}" for b, n in zip(u, c))
              + f"  (max/min = {c.max()/max(c.min(),1):.0f}x; engine cohorts: 0 rows — planned)")


if __name__ == "__main__":
    main()
