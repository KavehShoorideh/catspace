#!/usr/bin/env python
"""experiments/m3_build_region_table.py -- offline region x band table for the M3 subgoal API.

Assigns the M2a SF-labeled positions (70k, phi + committor_before/after + mover_loss) to the v2
bank's nearest centroid and aggregates per (region, mover-Elo band): SF-refereed crossing rate
(mover_loss >= --thr), mean mover-POV committor, count. TRAIN-half games only (game hash even) --
the odd half is the held-out set for experiments/m3_subgoal_gates.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="data/derived/transition_data_labeled.npz")
    ap.add_argument("--reach", default="data/derived/reach/reach_v2.npz")
    ap.add_argument("--out", default="data/derived/reach/region_table_v1.npz")
    ap.add_argument("--thr", type=float, default=0.2)
    ap.add_argument("--band-edges", type=float, nargs="+", default=[1500.0])
    ap.add_argument("--regions", type=int, default=0,
                    help="0 = use the field bank; K>0 = fit a FINER k-means on the labeled train half")
    args = ap.parse_args()

    d = dict(np.load(args.labeled, allow_pickle=True))
    z = np.load(args.reach, allow_pickle=True)
    train = (d["game"].astype(np.int64) % 2 == 0)                        # even games -> table
    phi = d["phi"][train].astype(np.float32)
    if args.regions > 0:
        from sklearn.cluster import KMeans
        bank = KMeans(n_clusters=args.regions, n_init=3, random_state=0).fit(
            phi).cluster_centers_.astype(np.float32)
    else:
        bank = z["bank"].astype(np.float32)                              # (G,64)
    G, B = len(bank), len(args.band_edges) + 1
    d2 = (phi * phi).sum(1)[:, None] + (bank * bank).sum(1)[None, :] - 2.0 * phi @ bank.T
    region = d2.argmin(1)
    band = np.searchsorted(np.asarray(args.band_edges), d["elo_mover"][train], side="right")
    crossing = (d["mover_loss"][train] >= args.thr).astype(np.float64)
    committor = d["committor_before"][train].astype(np.float64)

    rate = np.zeros((G, B)); qual = np.zeros((G, B)); cnt = np.zeros((G, B), np.int64)
    for b in range(B):
        m = band == b
        cnt[:, b] = np.bincount(region[m], minlength=G)
        with np.errstate(invalid="ignore"):
            rate[:, b] = np.bincount(region[m], weights=crossing[m], minlength=G) \
                / np.maximum(cnt[:, b], 1)
            qual[:, b] = np.bincount(region[m], weights=committor[m], minlength=G) \
                / np.maximum(cnt[:, b], 1)
    # shrink empty/thin cells toward the band mean (no fabricated certainty in sparse regions)
    for b in range(B):
        gm = crossing[band == b].mean(); qm = committor[band == b].mean()
        w = cnt[:, b] / (cnt[:, b] + 10.0)
        rate[:, b] = w * rate[:, b] + (1 - w) * gm
        qual[:, b] = w * qual[:, b] + (1 - w) * qm

    print(f"AUDIT: {train.sum():,} table rows -> {G} regions x {B} bands | "
          f"cells with n<10: {(cnt < 10).sum()}/{G*B} (shrunk) | "
          f"rate range {rate.min():.3f}-{rate.max():.3f} | base by band "
          f"{[round(crossing[band==b].mean(),3) for b in range(B)]}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, regions=bank, crossing_rate=rate, committor_mean=qual, count=cnt,
                        band_edges=np.asarray(args.band_edges), thr=args.thr,
                        meta_labeled=args.labeled, meta_reach=args.reach)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
