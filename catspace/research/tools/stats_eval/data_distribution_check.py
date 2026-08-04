#!/usr/bin/env python
"""catspace/research/tools/stats_eval/data_distribution_check.py -- validate DATA DISTRIBUTION & EVENNESS before any
full training (Kaveh: dot the i's on data distribution/evenness first). For a lichess/engine
position-shard set, reports per-GAME (deduped by game_id): count, positions/game, the STRENGTH
distribution (Elo bands -- is it even or skewed?), the OUTCOME balance (W/D/L), and the game-PHASE
coverage (ply). Flags evenness problems (skew, missing bands, outcome imbalance) that would bias
the committor / player-embedding before we spend a full run on them.
"""
from __future__ import annotations

import argparse, glob, os, sys
from pathlib import Path

import numpy as np
from catspace.io import paths


ELO_BANDS = [0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 4000]


def _bar(frac, width=30):
    n = int(round(frac * width))
    return "#" * n + "-" * (width - n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default=paths.shards("lichess_db_standard_rated_2019-01.full"))
    ap.add_argument("--max-shards", type=int, default=3)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.shards, "*.npz")))[:args.max_shards]
    if not files:
        print(f"no .npz shards under {args.shards}"); return
    print(f"[dist-check] {len(files)} shard(s) from {args.shards}")

    gid, welo, belo, res, ply = [], [], [], [], []
    for f in files:
        z = np.load(f)
        gid.append(z["game_id"]); welo.append(z["white_elo"]); belo.append(z["black_elo"])
        res.append(z["result"]); ply.append(z["ply"])
    gid = np.concatenate(gid); welo = np.concatenate(welo).astype(np.int32)
    belo = np.concatenate(belo).astype(np.int32); res = np.concatenate(res).astype(np.int32)
    ply = np.concatenate(ply).astype(np.int32)
    n_pos = len(gid)

    # per-GAME dedup (first row of each game_id)
    _, first = np.unique(gid, return_index=True)
    g_w, g_b, g_r = welo[first], belo[first], res[first]
    n_games = len(first)
    ppg = np.bincount(np.searchsorted(np.unique(gid), gid))  # positions per game
    print(f"\npositions {n_pos:,} | games {n_games:,} | positions/game med {int(np.median(ppg))} "
          f"mean {ppg.mean():.1f} max {ppg.max()}")

    # OUTCOME balance (per game). result convention: +1 White win / 0 draw / -1 Black win (verify).
    vals, cnts = np.unique(g_r, return_counts=True)
    print("\nOUTCOME (per game):")
    lab = {1: "White win", 0: "draw", -1: "Black win"}
    for v, c in zip(vals, cnts):
        print(f"  {lab.get(int(v), v):10} {c/n_games:5.1%} {_bar(c/n_games)} ({c:,})")

    # STRENGTH distribution (per game, min of the two Elos = the weaker player's band).
    print("\nSTRENGTH (per game, by min(White,Black) Elo band) -- EVEN? or skewed to ~1500?")
    band = np.minimum(g_w, g_b)
    h, _ = np.histogram(band, bins=ELO_BANDS)
    for i in range(len(ELO_BANDS) - 1):
        lo, hi = ELO_BANDS[i], ELO_BANDS[i + 1]
        f = h[i] / n_games
        flag = "  <-- SPARSE" if f < 0.03 else ""
        print(f"  {lo:>4}-{hi:<4} {f:5.1%} {_bar(f)} ({h[i]:,}){flag}")
    # evenness metric: entropy of the band histogram / max entropy (1.0 = perfectly even)
    p = h / h.sum(); p = p[p > 0]
    evenness = float(-(p * np.log(p)).sum() / np.log(len(ELO_BANDS) - 1))
    print(f"  strength EVENNESS (norm. entropy, 1.0=even): {evenness:.2f}  "
          f"({'skewed -> will need rebalancing/engine tails' if evenness < 0.8 else 'reasonably even'})")

    # PHASE coverage (ply distribution across positions)
    print("\nPHASE (ply, all positions):")
    for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 999)]:
        f = float(((ply >= lo) & (ply < hi)).mean())
        print(f"  ply {lo:>3}-{hi:<3} {f:5.1%} {_bar(f)}")

    print("\nNOTE: shards carry Elo (strength) + result + game_id/ply, NOT player NAMES -> z can be "
          "RATING-conditioned now; per-individual z needs re-processing raw PGN. Engine tails "
          "(CCRL/fastchess strong end, Maia-bot weak end) fill the strength EVENNESS gap.")
    print("DONE data_distribution_check", flush=True)


if __name__ == "__main__":
    main()
