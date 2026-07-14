#!/usr/bin/env python
"""
experiments/fallibility_prior.py — Stage 4: MEASURED population fallibility prior.

Kaveh: hand-coded eps(clock) is wrong (some players are calm under pressure).
This measures blunder rate vs (mover Elo bin, mover clock bucket) directly from
annotated Lichess rows (eval_cp present). Blunder = mover's move drops their
win-prob by > --thresh (lichess logistic on White-POV cp, signed by mover).
Offline ANALYSIS of engine annotations only -- never a training label (audit rule).

Outputs the prior table + held-out calibration (predicted vs actual rate per
bucket, ECE with bootstrap CI) and a CI-real test that the prior beats a
global-constant eps on held-out log-likelihood.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catspace.nn.features import elo_bin, clock_bucket, winprob_cp, N_ELO_BINS, N_CLOCK_BINS


def scan(shard_dir, cap_games):
    """Yield (elo_bin, clock_bucket, blunder01) per annotated mover-move."""
    out = []
    games = 0
    for path in sorted(Path(shard_dir).glob("shard_*.npz")):
        z = np.load(path)
        if "eval_cp" not in z.files:
            continue
        gid, cp = z["game_id"], z["eval_cp"]
        meta, clk = z["meta"], z["clock"]
        we, be = z["white_elo"], z["black_elo"]
        ends = np.flatnonzero(np.r_[np.diff(gid) != 0, True])
        starts = np.r_[0, ends[:-1] + 1]
        for s, e in zip(starts, ends):
            if not np.isfinite(cp[s:e + 1]).all():
                continue
            wp = winprob_cp(cp[s:e + 1])
            for i in range(s, e):
                stm_white = meta[i][0] == 0            # mover of the move i -> i+1
                dwp = (wp[i + 1 - s] - wp[i - s]) * (1 if stm_white else -1)
                elo = we[i] if stm_white else be[i]
                out.append((int(elo_bin(np.array([elo]))[0]),
                            int(clock_bucket(np.array([clk[i]]))[0]),
                            1 if dwp < -0.15 else 0))
            games += 1
            if games >= cap_games:
                return out, games
    return out, games


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default="data/shards/lichess_db_standard_rated_2019-01.prefix1gb")
    ap.add_argument("--cap-games", type=int, default=20000)
    ap.add_argument("--out", default="artifacts/experiments/fallibility_prior.json")
    args = ap.parse_args()

    rows, games = scan(args.shards, args.cap_games)
    rows = np.array(rows)
    print(f"{len(rows)} annotated moves from {games} games")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    hold, train = rows[idx[:len(rows) // 5]], rows[idx[len(rows) // 5:]]

    # prior table with Laplace smoothing
    tbl = np.zeros((N_ELO_BINS, N_CLOCK_BINS)); cnt = np.zeros_like(tbl)
    for e, c, b in train:
        tbl[e, c] += b; cnt[e, c] += 1
    prior = (tbl + 1) / (cnt + 2)
    g_rate = (train[:, 2].sum() + 1) / (len(train) + 2)

    # held-out: log-likelihood prior vs global-constant, bootstrap CI on the diff
    p_prior = np.clip(prior[hold[:, 0], hold[:, 1]], 1e-4, 1 - 1e-4)
    y = hold[:, 2]
    ll_p = y * np.log(p_prior) + (1 - y) * np.log(1 - p_prior)
    ll_g = y * np.log(g_rate) + (1 - y) * np.log(1 - g_rate)
    diff = ll_p - ll_g
    bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # ECE over populated buckets
    ece, tot = 0.0, 0
    for e in range(N_ELO_BINS):
        for c in range(N_CLOCK_BINS):
            m = (hold[:, 0] == e) & (hold[:, 1] == c)
            if m.sum() >= 30:
                ece += m.sum() * abs(y[m].mean() - prior[e, c]); tot += m.sum()
    ece = ece / max(tot, 1)
    print(f"VERDICT FALLIBILITY global_rate={g_rate:.3f}  "
          f"prior-vs-constant dLL={diff.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}] "
          f"[{'SIGNIFICANT' if lo > 0 else 'ns'}]  ECE={ece:.3f}")
    # does blunder rate actually RISE with time pressure? (sanity, per elo band)
    for e in [3, 5, 7]:
        r = [f"{prior[e,c]:.2f}" for c in range(N_CLOCK_BINS - 1)]
        print(f"  elo_bin {e}: blunder rate by clock bucket (SHORT->long): {r}")
    Path(args.out).write_text(json.dumps(dict(prior=prior.tolist(), games=games,
                                              global_rate=float(g_rate))))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
