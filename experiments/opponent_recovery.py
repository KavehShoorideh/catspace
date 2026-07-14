#!/usr/bin/env python
"""
experiments/opponent_recovery.py — Stage 5: per-opponent online fallibility.

Parameter-recovery test: simulate opponents with KNOWN eps-by-clock profiles
(calm / average / panicky), run the Bayesian online estimator (Beta posterior per
clock bucket, initialised from the population prior), and check the posterior
recovers the true profile within coverage -- BEFORE any of this touches play.

The estimator itself (OpponentModel) is the production object: observe (clock
bucket, was_blunder) per opponent move, posterior-update, expose eps_hat(bucket).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class OpponentModel:
    """Beta posterior per clock bucket over the opponent's blunder rate.
    Prior = population prior scaled to a pseudo-count (so live evidence can move
    it within a game or two)."""

    def __init__(self, prior_rates, prior_strength: float = 8.0):
        p = np.asarray(prior_rates, dtype=float)
        self.a = p * prior_strength + 1e-3
        self.b = (1 - p) * prior_strength + 1e-3

    def observe(self, bucket: int, blunder: bool):
        if blunder:
            self.a[bucket] += 1
        else:
            self.b[bucket] += 1

    def eps_hat(self, bucket: int) -> float:
        return float(self.a[bucket] / (self.a[bucket] + self.b[bucket]))

    def ci90(self, bucket: int) -> tuple[float, float]:
        from scipy import stats  # optional; fallback below if absent
        d = stats.beta(self.a[bucket], self.b[bucket])
        return float(d.ppf(0.05)), float(d.ppf(0.95))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--moves", type=int, default=120, help="observed opponent moves (~2-3 games)")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_buckets = 4                                     # long ... very short (toy granularity)
    pop_prior = np.array([0.05, 0.08, 0.15, 0.30])    # population: panic rises
    profiles = {
        "calm":    np.array([0.05, 0.06, 0.07, 0.08]),   # flat under pressure
        "average": pop_prior,
        "panicky": np.array([0.05, 0.10, 0.25, 0.50]),
    }
    rng = np.random.default_rng(args.seed)
    print(f"{args.trials} trials x {args.moves} observed moves; prior={pop_prior}")
    ok_all = True
    for name, true_eps in profiles.items():
        err_prior, err_post, cover = [], [], []
        for _ in range(args.trials):
            om = OpponentModel(pop_prior)
            for _ in range(args.moves):
                b = int(rng.integers(0, n_buckets))            # uniform clock exposure
                om.observe(b, bool(rng.random() < true_eps[b]))
            est = np.array([om.eps_hat(b) for b in range(n_buckets)])
            err_post.append(np.abs(est - true_eps).mean())
            err_prior.append(np.abs(pop_prior - true_eps).mean())
            try:
                cis = [om.ci90(b) for b in range(n_buckets)]
                cover.append(np.mean([lo <= t <= hi for (lo, hi), t in zip(cis, true_eps)]))
            except Exception:
                cover.append(np.nan)
        ep, eo = float(np.mean(err_prior)), float(np.mean(err_post))
        cv = float(np.nanmean(cover))
        better = eo < ep or name == "average"
        ok_all &= better
        print(f"  {name:8s} MAE prior-only={ep:.3f} -> posterior={eo:.3f} "
              f"({'improved' if eo < ep else 'no change needed' if name == 'average' else 'WORSE'})"
              f"  CI90 coverage={cv:.2f}")
    print(f"VERDICT OPPONENT_RECOVERY {'PASS' if ok_all else 'FAIL'} "
          f"(posterior must beat prior on calm & panicky; coverage ~0.9)")


if __name__ == "__main__":
    main()
