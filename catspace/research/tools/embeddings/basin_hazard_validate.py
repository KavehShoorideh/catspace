#!/usr/bin/env python
"""catspace/research/tools/embeddings/basin_hazard_validate.py -- does h actually find hazards, or
is it two fields disagreeing?

h = q_SF - q_human is a difference of two separately-trained models. Two models ALWAYS differ, so a
non-zero h proves nothing on its own. This script runs the two checks that can kill the claim, and
prints the numbers whether or not they support it.

CHECK 1 -- PREDICTIVE. If h(s) is a hazard, then among positions humans actually reach, high-h ones
must be followed by a LARGER realized loss of score in the human game than low-h ones. This is
measured on the replayed games themselves, on held-out data the fields never trained on, using the
realized future -- no model in the loop on the outcome side:

    drop(s, k) = q_human_realized(s) - q_human_realized(s + k plies)

...except q_realized would itself be a field readout. So the outcome side uses the GAME RESULT, the
only fully model-free quantity available: for each position, the white-POV realized score
(+1/0/-1). The test is then whether the eventual result is worse (from the side that is ahead by
q_SF) in high-h positions than in low-h ones, conditioned on q_SF -- i.e. holding the objective
value of the position FIXED and asking whether h predicts the residual. Conditioning is what makes
it a test of h rather than a rediscovery that bad positions lose.

CHECK 2 -- NULL SCALE. Two fields trained on the SAME mixed data with different seeds also produce
a non-zero h. That is the noise floor. The real h must exceed it. Pass --null-data to compare.

CHECK 3 -- CALIBRATION PARITY. h is only interpretable if both fields are calibrated; a systematic
confidence difference would masquerade as hazard everywhere. Reported as each field's mean |q| and
the correlation between them, so a pure scale difference is visible as a slope, not read as signal.
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def boot_diff(a, b, n=2000, seed=0):
    """Bootstrap 95% CI for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    d = da - db
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="artifacts/experiments/basin_hazard_data.npz")
    ap.add_argument("--null-data", default="", help="npz from a NULL field pair (same data, "
                                                    "different seeds) -- the noise floor")
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--bins", type=int, default=8, help="q_SF strata to condition on")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    z = np.load(args.data)
    m = (z["ply"] >= args.min_ply) & (z["source"] == 0)     # human positions only: their hazards
    h = (z["q_sf"] - z["q_human"])[m]
    qs, qh = z["q_sf"][m], z["q_human"][m]
    ply, res, gid = z["ply"][m], z["result"][m], z["gid"][m]
    print(f"[validate] {len(h):,} human positions, ply >= {args.min_ply} [{time.time()-t0:.0f}s]")

    # ---- CHECK 3 first: it decides whether 1 and 2 are even interpretable ----------------------
    print("\nCHECK 3 -- CALIBRATION PARITY (a pure confidence gap would fake hazard everywhere)")
    print(f"  mean |q| : human {np.abs(qh).mean():.4f} | SF {np.abs(qs).mean():.4f}")
    print(f"  corr(q_human, q_SF) = {np.corrcoef(qh, qs)[0,1]:+.4f}   "
          f"slope(q_SF ~ q_human) = {np.polyfit(qh, qs, 1)[0]:+.4f}")
    print("  a slope far from 1 means the two fields differ in SCALE, and part of h is that scale "
          "difference rather than a dynamics difference. Check 1 conditions on q_SF, so it "
          "survives a scale gap; the raw magnitude of h does not.")

    # ---- CHECK 1: does h predict the realized result, holding q_SF fixed? ----------------------
    # Score from the perspective of the side the ENGINE evaluation favours, so "worse than the
    # position deserves" is a single signed number regardless of colour.
    side = np.sign(qs)
    fav = side * res.astype(np.float64)                     # +1 the favoured side won, -1 it lost
    ok = side != 0
    print(f"\nCHECK 1 -- PREDICTIVE: within a q_SF stratum, does higher h mean a worse realized "
          f"result for the side the engine favours?")
    print(f"  {'q_SF stratum':>16s} {'n':>8s} {'lo-h score':>11s} {'hi-h score':>11s} "
          f"{'difference':>11s} {'95% CI':>19s}")
    edges = np.quantile(np.abs(qs[ok]), np.linspace(0, 1, args.bins + 1))
    diffs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = ok & (np.abs(qs) >= lo) & (np.abs(qs) < hi)
        if b.sum() < 400:
            continue
        hb = h[b] * side[b]                                 # hazard TO the favoured side
        q1, q3 = np.quantile(hb, [0.25, 0.75])
        if not q1 < q3:                                     # degenerate spread -> no contrast to test
            continue
        loM, hiM = b.copy(), b.copy()
        loM[b] = hb <= q1
        hiM[b] = hb >= q3
        # h is defined so that POSITIVE = the human side gives it away, i.e. relative to the
        # favoured side the sign flips with `side`; hi-h should therefore score WORSE.
        d, cl, cu = boot_diff(fav[hiM], fav[loM], seed=args.seed)
        diffs.append((d, cl, cu))
        print(f"  {lo:>7.3f}-{hi:<8.3f} {int(b.sum()):>8,} {fav[loM].mean():>+11.3f} "
              f"{fav[hiM].mean():>+11.3f} {d:>+11.3f} [{cl:>+7.3f},{cu:>+7.3f}]")
    if diffs:
        neg = sum(1 for d, cl, cu in diffs if cu < 0)
        pos = sum(1 for d, cl, cu in diffs if cl > 0)
        print(f"\n  VERDICT check 1: {neg}/{len(diffs)} strata show a SIGNIFICANTLY worse realized "
              f"result in the high-h quartile (CI entirely below 0); {pos} show the opposite. "
              f"h is predictive iff the first number dominates.")

    # ---- CHECK 2: the null floor --------------------------------------------------------------
    print("\nCHECK 2 -- NULL SCALE")
    print(f"  this pair  |h|: median {np.median(np.abs(h)):.4f}  p95 {np.percentile(np.abs(h),95):.4f}")
    if args.null_data:
        zn = np.load(args.null_data)
        mn = (zn["ply"] >= args.min_ply) & (zn["source"] == 0)
        hn = (zn["q_sf"] - zn["q_human"])[mn]
        print(f"  NULL pair  |h|: median {np.median(np.abs(hn)):.4f}  "
              f"p95 {np.percentile(np.abs(hn),95):.4f}   (n={len(hn):,})")
        r = np.median(np.abs(h)) / max(np.median(np.abs(hn)), 1e-9)
        print(f"  ratio {r:.2f}x -- the dynamics signal is real only to the extent this exceeds 1.")
    else:
        print("  (no --null-data given: the floor is UNMEASURED and the magnitude of h above "
              "carries no scale)")

    print(f"\n  {'ply band':>10s} {'n':>8s} {'median h':>9s} {'mean|h|':>8s}")
    for lo, hi in [(8, 20), (20, 40), (40, 60), (60, 90), (90, 10_000)]:
        b = (ply >= lo) & (ply < hi)
        if b.sum() < 200:
            continue
        print(f"  {lo:>4d}-{hi if hi<10000 else 'inf':>4} {int(b.sum()):>8,} "
              f"{np.median(h[b]):>+9.3f} {np.abs(h[b]).mean():>8.3f}")
    print(f"\ndone [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
