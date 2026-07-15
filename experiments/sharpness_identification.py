#!/usr/bin/env python
"""
experiments/sharpness_identification.py — identify (existence, sharpness) per
state from multi-epsilon rollout tables (Kaveh 2026-07-15).

Model under test:  -ln P(eps) ~= a + eps * S   per state, where
  a  = existence penalty (eps->0 intercept; should be ~0 on tb-won toy states
       with a tb-optimal White -- syzygy is the ground truth we validate the
       extrapolation against, because at full board there is no syzygy)
  S  = sharpness integral of the remaining path (slope in eps: how fast
       conversion probability decays per unit of per-move blunder rate)

Inputs: tb-White rollout tables at eps in {0.05, 0.10, 0.20}. Per state
(present in all tables with >= --min-n visits each), weighted least squares
of -ln P-hat on eps. VERDICTs: intercept calibration (existence), linearity
(is S a per-state constant?), S structure (vs |dtz| = exposure, i.e. path
length remaining).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.certainty_distill import spearman_ci
from experiments.value_fixed_point import TB


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", nargs="+", default=[
        "artifacts/experiments/certainty_table_eps05.json",
        "artifacts/experiments/certainty_table_demo_tb.json",
        "artifacts/experiments/certainty_table_eps20.json"])
    ap.add_argument("--eps", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--min-n", type=int, default=6)
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--out", default="artifacts/experiments/sharpness_table.json")
    args = ap.parse_args()
    assert len(args.tables) == len(args.eps)

    per_eps = []
    for t in args.tables:
        rows = json.loads(Path(t).read_text())["rows"]
        per_eps.append({r["fen"]: r for r in rows if r["n"] >= args.min_n})
    common = set(per_eps[0])
    for d in per_eps[1:]:
        common &= set(d)
    print(f"{len(common)} states present at all {len(args.eps)} eps levels "
          f"(n >= {args.min_n} each)")

    eps = np.array(args.eps)
    X = np.stack([np.ones_like(eps), eps], axis=1)
    out_rows, A, S, resid = [], [], [], []
    for fen in common:
        y, w = [], []
        for d, e in zip(per_eps, eps):
            r = d[fen]
            p = max(r["p_hat"], 1.0 / (r["n"] + 2))
            y.append(-np.log(p))
            w.append(r["n"])
        y, w = np.array(y), np.array(w)
        Xw = X * np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(Xw, y * np.sqrt(w), rcond=None)
        a, s = float(coef[0]), float(coef[1])
        pred = X @ coef
        A.append(a); S.append(s)
        resid.append(float(np.sqrt(np.average((y - pred) ** 2, weights=w))))
        out_rows.append(dict(fen=fen, existence=a, S=s,
                             n=[int(d[fen]["n"]) for d in per_eps]))
    A, S, resid = np.array(A), np.array(S), np.array(resid)

    # ground truth: |dtz| per state (exposure proxy) -- eval-only instrument
    tb = TB(args.syzygy_dir)
    dtz = []
    for r in out_rows:
        _, d = tb.wdl_dtz(chess.Board(r["fen"]))
        dtz.append(abs(d) if d is not None else np.nan)
    tb.close()
    dtz = np.array(dtz)
    ok = np.isfinite(dtz)

    print(f"VERDICT EXISTENCE intercept: median {np.median(A):+.3f}  "
          f"frac |a|<0.15: {(np.abs(A) < 0.15).mean():.2f}  "
          f"(truth: ~0 everywhere, all states tb-won)")
    print(f"VERDICT SHARPNESS S: median {np.median(S):.2f}  "
          f"IQR [{np.percentile(S,25):.2f},{np.percentile(S,75):.2f}]  "
          f"frac S<0: {(S<0).mean():.2f}")
    r, lo, hi = spearman_ci(S[ok], dtz[ok])
    print(f"VERDICT S-vs-|dtz| Spearman {r:+.3f} CI[{lo:+.3f},{hi:+.3f}] "
          f"(exposure: longer remaining path -> larger S expected)")
    # exposure-normalized sharpness: S per unit dtz -- the 'density' Kaveh means
    dens = S[ok] / np.maximum(dtz[ok], 1)
    print(f"VERDICT S-DENSITY (S/|dtz|): median {np.median(dens):.2f}  "
          f"IQR [{np.percentile(dens,25):.2f},{np.percentile(dens,75):.2f}]")
    print(f"VERDICT LINEARITY weighted-RMS residual: median {np.median(resid):.3f}  "
          f"p90 {np.percentile(resid,90):.3f}  (vs -lnP spans ~{np.median(S)*0.15:.2f} "
          f"over the eps range)")
    Path(args.out).write_text(json.dumps(dict(eps=args.eps, rows=out_rows)))
    print(f"-> {args.out}  ({len(out_rows)} states)")


if __name__ == "__main__":
    main()
