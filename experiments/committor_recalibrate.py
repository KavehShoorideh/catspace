#!/usr/bin/env python
"""
experiments/committor_recalibrate.py — monotone calibration of a committor head.

The MSE-distilled head RANKS conversion probability well (rho +0.6) but its
absolute scale is compressed (learned P_W spanned [0.19,0.37] vs empirical
[0,1]); end-to-end NLL training collapsed rank to the base rate (measured,
2026-07-15). Resolution: keep the MSE geometry, fit a 2-parameter MONOTONE
affine in d-space by smoothed binomial NLL:
    d' = a*d + b   (a > 0)   i.e.   P' = e^{-b} * P^a  (Platt in log space)
Rank is preserved EXACTLY (monotone), so play through the self-calibrating
MCTS squash is unchanged; the calibrated probabilities are for consumers that
need absolute P (goal-selection layer, surface atlas, viz). The affine is
stored IN the *_whead.pt payload ("affine": [a, b]); loaders that only rank
may ignore it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catspace.data.encode import encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids


def smoothed_nll(d, k, n):
    d = np.maximum(d, 1e-4)
    per = (k + 1) * d - (n - k + 1) * np.log1p(-np.exp(-d))
    return float(per.sum() / (n + 2).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/sep/committor.pt")
    ap.add_argument("--whead", default="data/derived/sep/committor_whead.pt")
    ap.add_argument("--table", default="artifacts/experiments/certainty_table_r2_K16.json")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from scipy.optimize import minimize
    from catspace.nn.fb import load_ckpt

    fb, _ = load_ckpt(Path(args.ckpt), "cpu")
    hp = torch.load(args.whead, map_location="cpu", weights_only=False)
    head = torch.nn.Sequential(torch.nn.Linear(hp["d_in"], 128), torch.nn.ReLU(),
                               torch.nn.Linear(128, 1), torch.nn.Softplus())
    head.load_state_dict(hp["state"]); head.eval(); fb.eval()

    rows = json.loads(Path(args.table).read_text())["rows"]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))
    n_hold = int(len(rows) * args.holdout_frac)
    hold, train = [rows[i] for i in order[:n_hold]], [rows[i] for i in order[n_hold:]]

    def d_of(rs):
        out = []
        with torch.no_grad():
            for i in range(0, len(rs), 512):
                ch = rs[i:i + 512]
                boards = [chess.Board(r["fen"]) for r in ch]
                packed = np.stack([encode_packed(b) for b in boards])
                meta = np.stack([encode_meta(b) for b in boards])
                om = omega_ids(np.full(len(ch), 1800), np.full(len(ch), 1800),
                               np.full(len(ch), np.nan))
                f = fb.embed_F(torch.from_numpy(feature_planes(packed, meta)),
                               torch.from_numpy(om))
                out.append(head(f).squeeze(-1).numpy())
        return np.concatenate(out)

    d_tr, d_ho = d_of(train), d_of(hold)
    k_tr = np.array([r["p_hat"] * r["n"] for r in train]); n_tr = np.array([r["n"] for r in train])
    k_ho = np.array([r["p_hat"] * r["n"] for r in hold]); n_ho = np.array([r["n"] for r in hold])

    def obj(x):
        a, b = np.exp(x[0]), np.exp(x[1]) - 1.0   # a>0; b>-1 (b<0 allowed mildly)
        return smoothed_nll(a * d_tr + max(b, -d_tr.min() + 1e-3), k_tr, n_tr)

    res = minimize(obj, x0=[0.0, 0.0], method="Nelder-Mead")
    a, b = float(np.exp(res.x[0])), float(np.exp(res.x[1]) - 1.0)

    def report(d, tag):
        P = np.exp(-np.maximum(d, 1e-4))
        pe = k_ho / n_ho
        o = np.argsort(P)
        ece = float(np.mean([abs(P[bb].mean() - pe[bb].mean())
                             for bb in np.array_split(o, 10) if len(bb)]))
        print(f"{tag}: span [{P.min():.2f},{P.max():.2f}]  ECE {ece:.3f}  "
              f"NLL {smoothed_nll(np.maximum(d,1e-4), k_ho, n_ho):.4f}")
        return ece

    print(f"fitted affine: d' = {a:.3f}*d + {b:+.3f}")
    e0 = report(d_ho, "BEFORE (held-out)")
    e1 = report(a * d_ho + b, "AFTER  (held-out)")
    print(f"VERDICT RECALIBRATION a={a:.3f} b={b:+.3f}  ECE {e0:.3f} -> {e1:.3f}  "
          f"(rank preserved exactly: monotone)")
    hp["affine"] = [a, b]
    import torch as _t
    _t.save(hp, args.whead)
    print(f"affine stored in {args.whead}")


if __name__ == "__main__":
    main()
