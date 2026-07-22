#!/usr/bin/env python
"""experiments/conditional_concepts.py -- concepts are CONDITIONAL (Kaposi 2026-07-21): the same
structural feature matters in one context and not another (bishop pair: worthless in a closed opening,
decisive in an open endgame). So concept discovery must condition on structural COVARIATES -- phase
(= distance from start / to end), branch (open vs closed), distance-to-mate -- not pool globally.

This PROVES the premise before we build the conditional discovery: for each named feature (mirror), it
measures the correlation with ground-truth advantage (Stockfish eval_cp) WITHIN strata of phase and
openness. If a feature's advantage-correlation swings across strata, its relevance is conditional, and a
global SAE necessarily washes it out -> we must condition.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed
from experiments.concept_features import features as named_features


def openness(board):
    """open files (no pawns of either color) -- a simple open/closed proxy."""
    pawns = board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
    files_with_pawn = {chess.square_file(s) for s in pawns}
    return 8 - len(files_with_pawn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)

    nz = np.load(args.shard)
    P, M, ply = np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["ply"]).astype(int)
    if "eval_cp" not in nz.files:
        raise SystemExit("shard has no eval_cp (Stockfish advantage) -- needed as the ground-truth target")
    ev = np.asarray(nz["eval_cp"]).astype(np.float32)
    ok = np.flatnonzero((ply >= args.min_ply) & np.isfinite(ev) & (np.abs(ev) < 2000))
    idx = ok[rng.permutation(len(ok))[:args.n]]
    Pk, Mk, evk = P[idx], M[idx], np.clip(ev[idx], -1000, 1000)

    feats = [named_features(board_from_packed(Pk[i], Mk[i])) for i in range(len(Pk))]
    pcnt = np.array([f["piece_count"][0] for f in feats])
    opn = np.array([openness(board_from_packed(Pk[i], Mk[i])) for i in range(len(Pk))])
    fnames = [n for n in feats[0] if not n.endswith("_ctrl") and n != "piece_count"]

    phase_bins = [("opening (>=26p)", pcnt >= 26), ("middle (16-25p)", (pcnt >= 16) & (pcnt <= 25)),
                  ("endgame (<=15p)", pcnt <= 15)]
    open_bins = [("closed (<=1 open file)", opn <= 1), ("open (>=3 open files)", opn >= 3)]

    def corr(feat_vals, mask):
        x = np.asarray(feat_vals, float)[mask]; y = evk[mask]
        if mask.sum() < 200 or x.std() < 1e-6:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    print(f"[stage] {len(Pk)} positions with eval_cp ({time.time()-t0:.0f}s)")
    print(f"VERDICT CONDITIONAL_CONCEPTS n={len(Pk)}  (corr of feature with white advantage eval_cp)")
    print(f"  {'feature':18s} " + " ".join(f"{nm.split()[0]:>10s}" for nm, _ in phase_bins) +
          "   | " + " ".join(f"{nm.split()[0]:>8s}" for nm, _ in open_bins))
    for nm in fnames:
        vals = [f[nm][0] for f in feats]
        by_phase = [corr(vals, mask) for _, mask in phase_bins]
        by_open = [corr(vals, mask) for _, mask in open_bins]
        def s(x): return f"{x:+.2f}" if x is not None else "   -- "
        swing = max([v for v in by_phase if v is not None] + [0]) - min([v for v in by_phase if v is not None] + [0])
        tag = "<- CONDITIONAL" if swing > 0.10 else ""
        print(f"  {nm:18s} " + " ".join(f"{s(v):>10s}" for v in by_phase) +
              "   | " + " ".join(f"{s(v):>8s}" for v in by_open) + f"   {tag}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
