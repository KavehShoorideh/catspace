#!/usr/bin/env python
"""experiments/concept_features.py -- which INTERPRETABLE features did the value field keep? (Kaposi
2026-07-21). The full-material probe was the wrong test: the value field SHOULD discard value-irrelevant
structure (a pawn asleep in the corner) and keep only what moves distance-to-outcome. So probe F/B for
specific NAMED features -- value-relevant ones (connected rooks, king safety, passed pawn, material,
bishop pair, phase) AND deliberately value-IRRELEVANT controls (a corner pawn, a specific back-rank
square).

Reframe prediction: value-relevant features decode well above chance; the irrelevant controls sit at
chance. Whatever decodes from F is a feature the field kept -- a candidate CONCEPT. (Predictive-power-
for-arrival, the other half of a concept, is the natural next pass.)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device

PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def _connected_rooks(board, color):
    rooks = list(board.pieces(chess.ROOK, color))
    for i in range(len(rooks)):
        for j in range(i + 1, len(rooks)):
            a, b = rooks[i], rooks[j]
            if (chess.square_rank(a) == chess.square_rank(b) or chess.square_file(a) == chess.square_file(b)):
                if not (chess.SquareSet.between(a, b) & board.occupied):
                    return True
    return False


def _passed_pawn(board, color):
    ours = board.pieces(chess.PAWN, color); theirs = board.pieces(chess.PAWN, not color)
    fwd = 1 if color == chess.WHITE else -1
    for sq in ours:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        blocked = any(chess.square_file(t) in (f - 1, f, f + 1) and
                      (chess.square_rank(t) - r) * fwd > 0 for t in theirs)
        if not blocked:
            return True
    return False


def _king_safe(board, color):
    k = board.king(color)
    return k is not None and chess.square_file(k) in (1, 2, 6, 7) and \
        chess.square_rank(k) == (0 if color == chess.WHITE else 7)


def features(board):
    """{name: (value, kind)} -- kind 'bin' or 'num'. Named, interpretable, value-relevant + controls."""
    W, B = chess.WHITE, chess.BLACK
    matw = sum(PIECE_VAL.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == W)
    matb = sum(PIECE_VAL.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == B)
    occ = board.occupied
    return {
        # --- value-relevant candidate concepts ---
        "connected_rooks_w": (_connected_rooks(board, W), "bin"),
        "passed_pawn_w":     (_passed_pawn(board, W), "bin"),
        "king_safe_w":       (_king_safe(board, W), "bin"),
        "bishop_pair_w":     (len(board.pieces(chess.BISHOP, W)) >= 2 and len(board.pieces(chess.BISHOP, B)) < 2, "bin"),  # ADVANTAGE: I have the pair, opponent doesn't
        "queens_on":         (bool(board.pieces(chess.QUEEN, W)) or bool(board.pieces(chess.QUEEN, B)), "bin"),
        "material_diff":     (matw - matb, "num"),
        "piece_count":       (len(board.piece_map()), "num"),
        # --- deliberately value-IRRELEVANT controls (should sit at chance) ---
        "pawn_on_a3_ctrl":   (bool(board.piece_at(chess.A3) and board.piece_at(chess.A3).piece_type == chess.PAWN), "bin"),
        "occupied_h1_ctrl":  (bool(occ & chess.BB_H1), "bin"),
        "pawn_on_afile_ctrl":(any(chess.square_file(s) == 0 for s in board.pieces(chess.PAWN, W)), "bin"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="data/derived/sep/lichess_gn_iqeqrl_full.pt")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=9000)
    ap.add_argument("--min-ply", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)

    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    nz = np.load(args.shard)
    P, M, ply = np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["ply"]).astype(int)
    idx = np.flatnonzero(ply >= args.min_ply); idx = idx[rng.permutation(len(idx))[:args.n]]
    Pk, Mk = P[idx], M[idx]

    feats = [features(board_from_packed(Pk[i], Mk[i])) for i in range(len(Pk))]
    names = list(feats[0].keys())
    with torch.no_grad():
        t = torch.from_numpy(feature_planes(Pk, Mk)).to(dev)
        F = fb.embed_F(t, torch.from_numpy(np.tile(om, (len(Pk), 1))).to(dev)).cpu().numpy()
        Bemb = fb.embed_B(t).cpu().numpy()
    print(f"[stage] {len(Pk)} positions (ply>={args.min_ply}), {len(names)} named features ({time.time()-t0:.0f}s)",
          flush=True)

    def decode(X, y, kind):
        y = np.array(y)
        if kind == "bin":
            if y.mean() < 0.02 or y.mean() > 0.98:
                return None, y.mean()                              # too rare to score
            s = cross_val_score(LogisticRegression(max_iter=200), X, y.astype(int), cv=3, scoring="roc_auc")
            return float(s.mean()), y.mean()
        s = cross_val_score(Ridge(alpha=1.0), X, y.astype(float), cv=3, scoring="r2")
        return float(s.mean()), float(y.std())

    pc = np.array([f["piece_count"][0] for f in feats], float).reshape(-1, 1)   # phase-only predictor
    print(f"VERDICT CONCEPT_FEATURES field={Path(args.field).stem} n={len(Pk)}")
    print(f"  {'feature':20s} {'kind':4s} {'F':>7s} {'B':>7s} {'phase':>7s} {'F-lift':>7s}   "
          f"(bin ROC-AUC/0.5, num R2; lift = F over phase-only)")
    for nm in names:
        vals = [f[nm][0] for f in feats]; kind = feats[0][nm][1]
        fF, base = decode(F, vals, kind); fB, _ = decode(Bemb, vals, kind)
        fPh, _ = decode(pc, vals, kind) if nm != "piece_count" else (None, 0)
        lift = (fF - fPh) if (fF is not None and fPh is not None) else None
        tag = "CONTROL" if nm.endswith("_ctrl") else ("CONCEPT?" if (lift is not None and lift > 0.05) else "")
        def s(x): return f"{x:.3f}" if x is not None else "   -- "
        print(f"  {nm:20s} {kind:4s} {s(fF):>7s} {s(fB):>7s} {s(fPh):>7s} {s(lift):>7s}   {tag}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
