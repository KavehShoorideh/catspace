#!/usr/bin/env python
"""
experiments/capacity_forensics.py — is the play-degrades-with-training effect
capacity collapse or honest specialization? (Kaveh 2026-07-16)

Across a run's ladder snapshots, measure head-free:
  1. EFFECTIVE RANK of F on a fixed mixed board sample (entropy of the
     singular-value spectrum) -- does the live subspace shrink?
  2. SUBSPACE ROTATION: principal angles between each snapshot's top-k F
     subspace and the final snapshot's -- does the geometry churn?
  3. REGIME-SPLIT FEATURE DRIFT: mean |F_t(s) - F_final(s)| separately for
     RARE-regime states (toy rook endgames) vs COMMON-regime states (human
     middlegames) -- is the rare regime overwritten while the common one
     stabilizes? (The frequency-weighted-reallocation signature.)
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


def eff_rank(X):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = (s ** 2) / (s ** 2).sum()
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-stem", default="data/derived/sep/committor_base_full")
    ap.add_argument("--steps", type=int, nargs="+",
                    default=[30000, 60000, 90000, 120000, 150000])
    ap.add_argument("--rare-table", default="artifacts/experiments/certainty_table_r3_K16.json",
                    help="rare-regime states (toy rook endgames)")
    ap.add_argument("--common-shards", default="data/shards/lichess_db_standard_rated_2019-01.prefix4gb")
    ap.add_argument("--n-per-regime", type=int, default=400)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.data.shards import LichessPairSource

    dev = pick_device("auto")
    rng = np.random.default_rng(args.seed)

    rows = json.loads(Path(args.rare_table).read_text())["rows"]
    rare = [rows[i]["fen"] for i in rng.choice(len(rows), args.n_per_regime, replace=False)]
    src = LichessPairSource(Path(args.common_shards), gamma=0.95)
    batch = next(iter(src.batches(args.n_per_regime, seed=args.seed)))
    common_packed = batch.anchors
    common_meta = batch.meta["board_meta"]

    def embed(fb, fens=None, packed=None, meta=None):
        if fens is not None:
            boards = [chess.Board(f) for f in fens]
            packed = np.stack([encode_packed(b) for b in boards])
            meta = np.stack([encode_meta(b) for b in boards])
        om = omega_ids(np.full(len(packed), 1800), np.full(len(packed), 1800),
                       np.full(len(packed), np.nan))
        with torch.no_grad():
            return fb.embed_F(torch.from_numpy(feature_planes(packed, meta)).to(dev),
                              torch.from_numpy(om).to(dev)).cpu().numpy()

    paths = [Path(f"{args.ckpt_stem}_step{s}.pt") for s in args.steps]
    paths.append(Path(f"{args.ckpt_stem}.pt"))
    labels = [f"step{s}" for s in args.steps] + ["final"]
    F_rare, F_common = {}, {}
    for p, lab in zip(paths, labels):
        fb, _ = load_ckpt(p, dev)
        fb.eval()
        F_rare[lab] = embed(fb, fens=rare)
        F_common[lab] = embed(fb, packed=common_packed, meta=common_meta)
        X = np.vstack([F_rare[lab], F_common[lab]])
        print(f"{lab:>10}: effective rank {eff_rank(X):5.2f} of {X.shape[1]}")

    def top_subspace(X, k):
        _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
        return Vt[:k].T

    Vf = top_subspace(np.vstack([F_rare["final"], F_common["final"]]), args.topk)
    print(f"\nsubspace rotation vs final (top-{args.topk} principal angles, deg):")
    for lab in labels[:-1]:
        V = top_subspace(np.vstack([F_rare[lab], F_common[lab]]), args.topk)
        s = np.linalg.svd(Vf.T @ V, compute_uv=False)
        ang = np.degrees(np.arccos(np.clip(s, 0, 1)))
        print(f"  {lab:>10}: mean {ang.mean():5.1f}  max {ang.max():5.1f}")

    print("\nregime-split feature drift |F_t - F_final| (mean L2 per state):")
    for lab in labels[:-1]:
        dr = float(np.linalg.norm(F_rare[lab] - F_rare["final"], axis=1).mean())
        dc = float(np.linalg.norm(F_common[lab] - F_common["final"], axis=1).mean())
        print(f"  {lab:>10}: rare {dr:.3f}  common {dc:.3f}  ratio rare/common {dr/max(dc,1e-9):.2f}")
    print("\nVERDICT CAPACITY_FORENSICS: see rank trajectory (collapse?), rotation "
          "(churn?), and rare/common drift ratio (>1 = rare regime overwritten more)")


if __name__ == "__main__":
    main()
