#!/usr/bin/env python
"""experiments/check_precision.py -- is the concept-direction PRECISE? (Kaposi 2026-07-21: "the issue is
precision, non-connected rooks are seen as connected".) For each named concept, compare the PRECISION@K
(fraction of the top-K highest-scoring positions that actually have the feature) of:
  * the SUPERVISED CAV -- a linear probe trained on the ground-truth label (the best achievable linear
    direction; held-out scored), and
  * the unsupervised SAE atom that best matches it.
If even the CAV is imprecise, the value field doesn't separate the concept cleanly (a field limit, not a
method one). Also dumps the top FALSE POSITIVES (highest CAV score but feature absent) so we can see what
the field is confusing it with.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from dictionary_learning.trainers import TopKTrainer
from experiments.concept_features import features as named_features


def prec_at(scores, label, ks=(5, 20, 100)):
    order = np.argsort(-scores)
    return {k: float(label[order[:k]].mean()) for k in ks}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="data/derived/sep/lichess_gn_iqeqrl_full.pt")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=14000)
    ap.add_argument("--dict", type=int, default=96)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    nz = np.load(args.shard)
    P, M, ply = np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["ply"]).astype(int)
    idx = np.flatnonzero(ply >= args.min_ply); idx = idx[rng.permutation(len(idx))[:args.n]]
    Pk, Mk = P[idx], M[idx]
    boards = [board_from_packed(Pk[i], Mk[i]) for i in range(len(Pk))]
    with torch.no_grad():
        t = torch.from_numpy(feature_planes(Pk, Mk)).to(dev)
        emb = fb.embed_F(t, torch.from_numpy(np.tile(om, (len(Pk), 1))).to(dev)).cpu().numpy()
    Xn = (emb - emb.mean(0)) / (emb.std(0) + 1e-8)
    X = torch.from_numpy(Xn).float().to(dev)
    tr = TopKTrainer(steps=args.steps, activation_dim=X.shape[1], dict_size=args.dict, k=args.k, layer=0,
                     lm_name="c", device=dev, warmup_steps=max(1, args.steps // 10), seed=args.seed)
    for s in range(args.steps):
        tr.update(s, X[torch.from_numpy(rng.integers(0, len(X), 1024)).to(dev)])
    with torch.no_grad():
        code = tr.ae.encode(X).cpu().numpy()

    feats = [named_features(b) for b in boards]
    fnames = [n for n in feats[0] if not n.endswith("_ctrl") and feats[0][n][1] == "bin"]
    Fmat = np.array([[float(f[n][0]) for n in fnames] for f in feats])
    tr_i, te_i = train_test_split(np.arange(len(Pk)), test_size=0.4, random_state=args.seed)
    print(f"VERDICT CHECK_PRECISION field={Path(args.field).stem} n={len(Pk)}  (precision@K on held-out)")
    print(f"  {'concept':16s} {'base':>5s} | {'CAV @5':>7s} {'@20':>5s} {'@100':>5s} | {'SAE @5':>7s} {'@20':>5s} {'@100':>5s}")
    fp_dump = None
    for j, nm in enumerate(fnames):
        y = Fmat[:, j]
        if not (0.03 < y.mean() < 0.97):
            continue
        clf = LogisticRegression(max_iter=300).fit(Xn[tr_i], y[tr_i])
        cav = clf.decision_function(Xn[te_i]); cavp = prec_at(cav, y[te_i])
        best_a = max(range(code.shape[1]),
                     key=lambda a: prec_at(code[te_i, a], y[te_i], (100,))[100] if (code[:, a] > 1e-6).mean() > 0.003 else 0)
        saep = prec_at(code[te_i, best_a], y[te_i])
        print(f"  {nm.replace('_w',''):16s} {y.mean():>4.0%} | {cavp[5]:>6.0%} {cavp[20]:>4.0%} {cavp[100]:>4.0%} | "
              f"{saep[5]:>6.0%} {saep[20]:>4.0%} {saep[100]:>4.0%}")
        if nm.startswith("connected"):
            order = te_i[np.argsort(-cav)]
            fps = [i for i in order if y[i] < 0.5][:4]              # top CAV score but NOT connected
            fp_dump = (nm, fps)
    if fp_dump:
        nm, fps = fp_dump
        print(f"  top CAV FALSE POSITIVES for {nm.replace('_w','')} (high score, feature ABSENT) -- what is the field confusing?")
        for i in fps:
            b = boards[i]; rk = list(b.pieces(0 + 4, True))         # white rooks (ROOK=4? use chess)
            import chess
            rooks = list(b.pieces(chess.ROOK, chess.WHITE))
            rsq = ",".join(chess.square_name(s) for s in rooks)
            print(f"      {b.fen()}   white rooks: {rsq or 'none'}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
