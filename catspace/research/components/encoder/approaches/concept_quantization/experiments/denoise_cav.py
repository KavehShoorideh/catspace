#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/concept_quantization/experiments/denoise_cav.py -- SHARPEN a concept direction by SAE-denoising, then use it as a subgoal
REGION. Imports the recipe of "Denoising Concept Vectors with Sparse Autoencoders for Improved Steering"
(arXiv 2505.15038, 2025) onto our IQE-QRL value field, per the 2026-07-21 literature synthesis.

Motivation (JOURNAL 2026-07-21): the incumbent field already carries structural concepts as linear
directions at mean AUC ~0.8 (connected_rooks 0.82); Kaposi's call is to USE those CAV directions as
planner subgoals rather than "fix" the field. Before wiring a concept as a subgoal region we sharpen its
direction: a raw diff-of-means / logistic CAV has an off-manifold noise component the SAE dictionary does
not span; reconstructing the CAV through the trained SAE (encode -> top-k -> decode) keeps only the
concept-aligned atoms and drops that noise.

For each named binary concept this compares, on a held-out split:
  * raw CAV          -- logistic-probe direction on standardized F(s).
  * SAE-denoised CAV -- ae.decode(ae.encode(cav)) using the field's own TopK SAE dictionary.
  * dist-to-region   -- the quasimetric reach score to the concept's goal REGION in B space
                        (d_c(s) = min over region-anchors of d(F(s), B_anchor)); this is the subgoal
                        reach cost the planner would use (QRL / Offline-GCRL waypoint selection).
reporting ROC-AUC (threshold-free) and precision@20, so we can see whether denoising crisps the direction
and whether the quasimetric reach score to the region tracks the concept.
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
from sklearn.metrics import roc_auc_score


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from dictionary_learning.trainers import TopKTrainer
from catspace.research.components.encoder.approaches.concept_quantization.experiments.concept_features import features as named_features
from catspace.io import paths


def prec_at(scores, label, k=20):
    order = np.argsort(-scores)
    return float(label[order[:k]].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default=paths.sep("lichess_gn_iqeqrl_full.pt"))
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--dict", type=int, default=96)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--region-frac", type=float, default=0.5,
                    help="fraction of concept-positive train positions whose B-embeddings seed the goal region anchors")
    ap.add_argument("--region-anchors", type=int, default=32)
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
        F = fb.embed_F(t, torch.from_numpy(np.tile(om, (len(Pk), 1))).to(dev))
        B = fb.embed_B(t)
        Fnp, Bt = F.cpu().numpy(), B
    mu, sd = Fnp.mean(0), Fnp.std(0) + 1e-8
    Xn = (Fnp - mu) / sd
    X = torch.from_numpy(Xn).float().to(dev)

    # field's own TopK SAE (same recipe as check_precision.py)
    tr = TopKTrainer(steps=args.steps, activation_dim=X.shape[1], dict_size=args.dict, k=args.k, layer=0,
                     lm_name="c", device=dev, warmup_steps=max(1, args.steps // 10), seed=args.seed)
    for s in range(args.steps):
        tr.update(s, X[torch.from_numpy(rng.integers(0, len(X), 1024)).to(dev)])
    ae = tr.ae

    feats = [named_features(b) for b in boards]
    fnames = [n for n in feats[0] if not n.endswith("_ctrl") and feats[0][n][1] == "bin"]
    Fmat = np.array([[float(f[n][0]) for n in fnames] for f in feats])
    tr_i, te_i = train_test_split(np.arange(len(Pk)), test_size=0.4, random_state=args.seed)

    print(f"VERDICT DENOISE_CAV field={Path(args.field).stem} n={len(Pk)} dict={args.dict} k={args.k}")
    print(f"  {'concept':16s} {'base':>5s} | {'rawCAV auc':>10s} {'p@20':>5s} | {'denoise auc':>11s} {'p@20':>5s} | {'dist2reg auc':>12s} {'p@20':>5s}")
    for j, nm in enumerate(fnames):
        y = Fmat[:, j]
        if not (0.05 < y.mean() < 0.95):
            continue
        # raw CAV: logistic direction on standardized F
        clf = LogisticRegression(max_iter=300).fit(Xn[tr_i], y[tr_i])
        w = clf.coef_[0].astype(np.float32)
        raw = Xn @ w
        # SAE-denoised CAV: reconstruct the direction through the dictionary (encode -> topk -> decode)
        with torch.no_grad():
            wv = torch.from_numpy(w / (np.linalg.norm(w) + 1e-9)).float().to(dev)[None, :]
            w_dn = ae.decode(ae.encode(wv)).cpu().numpy()[0]
        den = Xn @ (w_dn / (np.linalg.norm(w_dn) + 1e-9))
        # quasimetric reach to the concept's goal REGION: seed anchors from B of concept-positive train rows
        pos = tr_i[y[tr_i] > 0.5]
        rng.shuffle(pos); seed = pos[:max(args.region_anchors, int(len(pos) * args.region_frac))]
        with torch.no_grad():
            Bseed = Bt[torch.from_numpy(seed).to(dev)]
            if len(seed) > args.region_anchors:                       # k-means-lite: random anchor subset
                Bseed = Bseed[torch.from_numpy(rng.permutation(len(seed))[:args.region_anchors]).to(dev)]
            dreg = fb.distance_matrix(F, Bseed).min(dim=1).values.cpu().numpy()   # d_c(s) = min reach cost
        reach = -dreg                                                 # closer (smaller d) => more "in region"
        a_raw = roc_auc_score(y[te_i], raw[te_i]); a_den = roc_auc_score(y[te_i], den[te_i])
        a_reg = roc_auc_score(y[te_i], reach[te_i])
        print(f"  {nm.replace('_w',''):16s} {y.mean():>4.0%} | "
              f"{a_raw:>10.3f} {prec_at(raw[te_i], y[te_i]):>4.0%} | "
              f"{a_den:>11.3f} {prec_at(den[te_i], y[te_i]):>4.0%} | "
              f"{a_reg:>12.3f} {prec_at(reach[te_i], y[te_i]):>4.0%}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
