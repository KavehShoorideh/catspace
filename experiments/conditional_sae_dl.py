#!/usr/bin/env python
"""experiments/conditional_sae_dl.py -- the maintained SAE, FORKED to add conditioning (Kaposi
2026-07-21: "fork the maintained package and add the conditioning on top"; phase strata is the wrong
axis -- bishop-pair is open-vs-closed, so we condition on the full covariate vector, not phase bins).

Subclasses dictionary_learning's AutoEncoderTopK (keeps its TopK core, unit-norm decoder init) and
adds a FiLM gate on the pre-topk activations, driven by c = phase, openness, advantage, distance-to-
B-cluster (phase orthogonalized out of the rest). So each atom fires only in its DOMAIN, and the
context that matters for a concept (openness for bishop-pair) is available -- while the SAE itself
stays the imported implementation.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device
from contrib.dl_conditional import ConditionalTopKTrainer     # our contributable conditional SAE
from experiments.concept_features import features as named_features
from experiments.conditional_concepts import openness
from experiments.sae_concepts import heatmap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="data/derived/sep/lichess_gn_iqeqrl_full.pt")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--dict", type=int, default=96)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--anchors", type=int, default=6)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--top", type=int, default=200)
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
    ev = np.asarray(nz["eval_cp"]).astype(np.float32)
    ok = np.flatnonzero((ply >= args.min_ply) & np.isfinite(ev) & (np.abs(ev) < 2000))
    idx = ok[rng.permutation(len(ok))[:args.n]]
    Pk, Mk = P[idx], M[idx]
    boards = [board_from_packed(Pk[i], Mk[i]) for i in range(len(Pk))]
    with torch.no_grad():
        t = torch.from_numpy(feature_planes(Pk, Mk)).to(dev)
        F = fb.embed_F(t, torch.from_numpy(np.tile(om, (len(Pk), 1))).to(dev))
        B = fb.embed_B(t)
        km = KMeans(n_clusters=args.anchors, n_init=5, random_state=args.seed).fit(B.cpu().numpy())
        d_anchor = fb.distance_matrix(F, torch.from_numpy(km.cluster_centers_).float().to(dev)).cpu().numpy()
        F = F.cpu().numpy()

    pc = np.array([len(b.piece_map()) for b in boards], float)
    opn = np.array([openness(b) for b in boards], float)
    adv = np.clip(ev[idx], -1000, 1000)
    cov_names = ["phase", "openness", "advantage"] + [f"d_anchor{k}" for k in range(args.anchors)]
    C = np.column_stack([pc, opn, adv, d_anchor])
    zc = C[:, 0] - C[:, 0].mean()                                 # orthogonalize: partial phase out of the rest
    for k in range(1, C.shape[1]):
        ck = C[:, k] - C[:, k].mean(); C[:, k] = ck - (ck @ zc) / (zc @ zc + 1e-9) * zc
    C = (C - C.mean(0)) / (C.std(0) + 1e-8)
    X = torch.from_numpy((F - F.mean(0)) / (F.std(0) + 1e-8)).float().to(dev)
    Ct = torch.from_numpy(C).float().to(dev)

    tr = ConditionalTopKTrainer(steps=args.steps, activation_dim=X.shape[1], dict_size=args.dict,
                                k=args.k, cond_dim=C.shape[1], layer=0, lm_name=Path(args.field).stem,
                                device=dev, warmup_steps=max(1, args.steps // 10), seed=args.seed)
    XC = torch.cat([X, Ct], dim=1)                                # [x | cond] convention
    for step in range(args.steps):
        tr.update(step, XC[torch.from_numpy(rng.integers(0, len(XC), size=1024)).to(dev)])
    ae = tr.ae
    with torch.no_grad():
        code = ae.encode(X, cond=Ct).cpu().numpy()
        gate = torch.sigmoid(ae.gate(Ct)).cpu().numpy()
        ve = float(1 - (ae.decode(ae.encode(X, cond=Ct)) - X).pow(2).mean() / X.var())

    alive = np.flatnonzero((code > 1e-6).mean(0) > 0.003)
    feats = [named_features(b) for b in boards]
    fnames = [n for n in feats[0] if not n.endswith("_ctrl") and feats[0][n][1] == "bin"]  # binary -> prevalence is a %
    Fmat = np.array([[float(f[n][0]) for n in fnames] for f in feats])
    print(f"VERDICT CONDITIONAL_SAE_DL field={Path(args.field).stem} lib=dictionary_learning(forked) "
          f"dict={args.dict} k={args.k} var_expl={ve:.2f} alive={len(alive)}")
    print("  conditional concept per covariate (atom whose DOMAIN-gate most tracks it) + monosemantic prevalence:")
    base = Fmat.mean(0)
    dom = np.array([[np.corrcoef(gate[:, a], C[:, kk])[0, 1] for kk in range(C.shape[1])] for a in alive])
    for kk, cn in enumerate(cov_names):
        ai = int(np.abs(dom[:, kk]).argmax()); a = alive[ai]
        core = np.argsort(-code[:, a])[:args.top]
        contr = [Fmat[core, j].mean() / (base[j] + 1e-9) for j in range(len(fnames))]  # PREVALENCE contrast
        jm = int(np.argmax(contr))
        prev, bl = Fmat[core, jm].mean(), base[jm]
        named = fnames[jm].replace("_w", "") if contr[jm] > 1.5 else "weak"
        print(f"    {cn:11s}{'+' if dom[ai,kk]>0 else '-'} r={dom[ai,kk]:+.2f} -> atom {a:3d} [{named:14s}] "
              f"cluster {prev:.0%} vs base {bl:.0%} ({contr[jm]:.1f}x) | {heatmap(Pk, Mk, np.argsort(-code[:, a]))}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
