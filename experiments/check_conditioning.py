#!/usr/bin/env python
"""experiments/check_conditioning.py -- are the concepts real, and is the CONDITIONING doing anything?
(Kaposi 2026-07-21.) Two honest checks:

  1. Monosemanticity, incl. the top-5 ACTUALLY shown (not just top-200): for each named concept, the
     best atom's prevalence of that feature among its most-activating positions vs baseline.
  2. Conditioning shuffle-ablation: train (a) unconditional SAE, (b) conditional SAE with REAL
     covariates, (c) conditional SAE with SHUFFLED covariates. If (b) does not beat (c), the FiLM gate
     is ignoring the covariates -> conditioning is NOT working. Also reports gate variance across
     contexts (0 => the gate is constant => no conditioning).

All three use the maintained dictionary_learning TopK SAE (contrib ConditionalTopKTrainer for b/c).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from contrib.dl_conditional import ConditionalTopKTrainer
from dictionary_learning.trainers import TopKTrainer
from experiments.concept_features import features as named_features
from experiments.conditional_concepts import openness


def measure(code, Fmat, fnames, base, tops=(5, 200)):
    """for each concept: the atom that best concentrates it, and its prevalence at top-5 & top-200."""
    out = {}
    for j, nm in enumerate(fnames):
        best = None
        for a in range(code.shape[1]):
            if (code[:, a] > 1e-6).mean() < 0.003:
                continue
            order = np.argsort(-code[:, a])
            p200 = Fmat[order[:200], j].mean()
            if best is None or p200 > best[1]:
                best = (a, p200, Fmat[order[:tops[0]], j].mean())
        out[nm] = best                                            # (atom, prev@200, prev@5)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="data/derived/sep/lichess_gn_iqeqrl_full.pt")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--dict", type=int, default=96)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--anchors", type=int, default=6)
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
    C = np.column_stack([pc, opn, adv, d_anchor])
    zc = C[:, 0] - C[:, 0].mean()
    for kk in range(1, C.shape[1]):
        ck = C[:, kk] - C[:, kk].mean(); C[:, kk] = ck - (ck @ zc) / (zc @ zc + 1e-9) * zc
    C = (C - C.mean(0)) / (C.std(0) + 1e-8)
    Xn = (F - F.mean(0)) / (F.std(0) + 1e-8)
    X = torch.from_numpy(Xn).float().to(dev); Ct = torch.from_numpy(C).float().to(dev)
    Cs = torch.from_numpy(C[rng.permutation(len(C))]).float().to(dev)     # SHUFFLED covariates

    feats = [named_features(b) for b in boards]
    fnames = [n for n in feats[0] if not n.endswith("_ctrl") and feats[0][n][1] == "bin"]
    Fmat = np.array([[float(f[n][0]) for n in fnames] for f in feats]); base = Fmat.mean(0)

    def train_uncond():
        tr = TopKTrainer(steps=args.steps, activation_dim=X.shape[1], dict_size=args.dict, k=args.k, layer=0,
                         lm_name="c", device=dev, warmup_steps=max(1, args.steps // 10), seed=args.seed)
        for s in range(args.steps):
            tr.update(s, X[torch.from_numpy(rng.integers(0, len(X), 1024)).to(dev)])
        with torch.no_grad():
            return tr.ae.encode(X).cpu().numpy(), None

    def train_cond(cov):
        tr = ConditionalTopKTrainer(steps=args.steps, activation_dim=X.shape[1], dict_size=args.dict, k=args.k,
                                    cond_dim=cov.shape[1], layer=0, lm_name="c", device=dev,
                                    warmup_steps=max(1, args.steps // 10), seed=args.seed)
        XC = torch.cat([X, cov], 1)
        for s in range(args.steps):
            tr.update(s, XC[torch.from_numpy(rng.integers(0, len(XC), 1024)).to(dev)])
        with torch.no_grad():
            code = tr.ae.encode(X, cond=cov).cpu().numpy()
            gate = torch.sigmoid(tr.ae.gate(cov)).cpu().numpy()
        return code, gate

    print(f"[stage] training 3 SAEs ({time.time()-t0:.0f}s)...", flush=True)
    runs = {"unconditional": train_uncond(), "conditional (real c)": train_cond(Ct),
            "conditional (SHUFFLED c)": train_cond(Cs)}
    print(f"VERDICT CHECK_CONDITIONING field={Path(args.field).stem} n={len(Pk)} dict={args.dict} k={args.k}")
    for name, (code, gate) in runs.items():
        m = measure(code, Fmat, fnames, base)
        gtxt = ""
        if gate is not None:
            gtxt = f" | gate std across ctx {gate.std():.3f} (0=>ignores c)"
        print(f"  {name}:{gtxt}")
        for nm in fnames:
            a, p200, p5 = m[nm]
            print(f"      {nm.replace('_w',''):16s} best atom {a:3d}: top5 {p5:.0%} top200 {p200:.0%} (base {base[fnames.index(nm)]:.0%})")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
