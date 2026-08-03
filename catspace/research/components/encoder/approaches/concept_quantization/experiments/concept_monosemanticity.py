#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/concept_quantization/experiments/concept_monosemanticity.py -- the test Kaposi asked for (2026-07-21): if a concept is
clean, the feature should be PREVALENT in exactly ONE atom's cluster and wash out to baseline in all
others. Correlation can hide smearing; prevalence-per-cluster exposes it.

For each named feature, and each alive atom, prevalence = fraction of the atom's TOP-activating
positions that have the feature. A monosemantic concept: prevalence >> baseline in its atom, ~baseline
(residual ~1x) in every other atom. Uses the maintained dictionary_learning TopK SAE. Optionally
--by-phase (conditioned clusters): the endgame dictionary should localise endgame concepts, etc.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from dictionary_learning.trainers import TopKTrainer
from catspace.research.components.encoder.approaches.concept_quantization.experiments.concept_features import features as named_features
from catspace.io import paths


def train_sae(emb, args, dev, rng):
    mu, sd = emb.mean(0), emb.std(0) + 1e-8
    X = torch.from_numpy((emb - mu) / sd).float().to(dev)
    tr = TopKTrainer(steps=args.steps, activation_dim=X.shape[1], dict_size=args.dict, k=args.k, layer=0,
                     lm_name="catspace", device=dev, warmup_steps=max(1, args.steps // 10), seed=args.seed)
    for step in range(args.steps):
        tr.update(step, X[torch.from_numpy(rng.integers(0, len(X), size=1024)).to(dev)])
    with torch.no_grad():
        return tr.ae.encode(X).cpu().numpy()


def report(code, Fmat, fnames, top, label):
    alive = np.flatnonzero((code > 1e-6).mean(0) > 0.003)
    base = Fmat.mean(0)
    # prevalence[a, j] = fraction of atom a's top-`top` positions that have feature j
    prev = np.zeros((len(alive), Fmat.shape[1]))
    for ai, a in enumerate(alive):
        core = np.argsort(-code[:, a])[:top]
        prev[ai] = Fmat[core].mean(0)
    print(f"  === [{label}] alive={len(alive)}  (test: concept high in ONE atom, ~baseline elsewhere) ===")
    print(f"    {'feature':18s} {'base':>6s} {'top-atom':>9s} {'thatprev':>9s} {'others~':>8s} {'contrast':>9s} {'residual':>9s}")
    for j, nm in enumerate(fnames):
        if not (0.03 < base[j] < 0.97):
            continue
        a = int(prev[:, j].argmax()); top_prev = prev[a, j]
        others = np.delete(prev[:, j], a)
        other_mean = float(others.mean())
        contrast = top_prev / (base[j] + 1e-9)                    # how enriched the top atom is
        residual = other_mean / (base[j] + 1e-9)                  # ~1.0 = washed out (good)
        flag = "MONO" if (contrast >= 1.8 and residual <= 1.3) else ("smeared" if residual > 1.5 else "")
        print(f"    {nm.replace('_w',''):18s} {base[j]:>5.0%} {'atom '+str(alive[a]):>9s} {top_prev:>8.0%} "
              f"{other_mean:>7.0%} {contrast:>8.1f}x {residual:>8.2f}x  {flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default=paths.sep("lichess_gn_iqeqrl_full.pt"))
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--dict", type=int, default=96)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--top", type=int, default=200, help="top-activating positions per atom = its cluster core")
    ap.add_argument("--tower", choices=["F", "B"], default="F")
    ap.add_argument("--by-phase", action="store_true")
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
    cand = np.flatnonzero(ply >= args.min_ply)
    pool = cand[rng.permutation(len(cand))[:min(len(cand), 120000)]]
    pcnt = np.unpackbits(P[pool].reshape(len(pool), -1).view(np.uint8), axis=1).sum(1)
    take = args.n * (3 if args.by_phase else 1)
    perm = rng.permutation(len(pool))[:take]
    sel, pcs = pool[perm], pcnt[perm]
    Pk, Mk = P[sel], M[sel]
    with torch.no_grad():
        t = torch.from_numpy(feature_planes(Pk, Mk)).to(dev)
        emb = (fb.embed_F(t, torch.from_numpy(np.tile(om, (len(Pk), 1))).to(dev)) if args.tower == "F"
               else fb.embed_B(t)).cpu().numpy()
    feats = [named_features(board_from_packed(Pk[i], Mk[i])) for i in range(len(Pk))]
    fnames = [n for n in feats[0] if not n.endswith("_ctrl")]
    Fmat = np.array([[float(f[n][0]) for n in fnames] for f in feats])
    print(f"VERDICT CONCEPT_MONOSEMANTICITY field={Path(args.field).stem} tower={args.tower} by_phase={args.by_phase}")
    if args.by_phase:
        for lab, msk in [("opening", pcs >= 26), ("middle", (pcs >= 16) & (pcs <= 25)), ("endgame", pcs <= 15)]:
            if msk.sum() > args.dict * 4:
                report(train_sae(emb[msk], args, dev, rng), Fmat[msk], fnames, args.top, lab)
    else:
        report(train_sae(emb, args, dev, rng), Fmat, fnames, args.top, "all")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
