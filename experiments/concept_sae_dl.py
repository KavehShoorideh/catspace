#!/usr/bin/env python
"""experiments/concept_sae_dl.py -- concept discovery with the MAINTAINED SAE library, not a hand-roll
(Kaposi 2026-07-21 "import, don't reinvent"). Uses dictionary_learning (Bloom/Marks et al.) TopK sparse
autoencoder -- fixed-k sparsity (no L1 tuning) + built-in dead-atom revival (auxk). Per Concept Cones
(arXiv 2512.07355), the SAE's dictionary atoms, non-negatively combined by the code, ARE the concept
cones.

CONDITIONING (Kaposi): a plain/global SAE pools over all contexts and averages away context-specific
concepts. Neither the SAE nor Concept Cones handles that. The import-clean fix (no custom architecture)
is to run the SAME maintained SAE PER CONTEXT STRATUM -- `--by-phase` trains a separate dictionary for
opening / middlegame / endgame, so each concept surfaces where it applies (bishop-pair in the open
endgame, king-safety in the middlegame). Named features are a post-hoc mirror (= the CAV/CBM side).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from dictionary_learning.trainers import TopKTrainer
from experiments.concept_features import features as named_features
from experiments.sae_concepts import heatmap


def discover(emb, Pk, Mk, args, dev, rng, label):
    """train the maintained TopK SAE on `emb`, report the concept dictionary for this context."""
    if len(emb) < args.dict * 4:
        print(f"  [{label}] too few positions ({len(emb)}) for dict={args.dict}"); return
    mu, sd = emb.mean(0), emb.std(0) + 1e-8
    X = torch.from_numpy((emb - mu) / sd).float().to(dev)
    steps = args.steps
    tr = TopKTrainer(steps=steps, activation_dim=X.shape[1], dict_size=args.dict, k=args.k, layer=0,
                     lm_name=Path(args.field).stem, device=dev, warmup_steps=max(1, steps // 10), seed=args.seed)
    for step in range(steps):
        tr.update(step, X[torch.from_numpy(rng.integers(0, len(X), size=1024)).to(dev)])
    with torch.no_grad():
        code = tr.ae.encode(X).cpu().numpy()
        ve = float(1 - (tr.ae.decode(tr.ae.encode(X)) - X).pow(2).mean() / X.var())
    alive = np.flatnonzero((code > 1e-6).mean(0) > 0.003)
    feats = [named_features(board_from_packed(Pk[i], Mk[i])) for i in range(len(Pk))]
    fnames = [n for n in feats[0] if not n.endswith("_ctrl")]
    Fmat = np.array([[float(f[n][0]) for n in fnames] for f in feats])
    print(f"  === [{label}] n={len(Pk)} TopK SAE dict={args.dict} k={args.k} var-expl {ve:.2f} alive={len(alive)} ===")
    hits = []
    for j, nm in enumerate(fnames):
        cors = [abs(np.corrcoef(code[:, a], Fmat[:, j])[0, 1]) for a in alive]
        if max(cors) >= 0.30:
            hits.append(f"{nm.replace('_w','')}({max(cors):.2f})")
    print(f"    concepts present: {', '.join(hits)}")
    maxcor = np.array([max(abs(np.corrcoef(code[:, a], Fmat[:, j])[0, 1]) for j in range(len(fnames))) for a in alive])
    for a in alive[np.argsort(maxcor)][:3]:
        print(f"    novel atom {a:3d} (fires {100*(code[:,a]>1e-6).mean():.0f}%): {heatmap(Pk, Mk, np.argsort(-code[:, a]))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="data/derived/sep/lichess_gn_iqeqrl_full.pt")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--dict", type=int, default=128)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--tower", choices=["F", "B"], default="F")
    ap.add_argument("--by-phase", action="store_true", help="condition: a separate SAE dictionary per game phase")
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
    # sample broadly (stratified across phase so endgames are represented), then split by phase if asked
    pool = cand[rng.permutation(len(cand))[:min(len(cand), 120000)]]
    pcnt = np.unpackbits(P[pool].reshape(len(pool), -1).view(np.uint8), axis=1).sum(1)
    take = args.n * (3 if args.by_phase else 1)
    sel = pool[rng.permutation(len(pool))[:take]]; pcs = pcnt[rng.permutation(len(pool))[:take]]
    Pk, Mk = P[sel], M[sel]
    with torch.no_grad():
        t = torch.from_numpy(feature_planes(Pk, Mk)).to(dev)
        emb = (fb.embed_F(t, torch.from_numpy(np.tile(om, (len(Pk), 1))).to(dev)) if args.tower == "F"
               else fb.embed_B(t)).cpu().numpy()
    print(f"VERDICT CONCEPT_SAE_DL field={Path(args.field).stem} tower={args.tower} lib=dictionary_learning "
          f"by_phase={args.by_phase} ({time.time()-t0:.0f}s)")
    if args.by_phase:
        for lab, msk in [("opening pc>=26", pcs >= 26), ("middle pc16-25", (pcs >= 16) & (pcs <= 25)),
                         ("endgame pc<=15", pcs <= 15)]:
            discover(emb[msk], Pk[msk], Mk[msk], args, dev, rng, lab)
    else:
        discover(emb, Pk, Mk, args, dev, rng, "all")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
