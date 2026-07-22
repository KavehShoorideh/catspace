#!/usr/bin/env python
"""experiments/sae_concepts.py -- native concept discovery via a SPARSE AUTOENCODER (Kaposi 2026-07-21:
find MANY concept directions natively, no hand-coded features). PCA/ICA validated that concepts are
directions but gave entangled axes; a sparse autoencoder is the standard tool for pulling MONOSEMANTIC
feature-directions out of a learned embedding: an overcomplete dictionary (M >> d) with an L1 sparsity
penalty, so each position is a sparse mix of a few concept atoms and each atom tends to mean ONE thing.

  train SAE on F embeddings -> M dictionary atoms (concept directions) ->
  interpret each ALIVE atom natively by the shared piece-placement of its top-activating positions ->
  check (post-hoc mirror only) which atoms line up with named features (re-discovery) + count novel ones.

Nothing hand-coded enters the discovery; the named features are only a mirror held up afterward.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device
from experiments.concept_features import features as named_features   # VALIDATION MIRROR ONLY


class SAE(nn.Module):
    """tied-ish sparse autoencoder: x -> ReLU(W_e x + b_e) = sparse code -> W_d code (+ b_d) ~ x.
    Decoder atoms (columns of W_d) are unit-normalized each step (standard SAE)."""
    def __init__(self, d, m):
        super().__init__()
        self.enc = nn.Linear(d, m)
        self.dec = nn.Linear(m, d, bias=True)
        self.b_pre = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        code = torch.relu(self.enc(x - self.b_pre))
        return self.dec(code) + self.b_pre, code

    @torch.no_grad()
    def normalize_atoms(self):
        w = self.dec.weight                       # (d, m): columns are atoms
        self.dec.weight.copy_(w / (w.norm(dim=0, keepdim=True) + 1e-8))


def heatmap(Pk, Mk, order, top=40):
    sym = {}
    for i in order[:top]:
        for sq, p in board_from_packed(Pk[i], Mk[i]).piece_map().items():
            sym[(p.symbol(), sq)] = sym.get((p.symbol(), sq), 0) + 1
    hot = sorted(sym.items(), key=lambda kv: -kv[1])[:6]
    return " ".join(f"{s}{chess.square_name(sq)}:{100*c//top}%" for (s, sq), c in hot)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="data/derived/sep/lichess_gn_iqeqrl_full.pt")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--dict", type=int, default=128, help="dictionary size M (overcomplete: M >> d)")
    ap.add_argument("--l1", type=float, default=1e-3, help="sparsity penalty")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--tower", choices=["F", "B"], default="F")
    ap.add_argument("--min-ply", type=int, default=16)
    ap.add_argument("--stratify-phase", action="store_true",
                    help="sample uniformly across piece-count bins so endgame/middlegame concepts are "
                         "represented, not just the ply-16-40 development that dominates raw sampling")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    nz = np.load(args.shard)
    P, M, ply = np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["ply"]).astype(int)
    cand = np.flatnonzero(ply >= args.min_ply)
    if args.stratify_phase:                                        # even coverage across game phases
        pool = cand[rng.permutation(len(cand))[:min(len(cand), 80000)]]
        pcnt = np.unpackbits(P[pool].reshape(len(pool), -1).view(np.uint8), axis=1).sum(1)  # popcount = #pieces
        bins = np.digitize(pcnt, [8, 14, 20, 26]); per = args.n // 5
        idx = np.concatenate([pool[bins == b][:per] for b in range(5)])
        idx = idx[rng.permutation(len(idx))]
    else:
        idx = cand[rng.permutation(len(cand))[:args.n]]
    Pk, Mk = P[idx], M[idx]
    with torch.no_grad():
        t = torch.from_numpy(feature_planes(Pk, Mk)).to(dev)
        emb = (fb.embed_F(t, torch.from_numpy(np.tile(om, (len(Pk), 1))).to(dev)) if args.tower == "F"
               else fb.embed_B(t)).cpu().numpy()
    X = torch.from_numpy((emb - emb.mean(0)) / (emb.std(0) + 1e-8)).float().to(dev)

    sae = SAE(X.shape[1], args.dict).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for step in range(args.steps):
        bi = torch.from_numpy(rng.integers(0, len(X), size=1024)).to(dev)
        recon, code = sae(X[bi])
        loss = (recon - X[bi]).pow(2).mean() + args.l1 * code.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sae.normalize_atoms()
        if step % 1000 == 0 or step == args.steps - 1:
            with torch.no_grad():
                r, c = sae(X)
                ev = 1 - (r - X).pow(2).mean() / X.var()
                l0 = (c > 1e-4).float().sum(1).mean()
            print(f"  step {step:4d}  var-explained {float(ev):.3f}  active/pos {float(l0):.1f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad():
        _, code = sae(X)
    code = code.cpu().numpy()
    alive = np.flatnonzero((code > 1e-4).mean(0) > 0.002)          # atoms used by >0.2% of positions
    # validation mirror: named features (never entered training)
    feats = [named_features(board_from_packed(Pk[i], Mk[i])) for i in range(len(Pk))]
    fnames = [n for n in feats[0] if not n.endswith("_ctrl")]
    Fmat = np.array([[float(f[n][0]) for n in fnames] for f in feats])

    print(f"VERDICT SAE_CONCEPTS field={Path(args.field).stem} tower={args.tower} dict={args.dict} "
          f"alive={len(alive)}")
    print("  named concepts re-discovered (best-matching alive atom |corr|):")
    for j, nm in enumerate(fnames):
        cors = [abs(np.corrcoef(code[:, a], Fmat[:, j])[0, 1]) for a in alive]
        a = alive[int(np.argmax(cors))]
        print(f"      {nm:18s} -> atom {a:3d}  |corr| {max(cors):.2f}  [{'OK' if max(cors)>=0.30 else '--'}]")
    # novel atoms: strong + match no named feature -> new concepts, characterized natively
    maxcor = np.array([max(abs(np.corrcoef(code[:, a], Fmat[:, j])[0, 1]) for j in range(len(fnames)))
                       for a in alive])
    novel = alive[np.argsort(maxcor)][:8]                          # the LEAST named-feature-like atoms
    print(f"  most NOVEL alive atoms (least like any named feature) -- native heatmaps:")
    for a in novel:
        order = np.argsort(-code[:, a])
        print(f"      atom {a:3d} (|corr|<{maxcor[list(alive).index(a)]:.2f}, fires {100*(code[:,a]>1e-4).mean():.0f}%): "
              f"{heatmap(Pk, Mk, order)}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
