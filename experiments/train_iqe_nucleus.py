#!/usr/bin/env python
"""experiments/train_iqe_nucleus.py — the IQE tablebase FOUNDATION on the Lichess
near-mate nucleus (Kaveh 2026-07-19: IQE, nucleus-first). A fresh IQE quasimetric
field trained so distance-to-mate is DTM-ordered on the near-mate core, with
symmetry-invariance and material separation -- the rigid nucleus that later
fine-tunes propagate the far field onto.

Objectives (all AVOID the regression/collapse traps that failed earlier):
  L_rank  = WITHIN-material margin ranking: DTM(i)<DTM(j) => d(F(i),mate)+m < d(F(j),mate)
            (order, not absolute value; within material dodges the centroid-can't-
            order-across-materials wall).
  L_sym   = F(s) = F(horiz-mirror(s))                       symmetry-invariance
  L_sep   = push different-material pairs apart (>margin)   material clusters
The IQE architecture supplies the quasimetric (asymmetry/strata) by construction.

Usage:
  .venv/bin/python experiments/train_iqe_nucleus.py --steps 6000 \
    --data data/derived/lichess_nearmate.npz --out data/derived/sep/iqe_nucleus.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed, encode_meta, encode_packed
from catspace.nn.fb import TorchFB, pick_device, save_ckpt
from catspace.nn.features import feature_planes, omega_ids
from scipy.stats import spearmanr


def planes_of(packed, meta):
    return feature_planes(packed, meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/derived/lichess_nearmate.npz")
    ap.add_argument("--out", default="data/derived/sep/iqe_nucleus.pt")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--sep-margin", type=float, default=10.0)
    ap.add_argument("--w-sym", type=float, default=1.0)
    ap.add_argument("--w-sep", type=float, default=0.3)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = pick_device(args.device)
    torch.manual_seed(args.seed)
    dz = np.load(args.data)
    won = dz["dtm"] > 0
    packed, meta = dz["packed"][won], dz["meta"][won]
    dtm = dz["dtm"][won].astype(np.float32)
    # material key = sorted piece letters (coarse cluster label)
    def matkey(i):
        b = board_from_packed(packed[i], meta[i])
        return "".join(sorted(p.symbol() for p in b.piece_map().values()))
    print(f"[stage] {len(dtm)} won nucleus positions; building material keys...", flush=True)
    mats = np.array([matkey(i) for i in range(len(dtm))])
    uniq = {m: k for k, m in enumerate(sorted(set(mats)))}
    mat = np.array([uniq[m] for m in mats], dtype=np.int64)
    print(f"[stage] {len(uniq)} material classes", flush=True)

    fb = TorchFB(d=args.d, channels=args.channels, blocks=args.blocks, enc_out=args.d,
                 seed=args.seed, iqe=True, iqe_components=32, iqe_embed_scale=2.0,
                 iqe_leak_beta=10.0, spectral_norm=True, omega_free_field=True).to(dev)
    fb.train()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    opt = torch.optim.Adam(fb.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    # mate pole = mean B of the lowest-DTM (mate-in-1/2) nucleus positions
    low = np.argsort(dtm)[:512]

    def embF(idx):
        pl = torch.from_numpy(planes_of(packed[idx], meta[idx])).to(dev)
        o = torch.from_numpy(np.tile(om, (len(idx), 1))).to(dev)
        return fb.embed_F(pl, o)

    def embB(idx):
        return fb.embed_B(torch.from_numpy(planes_of(packed[idx], meta[idx])).to(dev))

    t0 = time.time()
    for step in range(args.steps):
        if step % 500 == 0:
            with torch.no_grad():
                zmate = embB(low).mean(0, keepdim=True).detach()          # (1,d) mate pole
        idx = rng.integers(0, len(dtm), size=args.batch)
        f = embF(idx)
        dm = fb.distance_matrix(f, zmate)[:, 0]                            # d(F(s), mate)
        dtm_b = torch.from_numpy(dtm[idx]).to(dev)
        mat_b = torch.from_numpy(mat[idx]).to(dev)
        # within-material DTM ranking: DTM_i<DTM_j => d_i+margin < d_j
        same = mat_b[:, None] == mat_b[None, :]
        closer = dtm_b[:, None] < dtm_b[None, :]                           # i closer to mate
        mask = same & closer
        diff = dm[None, :] - dm[:, None]                                   # d_j - d_i
        L_rank = torch.relu(args.margin - diff)[mask].mean() if mask.any() else torch.zeros((), device=dev)
        # symmetry
        mir = [board_from_packed(packed[i], meta[i]).transform(chess.flip_horizontal) for i in idx]
        pk_m = np.stack([encode_packed(b) for b in mir]); mt_m = np.stack([encode_meta(b) for b in mir])
        fm = fb.embed_F(torch.from_numpy(feature_planes(pk_m, mt_m)).to(dev),
                        torch.from_numpy(np.tile(om, (len(idx), 1))).to(dev))
        L_sym = ((f - fm) ** 2).sum(1).mean()
        # material separation (push different-material F apart)
        Dff = torch.cdist(f, f)
        L_sep = torch.relu(args.sep_margin - Dff[~same]).pow(2).mean() if (~same).any() else torch.zeros((), device=dev)
        loss = L_rank + args.w_sym * L_sym + args.w_sep * L_sep
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            with torch.no_grad():
                sp = spearmanr(dm.detach().cpu().numpy(), dtm[idx]).correlation
            print(f"  step {step:4d}  L_rank {float(L_rank):.4f} L_sym {float(L_sym):.4f} "
                  f"L_sep {float(L_sep):.4f}  spearman(d,DTM) {sp:+.3f}  ({time.time()-t0:.0f}s)", flush=True)

    fb.eval()
    # held-out spearman, per material
    ev = rng.choice(len(dtm), 1500, replace=False)
    with torch.no_grad():
        de = fb.distance_matrix(embF(ev), embB(low).mean(0, keepdim=True))[:, 0].cpu().numpy()
    overall = spearmanr(de, dtm[ev]).correlation
    save_ckpt(fb, Path(args.out), step=args.steps,
              zgoals={"MATE_W": embB(low).mean(0).detach().cpu()})
    print(f"saved {args.out}")
    print(f"VERDICT IQE_NUCLEUS overall_spearman(d,DTM)={overall:+.3f} "
          f"(centroid on incumbent was ~0/neg; >0.5 = a usable DTM-ordered nucleus)")


if __name__ == "__main__":
    main()
