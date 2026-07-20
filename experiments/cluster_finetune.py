#!/usr/bin/env python
"""experiments/cluster_finetune.py — encourage CLUSTER FORMATION in the field
(Kaveh 2026-07-19: the field should embed equivalent positions -- symmetry
variants, and shuffles at the same distance-to-mate -- close, so the subgoal
planner can jump cluster to cluster). The incumbent has NO such structure
(mirror not closer than random; within-DTM = between-DTM). Fine-tune F with:

  L_sym    = || F(pos) - F(horiz-mirror(pos)) ||^2      symmetry-invariance
  L_clust  = pull same-DTM(+material) pairs together, push different-DTM apart
  L_anchor = || F(pos) - F_frozen(pos) ||^2             keep conversion structure

Measures symmetry-invariance + DTM-clustering (within/between ratio) before/after.

Usage:
  .venv/bin/python experiments/cluster_finetune.py --steps 1500 \
    --out data/derived/sep/cert_base_cluster.pt
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
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device, save_ckpt


def planes_of(boards):
    pk = np.stack([encode_packed(b) for b in boards])
    mt = np.stack([encode_meta(b) for b in boards])
    return feature_planes(pk, mt)


def cluster_metrics(fb, boards, dtm, om, dev):
    """symmetry-invariance ratio + DTM within/between ratio (higher=more clustered)."""
    with torch.no_grad():
        pl = torch.from_numpy(planes_of(boards)).to(dev)
        o = torch.from_numpy(np.tile(om, (len(boards), 1))).to(dev)
        emb = fb.embed_F(pl, o).cpu().numpy()
        mir = [b.transform(chess.flip_horizontal) for b in boards]
        embm = fb.embed_F(torch.from_numpy(planes_of(mir)).to(dev), o).cpu().numpy()
    perm = np.random.default_rng(0).permutation(len(emb))
    d_sym = np.linalg.norm(emb - embm, axis=1).mean()
    d_rand = np.linalg.norm(emb - emb[perm], axis=1).mean()
    within, between = [], []
    rng = np.random.default_rng(1)
    for k in np.unique(dtm):
        ii = np.flatnonzero(dtm == k)
        if len(ii) < 2:
            continue
        for a in ii[:30]:
            oth = ii[ii != a]
            within.append(np.linalg.norm(emb[a] - emb[rng.choice(oth)]))
            dif = np.flatnonzero(dtm != k)
            between.append(np.linalg.norm(emb[a] - emb[rng.choice(dif)]))
    return d_rand / d_sym, np.mean(between) / np.mean(within)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--out", default="data/derived/sep/cert_base_cluster.pt")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-sym", type=float, default=1.0)
    ap.add_argument("--w-clust", type=float, default=1.0)
    ap.add_argument("--w-anchor", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = pick_device(args.device)
    torch.manual_seed(args.seed)
    fb, pay = load_ckpt(Path(args.ckpt), dev); fb.train()
    fb_frozen, _ = load_ckpt(Path(args.ckpt), dev); fb_frozen.eval()
    for p in fb_frozen.parameters():
        p.requires_grad_(False)
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    dz = np.load(args.dtm_npz)
    dtm_all, mat_all = dz["dtm"].astype(np.float32), dz["material"]
    rng = np.random.default_rng(args.seed)

    # eval set (held-out, krvk for the symmetry check clarity)
    ev = np.flatnonzero(mat_all == 2)[:400]
    ev_boards = [board_from_packed(dz["packed"][i], dz["meta"][i]) for i in ev]
    sym0, clu0 = cluster_metrics(fb_frozen, ev_boards, dtm_all[ev], om, dev)
    print(f"[before] symmetry_ratio={sym0:.2f} (mirror vs random; >1 better)  "
          f"dtm_clustering={clu0:.2f} (between/within; >1 better)", flush=True)

    opt = torch.optim.Adam(fb.parameters(), lr=args.lr)
    t0 = time.time()
    for step in range(args.steps):
        idx = rng.integers(0, len(dtm_all), size=args.batch)
        boards = [board_from_packed(dz["packed"][i], dz["meta"][i]) for i in idx]
        pl = torch.from_numpy(planes_of(boards)).to(dev)
        o = torch.from_numpy(np.tile(om, (len(boards), 1))).to(dev)
        f = fb.embed_F(pl, o)
        # symmetry
        mir = [b.transform(chess.flip_horizontal) for b in boards]
        fm = fb.embed_F(torch.from_numpy(planes_of(mir)).to(dev), o)
        L_sym = ((f - fm) ** 2).sum(1).mean()
        # anchor
        with torch.no_grad():
            f0 = fb_frozen.embed_F(pl, o)
        L_anchor = ((f - f0) ** 2).sum(1).mean()
        # DTM clustering: pull same-dtm+material close, push diff apart (margin)
        dtm_b = torch.from_numpy(dtm_all[idx]).to(dev)
        mat_b = torch.from_numpy(mat_all[idx].astype(np.int64)).to(dev)
        D = torch.cdist(f, f)                                   # (B,B)
        same = (dtm_b[:, None] == dtm_b[None, :]) & (mat_b[:, None] == mat_b[None, :])
        same.fill_diagonal_(False)
        diff = ~same
        diff.fill_diagonal_(False)
        L_clust = (D[same] ** 2).mean() if same.any() else torch.zeros((), device=dev)
        L_clust = L_clust + torch.relu(args.margin - D[diff]).pow(2).mean()
        loss = args.w_sym * L_sym + args.w_clust * L_clust + args.w_anchor * L_anchor
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  L_sym {float(L_sym):.4f} L_clust {float(L_clust):.4f} "
                  f"L_anchor {float(L_anchor):.4f}  ({time.time()-t0:.0f}s)", flush=True)

    fb.eval()
    sym1, clu1 = cluster_metrics(fb, ev_boards, dtm_all[ev], om, dev)
    save_ckpt(fb, Path(args.out), step=pay.get("step", 0), zgoals=pay.get("zgoals"),
              provenance=pay.get("provenance"))
    print(f"saved {args.out}")
    print(f"VERDICT CLUSTER symmetry_ratio {sym0:.2f}->{sym1:.2f}  "
          f"dtm_clustering {clu0:.2f}->{clu1:.2f}  (both >1 and rising => clusters formed)")


if __name__ == "__main__":
    main()
