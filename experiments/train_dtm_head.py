#!/usr/bin/env python
"""experiments/train_dtm_head.py — a DECOUPLED distance-to-mate head (Kaveh
2026-07-19, autonomous): the metric-coupled DTM hinge failed (fights QRL,
collapses). The literature (NN tablebase DTM approximation; ~85% on KRvK) uses a
plain SUPERVISED regression head instead. Predict dtm(s) from F(s) with a small
MLP + robust loss. Unlike the flat committor P(win), a DTM head has a GRADIENT
toward mate -- exactly what conversion needs. Navigate by minimizing it.

Trains on gen_dtm_data.py (+ optional forced-mate). Reports held-out Spearman
(the alignment the metric hinge never reached) per material. Saves the head.

Usage:
  .venv/bin/python experiments/train_dtm_head.py --ckpt data/derived/sep/cert_base_full.pt \
    --out data/derived/sep/cert_base_full_dtmhead.pt --epochs 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device


class DTMHead(nn.Module):
    def __init__(self, d_in, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1), nn.Softplus())  # dtm >= 0

    def forward(self, f):
        return self.net(f).squeeze(-1)


def embed_F_all(fb, packed, meta, dev, bs=4096):
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    out = []
    for i in range(0, len(packed), bs):
        pl = feature_planes(packed[i:i + bs], meta[i:i + bs])
        o = np.tile(om, (len(pl), 1))
        with torch.no_grad():
            out.append(fb.embed_F(torch.from_numpy(pl).to(dev),
                                  torch.from_numpy(o).to(dev)).cpu())
    return torch.cat(out)


def main():
    from scipy.stats import spearmanr
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--out", default="data/derived/sep/cert_base_full_dtmhead.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = pick_device(args.device)
    torch.manual_seed(args.seed)
    fb, pay = load_ckpt(Path(args.ckpt), dev); fb.eval()
    dz = np.load(args.dtm_npz)
    dtm = dz["dtm"].astype(np.float32); mat = dz["material"]
    t0 = time.time()
    F = embed_F_all(fb, dz["packed"], dz["meta"], dev)               # (N, d) cached embeds
    print(f"[stage] embedded {len(F)} F in {time.time()-t0:.1f}s (d={F.shape[1]})")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(F))
    n_val = len(F) // 6
    val, tr = perm[:n_val], perm[n_val:]
    Xtr, ytr = F[tr].to(dev), torch.from_numpy(dtm[tr]).to(dev)
    Xva, yva = F[val].to(dev), torch.from_numpy(dtm[val]).to(dev)

    head = DTMHead(F.shape[1], args.hidden).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    bs = 2048
    for ep in range(args.epochs):
        head.train()
        for i in range(0, len(Xtr), bs):
            idx = torch.randperm(len(Xtr), generator=g)[i:i + bs]
            pred = head(Xtr[idx])
            loss = nn.functional.smooth_l1_loss(pred, ytr[idx])       # robust to the long DTM tail
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            head.eval()
            with torch.no_grad():
                pv = head(Xva).cpu().numpy()
            sp = spearmanr(pv, yva.cpu().numpy()).correlation
            print(f"  ep {ep:3d}  val smooth_l1 {float(nn.functional.smooth_l1_loss(torch.from_numpy(pv), yva.cpu())):.3f}  "
                  f"spearman {sp:+.3f}", flush=True)

    head.eval()
    with torch.no_grad():
        pv = head(Xva).cpu().numpy()
    yv = yva.cpu().numpy(); mv = mat[val]
    names = {0: "krrkbp", 1: "krrvk", 2: "krvk"}
    overall = spearmanr(pv, yv).correlation
    torch.save({"state": head.state_dict(), "d_in": F.shape[1], "hidden": args.hidden}, args.out)
    print(f"saved {args.out}")
    percm = {names[int(m)]: round(float(spearmanr(pv[mv == m], yv[mv == m]).correlation), 3)
             for m in np.unique(mv)}
    print(f"VERDICT DTM_HEAD overall_spearman={overall:+.3f} per_material={percm} "
          f"(vs centroid ~0, composed ~+0.1 -- higher=usable navigation signal)")


if __name__ == "__main__":
    main()
