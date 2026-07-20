#!/usr/bin/env python
"""experiments/train_board_policy.py — a BOARD policy net that predicts the
TABLEBASE-OPTIMAL move (Kaveh 2026-07-19 autonomous). Value-navigation caps at
~0.55 for KRRvKBP (DTM regression is hard, 0.29). Predicting the best MOVE is a
ranking problem -- report top-1/top-3 accuracy per material; if high, MCTS with
this policy as priors should convert near-optimally (AlphaZero recipe).

Usage:
  .venv/bin/python experiments/train_board_policy.py --epochs 60 \
    --out data/derived/sep/board_policy.pt
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

from catspace.nn.encoder import BoardEncoder
from catspace.nn.features import feature_planes
from catspace.nn.fb import pick_device


class BoardPolicy(nn.Module):
    def __init__(self, channels=64, blocks=6, seed=0):
        super().__init__()
        self.enc = BoardEncoder(in_planes=20, channels=channels, blocks=blocks,
                                out_dim=256, seed=seed)
        self.head = nn.Linear(256, 4096)

    def forward(self, planes):
        return self.head(self.enc(planes))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/derived/dtm_policy.npz")
    ap.add_argument("--out", default="data/derived/sep/board_policy.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = pick_device(args.device)
    torch.manual_seed(args.seed)
    dz = np.load(args.data)
    y = dz["move_idx"].astype(np.int64); mat = dz["material"]
    keep = y >= 0
    planes = feature_planes(dz["packed"][keep], dz["meta"][keep]); y = y[keep]; mat = mat[keep]
    print(f"[stage] planes {planes.shape}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(planes)); nval = len(planes) // 6
    val, tr = perm[:nval], perm[nval:]
    Xtr, ytr = torch.from_numpy(planes[tr]), torch.from_numpy(y[tr])
    Xva, yva = torch.from_numpy(planes[val]).to(dev), y[val]

    net = BoardPolicy(args.channels, args.blocks, args.seed).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(args.seed)
    bs = 512
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        idx_all = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), bs):
            idx = idx_all[i:i + bs]
            loss = nn.functional.cross_entropy(net(Xtr[idx].to(dev)), ytr[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                lg = net(Xva).cpu().numpy()
            top1 = float((lg.argmax(1) == yva).mean())
            print(f"  ep {ep:3d}  top1 {top1:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    net.eval()
    with torch.no_grad():
        lg = net(Xva).cpu().numpy()
    top1 = lg.argmax(1) == yva
    top3 = np.array([yva[i] in lg[i].argsort()[-3:] for i in range(len(yva))])
    names = {0: "krrkbp", 1: "krrvk", 2: "krvk"}
    percm = {names[int(m)]: round(float(top1[mat[val] == m].mean()), 3) for m in np.unique(mat[val])}
    torch.save({"state": net.state_dict(), "channels": args.channels, "blocks": args.blocks}, args.out)
    print(f"saved {args.out}")
    print(f"VERDICT BOARD_POLICY top1={top1.mean():.3f} top3={top3.mean():.3f} per_material={percm} "
          f"(high top1 => MCTS with this prior should convert near-optimally)")


if __name__ == "__main__":
    main()
