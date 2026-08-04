#!/usr/bin/env python
"""catspace/research/components/planner/approaches/endgame_groundtruth/experiments/train_board_dtm.py — a BOARD-based distance-to-mate net (Kaveh
2026-07-19 autonomous). The DTM head on the incumbent F plateaued at spearman
0.227 -- F is DTM-poor. This trains a fresh ConvNet (BoardEncoder) DIRECTLY on
the board planes -> DTM, so it is NOT bottlenecked by F. Literature: NNs
approximate endgame tablebase DTM well (KRK ~85%). If this predicts DTM sharply,
navigating by it finds the DTM-navigation CEILING (does DTM nav beat the
committor's 0.567 when the predictor is good?).

Usage:
  .venv/bin/python catspace/research/components/planner/approaches/endgame_groundtruth/experiments/train_board_dtm.py --epochs 80 \
    --out data/derived/sep/board_dtm.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


from catspace.research.components.encoder.approaches.jepa_tokenizer.src.encoder import BoardEncoder
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
from catspace.io import paths


class BoardDTM(nn.Module):
    def __init__(self, channels=64, blocks=6, seed=0):
        super().__init__()
        self.enc = BoardEncoder(in_planes=20, channels=channels, blocks=blocks,
                                out_dim=128, seed=seed)
        self.head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1), nn.Softplus())

    def forward(self, planes):
        return self.head(self.enc(planes)).squeeze(-1)


def main():
    from scipy.stats import spearmanr
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtm-npz", default=paths.derived("dtm_endgame.npz"))
    ap.add_argument("--out", default=paths.sep("board_dtm.pt"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = pick_device(args.device)
    torch.manual_seed(args.seed)
    dz = np.load(args.dtm_npz)
    dtm = dz["dtm"].astype(np.float32); mat = dz["material"]
    t0 = time.time()
    planes = feature_planes(dz["packed"], dz["meta"])                # (N,20,8,8)
    print(f"[stage] planes {planes.shape} in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(planes))
    nval = len(planes) // 6
    val, tr = perm[:nval], perm[nval:]
    Xtr = torch.from_numpy(planes[tr]); ytr = torch.from_numpy(dtm[tr])
    Xva = torch.from_numpy(planes[val]).to(dev); yva = dtm[val]

    net = BoardDTM(args.channels, args.blocks, args.seed).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(args.seed)
    bs = 512
    for ep in range(args.epochs):
        net.train()
        idx_all = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), bs):
            idx = idx_all[i:i + bs]
            pred = net(Xtr[idx].to(dev))
            loss = nn.functional.smooth_l1_loss(pred, ytr[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                pv = net(Xva).cpu().numpy()
            print(f"  ep {ep:3d}  spearman {spearmanr(pv, yva).correlation:+.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    net.eval()
    with torch.no_grad():
        pv = net(Xva).cpu().numpy()
    names = {0: "krrkbp", 1: "krrvk", 2: "krvk"}
    mvv = mat[val]
    percm = {names[int(m)]: round(float(spearmanr(pv[mvv == m], yva[mvv == m]).correlation), 3)
             for m in np.unique(mvv)}
    torch.save({"state": net.state_dict(), "channels": args.channels, "blocks": args.blocks}, args.out)
    print(f"saved {args.out}")
    print(f"VERDICT BOARD_DTM overall_spearman={spearmanr(pv, yva).correlation:+.3f} "
          f"per_material={percm} (F-head plateaued 0.227; >0.6 => navigable)")


if __name__ == "__main__":
    main()
